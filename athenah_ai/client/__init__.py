#!/usr/bin/env python
# coding: utf-8


import os
from typing import Dict, Any, List

from dotenv import load_dotenv

import openai

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from athenah_ai.client.vector_store import VectorStore
from athenah_ai.logger import logger

load_dotenv()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
OPENAI_API_MODEL: str = "gpt-4o-mini"
MAX_TOKENS: int = 2000


def get_token_total(prompt: str) -> int:
    import tiktoken

    openai_model = OPENAI_API_MODEL
    encoding = tiktoken.encoding_for_model(openai_model)
    return len(encoding.encode(prompt))


class AthenaClient(VectorStore):
    """
    A client for interacting with the Athena AI chat model.

    Attributes:
        id (str): The ID of the client.
        model_group (str): The model group to use for the chat model.
        custom_model (str): The custom model to use for the chat model.
        version (str): The version of the chat model.
        model_name (str): The name of the chat model.
        temperature (float): The temperature parameter for generating responses.
        max_tokens (int): The maximum number of tokens for generating responses.
        top_p (int): The top-p parameter for generating responses.
        best_of (int): The best-of parameter for generating responses.
        frequency_penalty (float): The frequency penalty parameter for generating
        responses.
        presence_penalty (float): The presence penalty parameter for generating
        responses.
        stop (List[str]): The list of stop words for generating responses.
        has_history (bool): Whether the client has chat history.
        chat_history (List[str]): The chat history of the client.
        db (FAISS): The FAISS vector store for document retrieval.
    """

    id: str = ""
    model_group: str = "dist"
    custom_model: str = ""
    version: str = "v1"
    model_name: str = OPENAI_API_MODEL
    temperature: float = 0
    max_tokens: int = 600
    top_p: int = 1
    best_of: int = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
    stop: List[str] = []

    has_history: bool = False
    chat_history: List[str] = []
    db: FAISS = None

    def __init__(
        cls,
        id: str,
        model_group: str = "dist",
        custom_model: str = "",
        version: str = "v1",
        model_name: str = OPENAI_API_MODEL,
        temperature: float = 0,
        max_tokens: int = 1200,
        top_p: int = 1,
        best_of: int = 3,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop: List[str] = [],
    ):
        """
        Initializes the AthenaClient.

        Args:
            id (str): The ID of the client.
            model_group (str): The model group to use for the chat model.
            custom_model (str): The custom model to use for the chat model.
            version (str): The version of the chat model.
            model_name (str): The name of the chat model.
            temperature (float): The temperature parameter for generating responses.
            max_tokens (int): The maximum number of tokens for generating responses.
            top_p (int): The top-p parameter for generating responses.
            best_of (int): The best-of parameter for generating responses.
            frequency_penalty (float): The frequency penalty parameter for generating
            responses.
            presence_penalty (float): The presence penalty parameter for generating
            responses.
            stop (List[str]): The list of stop words for generating responses.
        """
        cls.id = id
        cls.model_group = model_group
        cls.custom_model = custom_model
        cls.version = version
        cls.model_name = model_name
        cls.temperature = temperature
        cls.max_tokens = max_tokens
        cls.top_p = top_p
        cls.best_of = best_of
        cls.frequency_penalty = frequency_penalty
        cls.presence_penalty = presence_penalty
        cls.stop = stop

        super().__init__(storage_type="local" if model_group == "dist" else "gcs")

        if cls.model_group and cls.custom_model:
            cls.db = cls.load(cls.custom_model, cls.model_group, cls.version)

        pass

    def prompt(cls, prompt: str) -> str:
        """
        Generates a response to the given prompt.

        Args:
            prompt (str): The prompt to generate a response to.

        Returns:
            str: The generated response.
        """
        cls.openai = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=cls.model_name,
            temperature=cls.temperature,
            max_tokens=MAX_TOKENS + get_token_total(prompt),
            n=cls.best_of,
            model_kwargs={
                "top_p": cls.top_p,
                "frequency_penalty": cls.frequency_penalty,
                "presence_penalty": cls.presence_penalty,
            },
        )

        num_indexs = cls.db.index_to_docstore_id
        logger.info(f"DB INDEXS: {len(num_indexs)}")
        retriever = cls.db.as_retriever()
        # similar_docs = cls.db.similarity_search_with_relevance_scores("ripple", k=3)
        # print(similar_docs)

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        question_answer_chain = create_stuff_documents_chain(
            cls.openai, retrieval_qa_chat_prompt
        )
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": prompt})
        return response["answer"]

    def base_prompt(cls, system: str = None, prompt: str = None) -> str:
        """
        Generates a response to the given system and prompt.

        Args:
            system (str): The system message.
            prompt (str): The user prompt.

        Returns:
            str: The generated response.
        """
        try:
            messages: List[Dict[str, Any]] = []
            if isinstance(system, str) and system != "":
                messages.append({"role": "system", "content": system})
            if isinstance(prompt, str) and prompt != "":
                messages.append({"role": "user", "content": prompt})

            response = openai.chat.completions.create(
                model=cls.model_name,
                messages=messages,
                temperature=cls.temperature,
                max_tokens=cls.max_tokens,
                top_p=cls.top_p,
                n=cls.best_of,
                frequency_penalty=cls.frequency_penalty,
                presence_penalty=cls.presence_penalty,
            )
            assistant_reply = response.choices[0].message.content
            return assistant_reply
        except Exception as e:
            raise ValueError(f"failed to generate a prompt completion: {str(e)}")
