#!/usr/bin/env python
# coding: utf-8

import os
import logging
from typing import Dict, Any, List

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
import openai

openai.api_key = OPENAI_API_KEY
MAX_TOKENS: int = 2000

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from athena_ai.client.vector_store import VectorStore

logger = logging.getLogger("app")


def get_token_total(prompt: str) -> int:
    import tiktoken

    openai_model = "gpt-4"
    encoding = tiktoken.encoding_for_model(openai_model)
    return len(encoding.encode(prompt))


class AthenaClient(VectorStore):
    id: str = ""
    model_group: str = "dist"
    custom_model: str = ""
    version: str = "v1"
    model_name: str = "gpt-4"
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
        model_name: str = "gpt-4",
        temperature: float = 0,
        max_tokens: int = 1200,
        top_p: int = 1,
        best_of: int = 3,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop: List[str] = [],
    ):
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

    def prompt(cls, prompt: str):
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
        logger.debug(f"DB INDEXS: {len(num_indexs)}")
        retriever = cls.db.as_retriever()

        system_prompt = "{context}"
        _prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(cls.openai, _prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": prompt})
        return response["answer"]

    def base_prompt(cls, system: str = None, prompt: str = None):
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
