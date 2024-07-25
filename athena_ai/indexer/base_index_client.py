#!/usr/bin/env python
# coding: utf-8

import os
import logging
from typing import Dict, Any, List
import shutil

from basedir import basedir

from unstructured.file_utils.filetype import FileType, detect_filetype
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from athena_ai.client import AthenaClient
from athena_ai.indexer.splitters import code_splitter, text_splitter
from athena_ai.logger import logger

EMBEDDING_MODEL: str = "text-embedding-3-large"
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")


def summarize_file(content: str):
    client = AthenaClient(id="id", model_name="gpt-3.5-turbo-16k")
    response = client.base_prompt(
        """
            Describe and summarize what this document says.
            Be very specific.
            Everything must be documented.
            Keep it very short and concise, this will be used for labeling a vector search.
            """,
        content,
    )
    return response


# def extract_functions(content: str, file_type: str):
#         client = AthenaClient(id='id', model_name="gpt-3.5-turbo-16k")
#         response = client.base_prompt(
#             f"""
#             Describe what each function in this {file_type} code does.
#             Be very specific.
#             Every function must be documented.
#             If there are no actual functions then return "None"
#             """,
#             content,
#         )
#         return response

chunk_size: int = 2000
chunk_overlap: int = 1000


class BaseIndexClient(object):
    storage_type: str = "local"  # local or gcs
    id: str = ""
    name: str = ""
    version: str = ""

    splited_docs: List[str] = []
    splited_metadatas: List[str] = []

    def __init__(
        cls, storage_type: str, id: str, dir: str, name: str, version: str = "v1"
    ) -> None:
        cls.storage_type = storage_type
        cls.id = id
        cls.name = name
        cls.version = version
        cls.base_path: str = os.path.join(basedir, dir)
        cls.name_path: str = os.path.join(cls.base_path, cls.name)
        cls.name_version_path: str = os.path.join(
            cls.base_path, f"{cls.name}-{cls.version}"
        )
        os.makedirs(cls.name_version_path, exist_ok=True)
        cls.splited_docs: List[str] = []
        cls.splited_metadatas: List[str] = []
        return

    def copy(cls, source: str, destination: str, is_dir: bool = False):
        if is_dir:
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, destination)

    def clean(cls, root: str) -> Dict[str, Any]:
        logger.info(f"CLEAN: {root}")
        all_files = []
        ignore_folders = [".git"]
        logger.info("Finding all files in the root folder...")
        for path, subdirs, files in os.walk(root):
            for name in files:
                folder_path = os.path.join(path, name)
                flag = 0
                for folder in ignore_folders:
                    if folder in folder_path:
                        flag = 1
                        break
                if flag != 1:
                    all_files.append(os.path.join(path, name))

        logger.info("Finding unknown file types...")
        unknown_files = []
        for file in all_files:
            if detect_filetype(file).value == 0:
                unknown_files.append(file)

        logger.info("Renaming unknown file types to .txt...")
        for file in unknown_files:
            new_name = file + ".txt"
            os.rename(file, new_name)

        logger.info("Finding all json files...")
        json_files = []
        for file in all_files:
            if detect_filetype(file).value == FileType.JSON.value:
                json_files.append(file)

        logger.info("Renaming json files to .txt...")
        for file in json_files:
            new_name = file + ".txt"
            os.rename(file, new_name)

        logger.info("Creating dictionary mapping file names to file paths...")

    def prepare(cls, root: str, full: bool = False):
        logger.info(f"PREPARE: {root}")
        loader = DirectoryLoader(root, silent_errors=False, recursive=True)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = doc.metadata["source"].strip(".txt")

        logger.info(f"DOCS: {len(docs)}")
        for doc in docs:
            language = None
            file_summary = None
            functions = None
            file_name: str = doc.metadata["source"]
            logger.info(file_name)

            if ".cpp" in file_name or ".h" in file_name:
                file_type = "cpp"
                language = Language.CPP
            elif ".js" in file_name:
                file_type = "js"
                language = Language.JS
            elif ".ts" in file_name:
                file_type = "ts"
                language = Language.TS
            elif ".py" in file_name:
                file_type = "py"
                language = Language.PYTHON
            else:
                file_type = "text"

            if language:
                # file_summary = summarize_file(doc.page_content) if full else None
                # functions = extract_functions(doc.page_content, file_type) if full else None
                splitter: RecursiveCharacterTextSplitter = code_splitter(
                    language,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            else:
                splitter: RecursiveCharacterTextSplitter = text_splitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            splits = splitter.split_text(doc.page_content)
            for index, split in enumerate(splits):
                if split.strip():
                    chunk_metadata = {
                        "source": file_name.split("/")[-1],
                        "file_type": file_type,
                        "chunk_index": index,
                        "total_chunks": len(splits),
                    }
                    if file_summary:
                        chunk_metadata["file_summary"] = file_summary
                    if functions:
                        chunk_metadata["functions"] = functions

                    cls.splited_docs.append(split)
                    cls.splited_metadatas.append(chunk_metadata)

    def build(cls):
        embedder = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            chunk_size=chunk_size,
        )

        return FAISS.from_documents(
            cls.splited_docs, embedding=embedder, metadatas=cls.splited_metadatas
        )

    def build_batch(cls, paths: List[str], full: bool = False):
        for path in paths:
            cls.clean(path)
            cls.prepare(path, full)

        embedder = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            chunk_size=chunk_size,
        )
        logger.info(len(cls.splited_docs))
        logger.info(len(cls.splited_metadatas))
        logger.debug(cls.splited_docs)
        return FAISS.from_texts(
            cls.splited_docs, embedding=embedder, metadatas=cls.splited_metadatas
        )

    def save(
        cls,
        store: FAISS = None,
    ):
        if cls.storage_type == "local":
            store.save_local(cls.name_version_path)
