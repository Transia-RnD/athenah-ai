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

from athena_ai.indexer.splitters import code_splitter, text_splitter

EMBEDDING_MODEL: str = "text-embedding-3-small"
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")

logger = logging.getLogger("app")


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
        cls.splited_docs: List[str] = []
        cls.splited_metadatas: List[str] = []

    def copy(cls, source: str, destination: str, is_dir: bool = False):
        if is_dir:
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, destination)

    def clean(cls, root: str) -> Dict[str, Any]:
        logger.debug(f"CLEAN: {root}")
        all_files = []
        ignore_folders = [".git"]
        logger.debug("Finding all files in the root folder...")
        for path, _, files in os.walk(root):
            for name in files:
                folder_path = os.path.join(path, name)
                flag = 0
                for folder in ignore_folders:
                    if folder in folder_path:
                        flag = 1
                        break
                if flag != 1:
                    all_files.append(os.path.join(path, name))
        logger.debug("Finding unknown file types...")
        unknown_files = []
        for file in all_files:
            if detect_filetype(file).value == 0:
                unknown_files.append(file)

        logger.debug("Renaming unknown file types to .txt...")
        for file in unknown_files:
            new_name = file + ".txt"
            os.rename(file, new_name)

        logger.debug("Finding all json files...")
        json_files = []
        for file in all_files:
            if detect_filetype(file).value == FileType.JSON.value:
                json_files.append(file)

        logger.debug("Renaming json files to .txt...")
        for file in json_files:
            new_name = file + ".txt"
            os.rename(file, new_name)

        logger.debug("Creating dictionary mapping file names to file paths...")

    def prepare(cls, root: str):
        logger.debug(f"PREPARE: {root}")
        loader = DirectoryLoader(root, silent_errors=False, recursive=True)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = doc.metadata["source"].strip(".txt")

        logger.debug(f"DOCS: {len(docs)}")
        for doc in docs:
            file_name = doc.metadata["source"]
            logger.debug(file_name)
            if ".cpp" in file_name or ".h" in file_name:
                file_name = file_name + "\n\n"
                splitter: RecursiveCharacterTextSplitter = code_splitter(
                    Language.CPP,
                    chunk_size=2000,
                    chunk_overlap=20,
                )
                # For Documents
                # splits = splitter.create_documents([doc.page_content], [doc.metadata])
                splits = splitter.split_text(doc.page_content)
            else:
                file_name = file_name + "\n\n"
                splitter: RecursiveCharacterTextSplitter = text_splitter(
                    chunk_size=2000,
                    chunk_overlap=20,
                )
                # For Documents
                # splits = splitter.create_documents([doc.page_content], [doc.metadata])
                splits = splitter.split_text(doc.page_content)

            # For Documents
            # cls.splited_docs.extend(splits)

            splits = [file_name + split for split in splits]
            cls.splited_docs.extend(splits)
            cls.splited_metadatas.extend([doc.metadata] * len(splits))

    def build(cls):
        embedder = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            chunk_size=1000,
        )

        return FAISS.from_documents(
            cls.splited_docs, embedding=embedder, metadatas=cls.splited_metadatas
        )

    def build_batch(cls, paths: List[str]):
        for path in paths:
            cls.clean(path)
            cls.prepare(path)

        # Build All Indexes
        embedder = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            chunk_size=1000,
        )
        logger.debug(cls.splited_docs)
        return FAISS.from_texts(
            cls.splited_docs, embedding=embedder, metadatas=cls.splited_metadatas
        )
        # For Documents
        # return FAISS.from_documents(cls.splited_docs, embedding=embedder)

    def save(
        cls,
        store: FAISS = None,
    ):
        if cls.storage_type == "local":
            store.save_local(cls.name_version_path)
