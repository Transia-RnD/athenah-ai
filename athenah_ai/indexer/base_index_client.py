#!/usr/bin/env python
# coding: utf-8

import os
from typing import Dict, Any, List
import shutil
import faiss
import pickle

from basedir import basedir

from unstructured.file_utils.filetype import FileType, detect_filetype
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from google.cloud.storage.bucket import Bucket, Blob
from athenah_ai.libs.google.storage import GCPStorageClient

from athenah_ai.client import AthenahClient
from athenah_ai.indexer.splitters import code_splitter, text_splitter
from athenah_ai.logger import logger

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL")
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", 2000))
GCP_INDEX_BUCKET: str = os.environ.get("GCP_INDEX_BUCKET", "athenah-ai-indexes")
chunk_overlap: int = 0


def summarize_file(content: str):
    client = AthenahClient(id="id", model_name="gpt-3.5-turbo-16k")
    response = client.base_prompt(
        """
            Describe and summarize what this document says.
            Be very specific.
            Everything must be documented.
            Keep it very short and concise,
            this will be used for labeling a vector search.
            """,
        content,
    )
    return response


# def extract_functions(content: str, file_type: str):
#         client = AthenahClient(id='id', model_name="gpt-3.5-turbo-16k")
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
        if cls.storage_type == "gcs":
            cls.storage_client: GCPStorageClient = GCPStorageClient().add_client()
            cls.bucket: Bucket = cls.storage_client.init_bucket(GCP_INDEX_BUCKET)
        pass

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
                splitter: RecursiveCharacterTextSplitter = code_splitter(
                    language,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=chunk_overlap,
                )
            else:
                splitter: RecursiveCharacterTextSplitter = text_splitter(
                    chunk_size=CHUNK_SIZE,
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
                    # Save split to file
                    split_file_path = os.path.join(
                        cls.name_version_path, f"split_{index}.txt"
                    )
                    with open(split_file_path, "w") as split_file:
                        split_file.write(split)

    def build_one(cls):
        embedder = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
        )

        return FAISS.from_texts(
            cls.splited_docs, embedding=embedder, metadatas=cls.splited_metadatas
        )

    def build_batch(cls, paths: List[str], full: bool = False):
        for path in paths:
            cls.clean(path)
            cls.prepare(path, full)

        embedder = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
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
            logger.info("SAVING LOCAL FAISS")
            store.save_local(cls.name_version_path)
            return

        if cls.storage_type == "gcs":
            logger.info("SAVING GCS FAISS")
            data_byte_array = pickle.dumps((store.docstore, store.index_to_docstore_id))
            blob: Blob = cls.bucket.blob(f"{cls.name}/{cls.version}/index.pkl")
            blob.upload_from_string(data_byte_array)
            temp_file_name = "/tmp/index.faiss"
            faiss.write_index(store.index, temp_file_name)
            blob: Blob = cls.bucket.blob(f"{cls.name}/{cls.version}/index.faiss")
            blob.upload_from_filename(temp_file_name)
            return
