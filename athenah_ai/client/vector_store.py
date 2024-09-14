#!/usr/bin/env python
# coding: utf-8

import os
from io import BytesIO
import faiss
import pickle

from basedir import basedir

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from cachetools import cached, TTLCache

from google.cloud.storage.bucket import Bucket
from athenah_ai.libs.google.storage import GCPStorageClient
from athenah_ai.logger import logger

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL")
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", 2000))
GCP_INDEX_BUCKET: str = os.environ.get("GCP_INDEX_BUCKET", "athenah-ai-indexes")

cache = TTLCache(maxsize=100, ttl=3600)


class VectorStore(object):
    storage_type: str = "local"  # local or gcs

    def __init__(cls, storage_type: str) -> None:
        cls.storage_type = storage_type
        pass

    def load(cls, name: str, dir: str = "dist", version: str = "v1") -> FAISS:
        if cls.storage_type == "local":
            logger.info("LOADING LOCAL FAISS")
            return cls.load_local(
                dir,
                name,
                version,
            )

        if cls.storage_type == "gcs":
            logger.info("LOADING GCS FAISS")
            try:
                return cls.load_local(
                    dir,
                    name,
                    version,
                )
            except Exception:
                cls.storage_client: GCPStorageClient = GCPStorageClient().add_client()
                cls.bucket: Bucket = cls.storage_client.init_bucket(GCP_INDEX_BUCKET)
                return cls.load_gcs(
                    name,
                    version,
                )

    def load_local(cls, dir: str, name: str, version: str) -> FAISS:
        embedder = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
        )
        cls.base_path: str = os.path.join(basedir, dir)
        cls.name_path: str = os.path.join(cls.base_path, name)
        cls.name_version_path: str = os.path.join(cls.base_path, f"{name}-{version}")
        return FAISS.load_local(
            f"{cls.name_version_path}", embedder, allow_dangerous_deserialization=True
        )

    @cached(cache)
    def load_gcs(cls, name: str, version: str) -> FAISS:
        blob = cls.bucket.blob(f"{name}/{version}/index.pkl")
        data_byte_array = BytesIO()
        blob.download_to_file(data_byte_array)
        docstore, index_to_docstore_id = pickle.loads(data_byte_array.getvalue())
        embedder = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
        )
        blob = cls.bucket.blob(f"{name}/{version}/index.faiss")
        blob.download_to_filename("/tmp/index.faiss")
        index = faiss.read_index("/tmp/index.faiss")
        store = FAISS(embedder, index, docstore, index_to_docstore_id)
        cls.base_path: str = os.path.join(basedir, "dist")
        cls.name_version_path: str = os.path.join(cls.base_path, f"{name}-{version}")
        store.save_local(cls.name_version_path)
        return store
