#!/usr/bin/env python
# coding: utf-8

import os

from basedir import basedir

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")


class VectorStore(object):
    storage_type: str = "local"  # local or gcs

    def __init__(cls, storage_type: str) -> None:
        cls.storage_type = storage_type
        pass

    def load(cls, name: str, dir: str = "dist", version: str = "v1") -> FAISS:
        if cls.storage_type == "local":
            return cls.load_local(
                dir,
                name,
                version,
            )

    def load_local(cls, dir: str, name: str, version: str) -> FAISS:
        embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        cls.base_path: str = os.path.join(basedir, dir)
        cls.name_path: str = os.path.join(cls.base_path, name)
        cls.name_version_path: str = os.path.join(cls.base_path, f"{name}-{version}")
        return FAISS.load_local(
            f"{cls.name_version_path}", embedder, allow_dangerous_deserialization=True
        )
