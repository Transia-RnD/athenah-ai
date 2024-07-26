#!/usr/bin/env python
# coding: utf-8

import logging

# from typing import Dict, Any

from testing_config import BaseTestConfig

from athenah_ai.indexer import AthenahIndexer
from athenah_ai.client.vector_store import VectorStore


# Create Logger
logger = logging.getLogger("app")


class TestIndexer(BaseTestConfig):
    file_path: str = "./tests/fixtures/BuildInfo.cpp"
    dir_path: str = "./tests/fixtures"

    def test_index_file(cls):
        indexer = AthenahIndexer("local", "id", "dist", "test_index_file", "v1")
        indexer.index_file(cls.file_path, "test_index_file")
        vector_store = VectorStore("local")
        db = vector_store.load("test_index_file", "dist", "v1")
        num_indexs = len(db.index_to_docstore_id)
        cls.assertEqual(num_indexs, 4)

    def test_index_dir(cls):
        indexer = AthenahIndexer("local", "id", "dist", "test_index_dir", "v1")
        indexer.index_dir(cls.dir_path, ["."], "test_index_dir")
        vector_store = VectorStore("local")
        db = vector_store.load("test_index_dir", "dist", "v1")
        num_indexs = len(db.index_to_docstore_id)
        cls.assertEqual(num_indexs, 4)
