#!/usr/bin/env python
# coding: utf-8

import logging

# from typing import Dict, Any

from testing_config import BaseTestConfig

from athenah_ai.indexer import AthenaIndexer
from athenah_ai.client import AthenaClient


# Create Logger
logger = logging.getLogger("app")


class TestIndexer(BaseTestConfig):
    file_path: str = "./tests/fixtures/BuildInfo.cpp"
    prompt: str = "What does the isNewerVersion function return?"
    response: str = (
        "The `isNewerVersion` function returns `true` if the provided `version` is a valid Rippled version (as determined by the `isRippledVersion` function) and if it is greater than the encoded version obtained from `getEncodedVersion()`. If the `version` is not a valid Rippled version, it returns `false`."
    )

    def test_client(cls):
        indexer = AthenaIndexer("local", "id", "dist", "test_index_file", "v1")
        indexer.index_file(cls.file_path, "test_index_file")
        client = AthenaClient("id", "dist", "test_index_file", "v1")
        response = client.prompt(cls.prompt)
        cls.assertEqual(response, cls.response)
