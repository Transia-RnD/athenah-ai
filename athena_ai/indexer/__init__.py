#!/usr/bin/env python
# coding: utf-8

import logging
import os
from typing import List, Union

from basedir import basedir
from athena_ai.indexer.local_index_client import LocalIndexClient
from athena_ai.indexer.gcs_index_client import GCSIndexClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("app")


class AthenaIndexer(object):
    storage_type: str = "local"  # local or gcs
    id: str = ""
    name: str = ""
    version: str = ""
    index_client: Union[LocalIndexClient, GCSIndexClient] = None

    def __init__(
        cls,
        storage_type: str,
        id: str,
        name: str,
        version: str,
    ):
        cls.storage_type = storage_type
        cls.id = id
        cls.name = name
        cls.version = version
        pass

    def index_dir(cls, source: str, files: List[str], name: str):
        source_name: str = f"{name}-source"
        dest_filepath: str = os.path.join(basedir, f"dist/{name}/{source_name}")
        print(f"STORAGE: {cls.storage_type}")
        print(f"NAME: {name}")
        print(f"SOURCE: {source}")
        print(f"FILES: {files}")
        print(f"DEST PATH: {dest_filepath}")
        if cls.storage_type == "local":
            index_client = LocalIndexClient("id", "dist", f"{name}", "v1")
            index_client.copy(source, dest_filepath, True)
            index_client.build(source_name, files)

        if cls.storage_type == "gcs":
            index_client = GCSIndexClient("id", "dist", f"{name}", "v1")
            index_client.copy(source, dest_filepath, True)
            index_client.build(source_name, files)

    def index_file(cls, file_path: str, name: str):
        source_name: str = f"{name}-source"
        dest_filepath: str = os.path.join(basedir, f"dist/{name}/{source_name}")
        print(f"STORAGE: {cls.storage_type}")
        print(f"NAME: {name}")
        print(f"FILE PATH: {file_path}")
        print(f"DEST PATH: {dest_filepath}")
        if cls.storage_type == "local":
            index_client = LocalIndexClient("id", "dist", f"{name}", "v1")
            index_client.copy(file_path, dest_filepath, False)
            index_client.build(source_name, dest_filepath)

        if cls.storage_type == "gcs":
            index_client = GCSIndexClient("id", "dist", f"{name}", "v1")
            index_client.copy(file_path, dest_filepath, False)
            index_client.build(source_name, dest_filepath)
