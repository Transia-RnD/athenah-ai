#!/usr/bin/env python
# coding: utf-8

import os
from typing import List

from basedir import basedir
from dotenv import load_dotenv

from athenah_ai.indexer.index_client import IndexClient
from athenah_ai.logger import logger

load_dotenv()


class AthenahIndexer(IndexClient):
    storage_type: str = "local"  # local or gcs
    id: str = ""
    dir: str = ""
    name: str = ""
    version: str = ""

    def __init__(
        cls,
        storage_type: str,
        id: str,
        dir: str,
        name: str,
        version: str,
    ):
        cls.storage_type = storage_type
        cls.id = id
        cls.dir = dir
        cls.name = name
        cls.version = version
        super().__init__(cls.storage_type, cls.id, cls.dir, cls.name, cls.version)
        pass

    def index_dir(cls, source: str, files: List[str], name: str, full: bool = False):
        source_name: str = f"{name}-source"
        dest_filepath: str = os.path.join(basedir, f"dist/{name}/{source_name}")
        logger.info(f"STORAGE: {cls.storage_type}")
        logger.info(f"NAME: {name}")
        logger.info(f"SOURCE: {source}")
        logger.info(f"FILES: {files}")
        logger.info(f"DEST PATH: {dest_filepath}")
        cls.copy(source, dest_filepath, True)
        cls.build(source_name, files, full)

    def index_file(cls, file_path: str, name: str, full: bool = False):
        source_name: str = f"{name}-source"
        dest_filepath: str = os.path.join(basedir, f"dist/{name}/{source_name}")
        logger.info(f"STORAGE: {cls.storage_type}")
        logger.info(f"NAME: {name}")
        logger.info(f"FILE PATH: {file_path}")
        logger.info(f"DEST PATH: {dest_filepath}")
        cls.copy(file_path, dest_filepath, False)
        cls.build(source_name, dest_filepath, full)
