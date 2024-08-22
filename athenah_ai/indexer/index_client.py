#!/usr/bin/env python
# coding: utf-8

import os
from typing import List, Union
import shutil
from shutil import ignore_patterns

from basedir import basedir

from langchain_community.vectorstores import FAISS

from athenah_ai.indexer.base_index_client import BaseIndexClient


class IndexClient(BaseIndexClient):
    storage_type: str = "local"  # local or gcs
    id: str = ""
    dir: str = ""
    name: str = ""
    version: str = ""

    def __init__(
        cls, storage_type: str, id: str, dir: str, name: str, version: str = "v1"
    ) -> None:
        cls.storage_type = storage_type
        cls.id = id
        cls.dir = dir
        cls.name = name
        cls.version = version
        cls.dist_path: str = os.path.join(basedir, "dist")
        cls.base_path: str = os.path.join(basedir, dir)
        cls.name_path: str = os.path.join(cls.base_path, cls.name)
        cls.name_version_path: str = os.path.join(
            cls.base_path, f"{cls.name}-{cls.version}"
        )
        os.makedirs(cls.base_path, exist_ok=True)
        os.makedirs(cls.name_path, exist_ok=True)
        super().__init__(cls.storage_type, cls.id, cls.dir, cls.name, cls.version)

    def remove(cls, dest: str, is_dir: bool = False):
        if is_dir:
            shutil.rmtree(dest, ignore_errors=True)
        else:
            os.remove(dest)

    def copy(cls, source: str, dest: str, is_dir: bool = False):
        if is_dir:
            shutil.copytree(
                source,
                dest,
                dirs_exist_ok=True,
                ignore=ignore_patterns(
                    "node_modules*",
                    "dist*",
                    "build*",
                    ".git*",
                    ".venv*",
                    ".vscode*",
                    "__pycache__*",
                    "poetry.lock",
                ),
            )
        else:
            os.makedirs(dest, exist_ok=True)
            file_name: str = source.split("/")[-1]
            shutil.copyfile(source, f"{dest}/{file_name}")

    def build(
        cls, name: str, folders: Union[List[str], str] = None, full: bool = False
    ):
        if type(folders) is list:
            build_paths: List[str] = [f"{cls.name_path}/{name}/{f}" for f in folders]
            store: FAISS = cls.build_batch(build_paths, full)
            cls.save(store)
            return store
        elif type(folders) is str or not folders:
            cls.clean(cls.name_path)
            cls.prepare(cls.name_path, full)
            store: FAISS = cls.build_one()
            cls.save(store)
            return store

        raise ValueError(f"unimplemented: {len(folders)}")
