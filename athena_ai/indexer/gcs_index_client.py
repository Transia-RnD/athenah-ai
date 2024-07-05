#!/usr/bin/env python
# coding: utf-8

import os
import json
from typing import Dict, Any, List, Union
import requests
from shutil import ignore_patterns

from langchain_community.vectorstores import FAISS

from basedir import basedir


class GCSIndexClient(object):
    id: str = ""
    dir: str = ""
    name: str = ""
    version: str = ""
    gcs_name: str = ""

    def __init__(cls, id: str, dir: str, name: str, version: str = "v1") -> None:
        cls.id = id
        cls.dir = dir
        cls.name = name
        cls.version = version

        # OLD CODE
        cls.base_path: str = os.path.join(basedir, dir)
        cls.name_path: str = os.path.join(cls.base_path, cls.name)
        cls.name_version_path: str = os.path.join(
            cls.base_path, f"{cls.name}-{cls.version}"
        )
        os.makedirs(cls.base_path, exist_ok=True)
        os.makedirs(cls.name_path, exist_ok=True)
        os.makedirs(cls.name_version_path, exist_ok=True)

        # NEW CODE
        cls.gcs_name: str = f"{cls.name}/{cls.version}/index"
        super().__init__()

    def copy(cls, source: str, dest: str, is_dir: bool = False):
        import os, shutil
        from shutil import ignore_patterns

        # Check if the destination directory has files
        if os.path.exists(dest) and os.path.isdir(dest):
            if len(os.listdir(dest)) > 0:
                # Remove all files in the destination directory
                for file_name in os.listdir(dest):
                    file_path = os.path.join(dest, file_name)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print("Failed to delete %s. Reason: %s" % (file_path, e))

        if is_dir:
            shutil.copytree(
                source,
                dest,
                dirs_exist_ok=True,
                ignore=ignore_patterns("node_modules*", "dist*", "build*"),
            )
        else:
            file_name: str = source.split("/")[-1]
            shutil.copyfile(source, f"{dest}/{file_name}")

    def build(cls, name: str = None, include: List[str] = None) -> FAISS:
        repo_client = AIRepoClient(
            id=cls.id,
            dir=cls.dir,
            name=cls.name,
            version=cls.version,
        )

        if type(include) == list and len(include) > 0:
            _path: str = os.path.join(cls.name_path, name)
            print("HAS INCLUDE")
            build_paths: List[str] = [f"{_path}/{f}" for f in include]
            print(f"BUILD PATHS: {build_paths}")
            store: FAISS = repo_client.build_batch(build_paths)
            print(f"SAVE PATHS: {cls.gcs_name}")
            cls.save(cls.gcs_name, store)
            return store

        else:
            print("NO FOLDERS")
            print(f"BUILDING: {cls.name_path}")
            repo_client.clean(cls.name_path)
            repo_client.prepare(cls.name_path)
            store: FAISS = repo_client.build()
            cls.save(cls.gcs_name, store)
            return store
