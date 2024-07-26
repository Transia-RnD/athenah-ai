#!/usr/bin/env python
# coding: utf-8

import os
from typing import List

from athenah_ai.utils.fs import write_file, read_file
from athenah_ai.client import AthenaClient, get_token_total
from athenah_ai.logger import logger


class AthenaLabeler(object):
    storage_type: str = "local"
    id: str = ""
    name: str = ""
    version: str = ""

    def __init__(cls, storage_type: str, id: str, name: str, version: str):
        cls.storage_type = storage_type
        cls.id = id
        cls.name = name
        cls.version = version
        cls.ai_client = AthenaClient(
            id=cls.id, model_group=f"bot-{cls.id}", model_name="gpt-3.5-turbo-16k"
        )

    @staticmethod
    def save_response(file_path: str, response: str):
        logger.info(f"SAVING .AI FILE: {file_path}")
        directory, file_name = os.path.split(file_path)
        base_name, extension = os.path.splitext(file_name)
        new_file_name = base_name + extension + ".ai"
        new_file_path = os.path.join(directory, new_file_name)
        write_file(new_file_path, response)

    def process_file(
        cls,
        ai_client: AthenaClient,
        file_path: str,
        prompt: str,
        skip_files: List[str],
    ):
        content: str = read_file(file_path)
        token_total = get_token_total(content)
        logger.info(f"FILE: {file_path} - TOKENS: {token_total}")
        if int(token_total) > 10000:
            skip_files.append(file_path)
            return

        try:
            response = ai_client.base_prompt(prompt, content)
            cls.save_response(file_path, response)
        except Exception as e:
            logger.info(f"File {file_path} failed: {e}")
            skip_files.append(file_path)

    def process_directory(
        cls, ai_client: AthenaClient, dir: str, prompt: str, skip_files: List[str]
    ):
        for root, dirs, files in os.walk(dir):
            for file in files:
                file_path = os.path.join(root, file)
                cls.process_file(ai_client, file_path, prompt, skip_files)

    def docstring_code(cls, filename: str, content: str):
        prompt = """
        You are a documentation ai bot. Your job is to accurately document code to
        docstring standards for the type of code provided.
        """
        try:
            response = cls.ai_client.base_prompt(prompt, content)
            return response
        except Exception as e:
            logger.info(f"File {filename} failed: {e}")

    def sumarize_code(cls, dir: str, folders: List[str] = []):
        skip_files: List[str] = []
        prompt = """
        Describe what each function in this code does. Be very specific.
        Every function must be documented.
        """

        if folders:
            for root_dir in folders:
                this_dir = os.path.join(dir, root_dir)
                cls.process_directory(cls.ai_client, this_dir, prompt, skip_files)
        else:
            cls.process_directory(cls.ai_client, dir, prompt, skip_files)

    def sumarize_text(cls, dir: str):
        skip_files: List[str] = []
        prompt = """
        Describe and summarize what this file says. Be very specific.
        Everything must be documented.
        """
        cls.process_directory(cls.ai_client, dir, prompt, skip_files)
