#!/usr/bin/env python
# coding: utf-8

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language


def code_splitter(
    language: Language = Language.CPP, chunk_size: int = 1000, chunk_overlap: int = 20
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_language(
        language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )


def text_splitter(
    chunk_size: int = 1000, chunk_overlap: int = 20
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
