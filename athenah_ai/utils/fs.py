#!/usr/bin/env python
# coding: utf-8

from typing import Dict, Any  # noqa: F401
import json


def read_json(path: str) -> Dict[str, object]:
    """Read Json

     # noqa: E501

    :param path: Path to json
    :type path: str

    :rtype: Dict[str, object]
    """
    with open(path) as json_file:
        return json.load(json_file)


def write_json(data: Dict[str, object], path: str):
    """Write Json

     # noqa: E501

    :param path: Path to json
    :type path: str

    :rtype: None
    """
    with open(path, "w") as json_file:
        json.dump(data, json_file)


def write_file(path: str, data: Any) -> str:
    """Write File

     # noqa: E501

    :param path: Path to file
    :type path: str

    :rtype: str
    """
    with open(path, "w") as f:
        return f.write(data)


def read_file(path: str) -> str:
    """Read File

     # noqa: E501

    :param path: Path to file
    :type path: str

    :rtype: str
    """
    with open(path, "r") as f:
        return f.read()
