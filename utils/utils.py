import json
import os
import logging
from typing import List

def read_json_file(file_path: str) -> list:
    """
    Reads a JSON file and returns its content.

    Args:
        file_path (str): The path of the JSON file to read.

    Returns:
        list: The content of the JSON file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def write_json_file(data: List[dict], output_file_path: str):
    """
    Writes the given data to a JSON file.

    Args:
        data (List[dict]): The data to write.
        output_file_path (str): The file path to write the data to.
    """
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
        logging.info(f"Data written to {output_file_path}")
