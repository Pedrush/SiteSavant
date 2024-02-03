import json
import os
import logging
import yaml
import h5py
from typing import List, Dict, Any

def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a JSON file and returns its content.

    Args:
        file_path (str): The path of the JSON file to read.

    Returns:
        List[Dict[str, Any]]: The content of the JSON file.

    Raises:
        IOError: If there is an error reading the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except IOError as e:
        raise IOError(f"Error reading JSON file: {e}")

def write_json_file(data: List[dict], output_file_path: str, timestamp: str = None):
    """
    Writes the given data to a JSON file. Optionally appends a timestamp to the filename.

    Args:
        data (List[dict]): The data to write.
        output_file_path (str): The file path to write the data to.
        timestamp (str, optional): Timestamp string to append to the filename.
    """
    if timestamp:
        file_name, file_extension = os.path.splitext(output_file_path)
        output_file_path = f"{file_name}_{timestamp}{file_extension}"

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
        logging.info(f"Data written to {output_file_path}")

def read_yaml_file(file_path):
    """
    Reads a YAML file and returns its content.

    Args:
        file_path (str): The path of the YAML file to read.

    Returns:
        dict or None: The content of the YAML file if successfully read and parsed, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"YAML file not found: {file_path}")
        return None
    except yaml.YAMLError as exc:
        logging.error(f"Error parsing YAML file: {file_path} - {exc}")
        return None
    
def load_embeddings(file_path: str) -> Dict[str, Any]:
    """
    Reads an HDF5 file and returns a dictionary of embeddings.

    Args:
        file_path (str): The path of the HDF5 file to read.

    Returns:
        Dict[str, Any]: A dictionary of embeddings keyed by embedding_id.

    Raises:
        IOError: If there is an error reading the HDF5 file.
    """
    embeddings = {}
    try:
        with h5py.File(file_path, 'r') as file:
            for key in file.keys():
                embeddings[key] = file[key][:]
        return embeddings
    except IOError as e:
        raise IOError(f"Error reading HDF5 file: {e}")

def join_data(records: List[Dict[str, Any]], embeddings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Joins records with their corresponding embeddings based on the embedding_id.

    Args:
        records (List[Dict[str, Any]]): A list of records from the JSON file.
        embeddings (Dict[str, Any]): A dictionary of embeddings from the HDF5 file.

    Returns:
        List[Dict[str, Any]]: A list of records with embeddings added to them.
    """
    for record in records:
        embedding_id = record.get('embedding_id')
        record['embedding'] = embeddings.get(embedding_id, None)  # None if embedding_id is not found
    return records

def validate_embeddings(embeddings_data: List[Dict[str, Any]]) -> None:
    """
    Validates the embeddings in the data of records.
    Ensures that all embeddings have the same dimensions and are of the correct type (float).

    Args:
        embeddings_data (List[Dict[str, Any]]): A list of dictionaries
        containing embeddings and the relevant metadata.

    Raises:
        ValueError: If the data is empty, embeddings are not floats, or have inconsistent dimensions.

    """
    if not embeddings_data:
        raise ValueError("No data to validate.")

    embedding_length = len(embeddings_data[0]['embedding'])
    for record in embeddings_data:
        if not all(isinstance(x, float) for x in record['embedding']):
            raise ValueError("Embeddings must be floats.")
        if len(record['embedding']) != embedding_length:
            raise ValueError("Inconsistent embedding dimensions found.")
        
def save_embeddings_and_metadata(
        data: List[Dict[str, Any]],
        data_dir: str,
        metadata_file_name: str = 'processed_metadata',
        embeddings_file_name: str = 'processed_embeddings_values',
        timestamp: str = None
    ) -> None:
    """
    Saves data into separate JSON and HDF5 files in the specified directory. 
    Optionally appends a timestamp to the filenames.

    Args:
        data (List[Dict[str, Any]]): The data to be saved.
        data_dir (str): The directory where the data will be saved.
        metadata_file_name (str): The desired file name of the metadata.
        embeddings_file_name (str): The desired file name of the embeddings.
        timestamp (str, optional): Timestamp string to append to the filenames.

    Raises:
        ValueError: If data is empty or improperly formatted.
        IOError: If there's an error in file operations.
    """
    if not data:
        raise ValueError("No data provided for saving.")

    if timestamp:
        metadata_file_name += f"_{timestamp}"
        embeddings_file_name += f"_{timestamp}"

    json_file_path = os.path.join(data_dir, f"{metadata_file_name}.json")
    hdf5_file_path = os.path.join(data_dir, f"{embeddings_file_name}.h5")

    try:
        with h5py.File(hdf5_file_path, 'w') as hdf5_file, open(json_file_path, 'w') as json_file:
            json_data = []
            for i, record in enumerate(data):
                if 'embedding' not in record:
                    raise ValueError(f"Missing 'embedding' key in record {i}")

                dataset_name = f"embedding_{i}"
                json_record = {key: record[key] for key in record if key != 'embedding'}
                json_record['embedding_id'] = dataset_name
                json_data.append(json_record)
                hdf5_file.create_dataset(dataset_name, data=record['embedding'])

            json.dump(json_data, json_file, indent=4)

        logging.info(f"Metadata and embeddings' values saved to:\n{json_file_path}\n{hdf5_file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving data: {e}")
        raise IOError("Failed to save data.") from e
    
def read_markdown_file(file_path: str) -> str:
    """
    Reads a Markdown file and returns its contents.

    Args:
        file_path (str): The path to the Markdown file.

    Returns:
        str: The contents of the Markdown file if successful, or an error message.

    Raises:
        FileNotFoundError: If the Markdown file does not exist.
        Exception: For other issues that may occur while reading the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"
