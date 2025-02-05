#!/usr/bin/env python3
"""
Split a JSONL dataset into train, test, and validation splits and upload them to the Hugging Face Hub.

This script performs the following steps:
  1. Reads an input JSONL file containing dataset samples.
  2. Randomly shuffles the samples.
  3. Splits the dataset into fixed splits:
       - test.jsonl: 1 sample
       - validation.jsonl: 1 sample
       - train.jsonl: all remaining samples
  4. Saves each split as a JSONL file in a temporary output directory.
  5. Generates a "dataset_info.json" file containing metadata and split information.
  6. Uploads all the files in the output directory to a specified Hugging Face Hub repository.

Usage:
  python create_splits_and_upload_dataset.py --input_file input.jsonl --repo_id username/dataset-name

Dependencies:
  - huggingface_hub (for interacting with Hugging Face Hub)
    pip install huggingface_hub
  - tqdm (for progress bars)
    pip install tqdm
"""

import argparse
import json
import random
import os
import sys
import tempfile
import logging
from tqdm import tqdm
from huggingface_hub import HfApi

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
    )

def split_and_save(input_file: str, output_dir: str):
    """
    Reads the input JSONL file, shuffles its lines, splits them into train, test, and validation splits,
    and saves each split in the output directory along with a dataset_info.json file.

    Fixed splits:
       - test.jsonl: 250 sample
       - validation.jsonl: 250 sample
       - train.jsonl: remaining samples

    Parameters:
        input_file (str): Path to the input JSONL file.
        output_dir (str): Directory where the split files and metadata will be saved.
    """
    # Read all lines from the input file.
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except Exception as e:
        logging.error(f"Error reading input file {input_file}: {e}")
        sys.exit(1)

    total_samples = len(lines)
    if total_samples < 2:
        logging.error("Input file must contain at least 2 samples to create test and validation splits.")
        sys.exit(1)

    random.shuffle(lines)
    logging.debug(f"Shuffled {total_samples} samples.")

    # Define fixed split counts.
    test_count = 250
    validation_count = 250
    train_count = total_samples - (test_count + validation_count)
    logging.debug(f"Splitting dataset: train={train_count}, validation={validation_count}, test={test_count}")

    # Prepare splits.
    split_files = {
        "test.jsonl": lines[:test_count],
        "validation.jsonl": lines[test_count:test_count + validation_count],
        "train.jsonl": lines[test_count + validation_count:]
    }

    os.makedirs(output_dir, exist_ok=True)

    # Write each split to its respective file.
    for filename, split_lines in split_files.items():
        file_path = os.path.join(output_dir, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as outfile:
                for line in split_lines:
                    outfile.write(line)
            logging.debug(f"Wrote {len(split_lines)} samples to {filename}")
        except Exception as e:
            logging.error(f"Error writing file {file_path}: {e}")
            sys.exit(1)

    # Create dataset metadata.
    dataset_info = {
        "splits": [
            {"name": "train", "num_examples": len(split_files["train.jsonl"])},
            {"name": "validation", "num_examples": len(split_files["validation.jsonl"])},
            {"name": "test", "num_examples": len(split_files["test.jsonl"])}
        ],
        "total_samples": total_samples,
        "format": "jsonl",
        "description": "Dataset split information for Hugging Face repository",
        "citation": "",
        "license": ""
    }

    info_file_path = os.path.join(output_dir, "dataset_info.json")
    try:
        with open(info_file_path, 'w', encoding='utf-8') as info_file:
            json.dump(dataset_info, info_file, indent=4)
        logging.debug("Created dataset_info.json with metadata.")
    except Exception as e:
        logging.error(f"Error writing dataset_info.json: {e}")
        sys.exit(1)

    print("Dataset split completed and dataset_info.json created.")

def push_to_huggingface(output_dir: str, repo_id: str):
    """
    Uploads all files in the specified output directory to a Hugging Face Hub repository.

    Parameters:
        output_dir (str): Directory containing the split files and metadata.
        repo_id (str): The Hugging Face repository ID (e.g., "username/dataset-name").
    """
    api = HfApi()

    # Create the repository if it doesn't exist.
    try:
        api.repo_info(repo_id, repo_type="dataset")
    except Exception:
        api.create_repo(repo_id, repo_type="dataset")
        logging.debug(f"Created repository {repo_id} on Hugging Face Hub.")

    # Upload each file in the output directory.
    for file_name in tqdm(os.listdir(output_dir), desc="Uploading files"):
        file_path = os.path.join(output_dir, file_name)
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="dataset"
            )
            logging.debug(f"Uploaded {file_name} to repository {repo_id}.")
        except Exception as e:
            logging.error(f"Error uploading {file_name}: {e}")
            sys.exit(1)

    print("\nAll splits and metadata pushed to Hugging Face Hub.")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Split a JSONL dataset into fixed train, test, and validation splits and push to the Hugging Face Hub"
    )
    parser.add_argument("--input_file", required=True, help="Path to the input JSONL file")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repository ID (e.g., 'username/dataset-name')")
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_arguments()

    # Create a temporary directory to store the split files.
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        split_and_save(args.input_file, temp_dir)
        push_to_huggingface(temp_dir, args.repo_id)
        print("Temporary directory contents have been pushed. The temporary directory will now be deleted.")

if __name__ == "__main__":
    main()

