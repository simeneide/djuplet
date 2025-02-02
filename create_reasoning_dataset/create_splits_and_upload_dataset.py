#!/usr/bin/env python3
"""
Split a large JSONL dataset into multiple splits and upload them to the Hugging Face Hub.

This script performs the following steps:
  1. Reads an input JSONL file containing dataset samples.
  2. Randomly shuffles the lines.
  3. Splits the dataset into predefined splits:
       - train.jsonl: 1,000,000 samples
       - validation.jsonl: 10,000 samples
       - test.jsonl: 10,000 samples
       - validation1000.jsonl: 1,000 samples
       - test1000.jsonl: 1,000 samples
       - validation100.jsonl: 100 samples
       - test100.jsonl: 100 samples
       - pretrain.jsonl: 10,000 samples
       - reserve.jsonl: 100,000 samples
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

import json
import random
import argparse
import os
import tempfile
from tqdm import tqdm
from huggingface_hub import HfApi

def split_and_save(input_file: str, output_dir: str):
    """
    Reads the input JSONL file, shuffles its lines, splits them into predefined splits,
    and saves each split in the output directory along with a dataset_info.json file.

    Parameters:
        input_file (str): Path to the input JSONL file.
        output_dir (str): Directory where the split files and metadata will be saved.
    """
    # Read all lines from the input file.
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Randomly shuffle the dataset lines.
    random.shuffle(lines)

    # Predefined splits and their desired sample counts.
    splits = {
        "train.jsonl": 1_000_000,
        "validation.jsonl": 10_000,
        "test.jsonl": 10_000,
        "validation1000.jsonl": 1_000,
        "test1000.jsonl": 1_000,
        "validation100.jsonl": 100,
        "test100.jsonl": 100,
        "pretrain.jsonl": 10_000,
        "reserve.jsonl": 100_000,
    }

    index = 0
    os.makedirs(output_dir, exist_ok=True)
    dataset_info = {"splits": [], "total_samples": 0}

    # For each split, write the specified number of lines to a file.
    for filename, count in splits.items():
        split_name = filename.replace(".jsonl", "")
        actual_count = min(count, len(lines) - index)

        # Record split metadata in Hugging Face's expected format.
        dataset_info["splits"].append({
            "name": split_name,
            "num_examples": actual_count
        })
        dataset_info["total_samples"] += actual_count

        split_file_path = os.path.join(output_dir, filename)
        with open(split_file_path, 'w', encoding='utf-8') as outfile:
            for _ in range(actual_count):
                outfile.write(lines[index])
                index += 1

    # Additional dataset metadata.
    dataset_info.update({
        "format": "jsonl",
        "description": "Dataset split information for Hugging Face repository",
        "citation": "",  # Add citation if needed.
        "license": ""    # Add license information if needed.
    })

    # Save metadata to dataset_info.json.
    info_file_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_file_path, 'w', encoding='utf-8') as info_file:
        json.dump(dataset_info, info_file, indent=4)

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
        api.repo_info(repo_id)
    except Exception:
        api.create_repo(repo_id, repo_type="dataset")

    # Upload each file in the output directory to the repository.
    for file_name in tqdm(os.listdir(output_dir), desc="Uploading files"):
        file_path = os.path.join(output_dir, file_name)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_name,
            repo_id=repo_id,
            repo_type="dataset"
        )

    print("\nAll splits and metadata pushed to Hugging Face Hub.")

def main():
    """
    Parses command-line arguments and executes the splitting and uploading process.
    
    Required arguments:
      --input_file: Path to the input JSONL file.
      --repo_id: Hugging Face repository ID (e.g., "username/dataset-name").
    """
    parser = argparse.ArgumentParser(description="Split a dataset and push it to the Hugging Face Hub")
    parser.add_argument("--input_file", required=True, help="Input JSONL file path")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repo ID (e.g., 'username/dataset-name')")
    args = parser.parse_args()

    # Create a temporary directory to store the split files.
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        split_and_save(args.input_file, temp_dir)
        push_to_huggingface(temp_dir, args.repo_id)
        print("Temporary directory contents have been pushed. The temporary directory will now be deleted.")

if __name__ == "__main__":
    main()

