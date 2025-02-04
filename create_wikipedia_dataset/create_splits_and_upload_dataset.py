#!/usr/bin/env python3
import json
import random
import argparse
import os
import tempfile
from tqdm import tqdm
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

def split_and_save(input_file: str, output_dir: str):
    """
    Reads the input JSONL file, shuffles its lines, splits them into predefined splits,
    and saves each split in the output directory along with a dataset_info.json file.
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    random.shuffle(lines)

    splits = {
        "train.jsonl":         1_000_000,
        "validation.jsonl":    10_000,
        "test.jsonl":          10_000,
        "validation1000.jsonl":1_000,
        "test1000.jsonl":      1_000,
        "validation100.jsonl": 100,
        "test100.jsonl":       100,
        "pretrain.jsonl":      10_000,
        "reserve.jsonl":       100_000,
    }

    os.makedirs(output_dir, exist_ok=True)
    dataset_info = {"splits": [], "total_samples": 0}

    index = 0
    for filename, count in splits.items():
        split_name = filename.replace(".jsonl", "")
        actual_count = min(count, len(lines) - index)

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

    dataset_info.update({
        "format": "jsonl",
        "description": "Dataset split information for Hugging Face repository",
        "citation": "",
        "license": ""
    })

    info_file_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_file_path, 'w', encoding='utf-8') as info_file:
        json.dump(dataset_info, info_file, indent=4)

    print("Dataset split completed and dataset_info.json created.")

def push_to_huggingface(output_dir: str, repo_id: str):
    """
    Uploads all files in the specified output directory to a Hugging Face Hub repository.
    If the repo does not exist, create it. If it already exists, push anyway.
    This version does not rely on HfHubHTTPError, just requests' HTTPError and its status code.
    """
    api = HfApi()

    # Check if the repo exists. If 404 => create. If 409 => push anyway. Otherwise re-raise the error.
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"Repository '{repo_id}' already exists. Will push anyway.")
    except HTTPError as e:
        if e.response is not None:
            status_code = e.response.status_code
            # 404: Repository not found => create it
            if status_code == 404:
                print(f"Repository '{repo_id}' does not exist. Creating it now...")
                try:
                    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
                except HTTPError as create_err:
                    if create_err.response is not None and create_err.response.status_code == 409:
                        # 409 conflict => it might have been created in parallel, push anyway
                        print(f"Repo '{repo_id}' was just created, or there's a naming conflict. Will push anyway.")
                    else:
                        raise create_err
            else:
                raise e
        else:
            raise e

    # Now push the files (whether newly created or already existing)
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
    parser = argparse.ArgumentParser(description="Split a dataset and push it to the Hugging Face Hub")
    parser.add_argument("--input_file", required=True, help="Input JSONL file path")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repo ID (e.g., 'username/dataset-name')")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        split_and_save(args.input_file, temp_dir)
        push_to_huggingface(temp_dir, args.repo_id)
        print("Temporary directory contents have been pushed. The temporary directory will now be deleted.")

if __name__ == "__main__":
    main()

