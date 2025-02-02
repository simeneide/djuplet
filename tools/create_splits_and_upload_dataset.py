import json
import random
import argparse
import os
from tqdm import tqdm
from huggingface_hub import HfApi

def split_and_save(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    random.shuffle(lines)
    
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
    dataset_info = {"splits": [], "total_samples": 0}  # Corrected structure
    
    for filename, count in splits.items():
        split_name = filename.replace(".jsonl", "")
        actual_count = min(count, len(lines) - index)
        
        # Add split information in Hugging Face's expected format
        dataset_info["splits"].append({
            "name": split_name,
            "num_examples": actual_count
        })
        dataset_info["total_samples"] += actual_count
        
        with open(f"{output_dir}/{filename}", 'w', encoding='utf-8') as outfile:
            for _ in range(actual_count):
                outfile.write(lines[index])
                index += 1
    
    # Add additional metadata
    dataset_info.update({
        "format": "jsonl",
        "description": "Dataset split information for Hugging Face repository",
        "citation": "",  # Add your citation here if needed
        "license": ""    # Add license information if needed
    })
    
    # Save dataset_info.json
    with open(f"{output_dir}/dataset_info.json", 'w', encoding='utf-8') as info_file:
        json.dump(dataset_info, info_file, indent=4)
    
    print("Dataset split completed and dataset_info.json created.")

def push_to_huggingface(output_dir, repo_id):
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        api.repo_info(repo_id)
    except:
        api.create_repo(repo_id, repo_type="dataset")
    
    # Upload all files with progress bar
    for file_name in tqdm(os.listdir(output_dir), desc="Uploading files"):
        file_path = os.path.join(output_dir, file_name)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_name,
            repo_id=repo_id,
            repo_type="dataset"
        )
    
    print("\nAll splits and metadata pushed to Hugging Face Hub.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and push dataset to Hugging Face Hub")
    parser.add_argument("--input_file", required=True, help="Input JSONL file path")
    parser.add_argument("--output_dir", required=True, help="Output directory for splits")
    parser.add_argument("--repo_id", required=True, help="HF repo ID (e.g., 'username/dataset-name')")
    
    args = parser.parse_args()
    
    split_and_save(args.input_file, args.output_dir)
    push_to_huggingface(args.output_dir, args.repo_id)
