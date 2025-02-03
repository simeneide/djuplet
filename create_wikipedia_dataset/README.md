# Norwegian Wikipedia Data Processing Pipeline

This repository provides a complete end-to-end pipeline for processing Norwegian Wikipedia data. The pipeline consists of several sequential scripts that perform the following tasks:

1. **Download and Extract Wikipedia Paragraphs**  
   `download_wiki_paragraphs.py` downloads the latest Wikipedia pages-articles dump for the specified language and extracts valid paragraphs into a JSONL file.

2. **Corrupt Paragraphs (Data Augmentation)**  
   `corrupt_paragraphs.py` reads the JSONL file of paragraphs and applies randomized corruption transformations to each paragraph. This step adds additional fields with corrupted text to each JSON record.

3. **Create Dataset Splits and Upload to Hugging Face Hub**  
   `create_splits_and_upload_dataset.py` takes the final JSONL file, shuffles and splits it into several predefined splits (train, validation, test, etc.), and uploads all the split files (along with metadata) to a Hugging Face Hub repository.

## Repository Structure

- **download_wiki_paragraphs.py**: Downloads and extracts valid paragraphs from a Wikipedia dump.
- **validate_and_orrupt_paragraphs.py**: Validates every line, clean up some oddities and spplies various corruption transformations to the extracted paragraphs.
- **create_splits_and_upload_dataset.py**: Splits the processed data into several dataset splits and uploads them to the Hugging Face Hub.
- **README.md**: This file.

# Procedure

## 1. Download and Extract Wikipedia Paragraphs
Run the download_wiki_paragraphs.py script to download the Norwegian Wikipedia dump and extract valid paragraphs:

```bash
python download_wiki_paragraphs.py --language no --output_file ../data/norwegian.jsonl --temp_dump_file ../data/norwegian.xml.bz2
 
```

Parameters:
```bash
--language: Wikipedia language code (e.g., no for Norwegian).
--output_file: Destination JSONL file to store extracted paragraphs.
[--max_paragraphs: Maximum number of paragraphs to extract. For testing.]
[--minimum_words_paragraph: Minimum number of words for a paragraph to be considered valid.]
```

## 2. Apply Corruption Transformations
Run the validate_and_corrupt_paragraphs.py script to add corrupted versions of the paragraphs:

```bash
python validate_paragraphs.py --input_file norwegian.jsonl --output_file norwegian_corrupt.jsonl
```

This script reads each JSON record from norwegian.jsonl, randomly selects a corruption level (0–9), and adds two new fields:
corrupt: The corrupted paragraph text.
corrupt_level: The integer level of corruption applied.

Parameters:
```bash
--input_file: The processed JSONL file (e.g., output from the Wikipedia Download  step).
--output_file: Destination JSONL file to store jsonlines with corrupted  paragraphs.
```


## 3. Create Dataset Splits and Upload to Hugging Face Hub
Finally, split the processed dataset into multiple splits and upload them:

```bash
python create_splits_and_upload_dataset.py --input_file norwegian_corrupt.jsonl --repo_id your_username/your_dataset_name
```

Parameters:
```bash
--input_file: The processed JSONL file (e.g., output from the Corruption step).
--repo_id: Your Hugging Face repository identifier (e.g., username/dataset-name).
```



