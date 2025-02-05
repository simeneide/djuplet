#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
    )

def load_template(template_path: Path) -> str:
    try:
        with template_path.open("r", encoding="utf-8") as file:
            template = file.read()
            logging.debug(f"Loaded template from {template_path}")
            return template
    except Exception as e:
        logging.error(f"Error reading template file {template_path}: {e}")
        sys.exit(1)

def process_jsonl(input_path: Path, output_path: Path, prompt_template: str):
    try:
        with input_path.open("r", encoding="utf-8") as infile, \
             output_path.open("w", encoding="utf-8") as outfile:
            for line_number, line in enumerate(infile, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error at line {line_number}: {e}")
                    continue

                # Skip record if the reasoning field contains "ERROR"
                reasoning = record.get("reasoning", "")
                if "ERROR" in reasoning:
                    logging.debug(f"Skipping line {line_number} due to 'ERROR' in reasoning field.")
                    continue

                # Extract required fields
                corrupt = record.get("corrupt", "")
                original_text = record.get("original_text", "")

                # Build the prompt string
                prompt = f"{prompt_template}{corrupt}<think>{reasoning}</think>{original_text}"
                record["text"] = prompt

                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                logging.debug(f"Processed line {line_number}")
    except Exception as e:
        logging.error(f"Error processing jsonl file: {e}")
        sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process a JSONL file and add a 'text' key with the prompt to each record."
    )
    parser.add_argument(
        "--input_file", required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_file", required=True,
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--template_file", default="../templates/prompt_template.txt",
        help="Path to the prompt template file (default: ../templates/prompt_template.txt)."
    )
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_arguments()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    template_path = Path(args.template_file)

    if not input_path.exists():
        logging.error(f"Input file does not exist: {input_path}")
        sys.exit(1)

    prompt_template = load_template(template_path)
    process_jsonl(input_path, output_path, prompt_template)
    logging.info("Processing completed successfully.")

if __name__ == "__main__":
    main()

