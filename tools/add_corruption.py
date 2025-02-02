import json
import random
import argparse
from corrupter import corrupt_paragraph
from tqdm import tqdm

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(lines, desc="Processing", unit="line", total=len(lines)):
            data = json.loads(line)
            if 'text' in data:
                level = random.randint(0, 9)
                data['corrupt'] = corrupt_paragraph(data['text'], level)
                data['corrupt_level'] = level
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply corrupt_paragraph to a JSONL file.")
    parser.add_argument("--input_file", required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_file", required=True, help="Path to the output JSONL file")
    args = parser.parse_args()
    
    process_jsonl(args.input_file, args.output_file)

