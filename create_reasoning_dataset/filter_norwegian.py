import json
import argparse
import fasttext
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Download and load the GlotLID model
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None)
model = fasttext.load_model(model_path)

def detect_language(text):
    """Detect language using GlotLID."""
    cleaned_text = text.replace("\n", " ")  # Remove newlines to avoid fasttext error
    prediction = model.predict(cleaned_text.strip())[0][0]
    return prediction.replace("__label__", "")  # Remove FastText label prefix

def filter_norwegian(input_file, output_file):
    # Count total lines for progress bar
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         tqdm(total=total_lines, desc="Processing", unit=" lines") as pbar:
        
        for line in infile:
            try:
                data = json.loads(line)
                if 'reasoning' in data:
                    detected_lang = detect_language(data['reasoning'])
                    #if detected_lang in {'nob_Latn', 'nno_Latn'}:  # Norwegian Bokmål or Nynorsk
                    if detected_lang in {'nob_Latn'}:  # Norwegian Bokmål
                        json.dump(data, outfile, ensure_ascii=False)
                        outfile.write('\n')
            except Exception as e:
                print(f"Skipping line due to error: {e}")

            pbar.update(1)  # Update progress bar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSONLines based on Norwegian language detection in the reasoning field using GlotLID.")
    parser.add_argument("--input_file", required=True, help="Path to input JSONLines file.")
    parser.add_argument("--output_file", required=True, help="Path to output JSONLines file.")
    args = parser.parse_args()

    filter_norwegian(args.input_file, args.output_file)
