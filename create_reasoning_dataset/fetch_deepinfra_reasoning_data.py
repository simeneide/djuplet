#!/usr/bin/env python3
import os
import json
import argparse
from openai import OpenAI
from tqdm import tqdm

def process_file(input_file, template, output_file, api_key):
    """
    Processes a JSON-lines input file using the DeepSeek API and saves the output.
    For each input JSON, it sends a prompt created from the "text" field.
    The API response is expected to be a ChatCompletion object where:
      - response.choices[0].message.reasoning_content contains the reasoning.
    In the output JSON:
      - The original value from "text" is stored in "original_text".
      - It is also copied to "text_result".
      - The API's reasoning (from reasoning_content) is stored in "reasoning".
      - The original "text" field is then removed.
      
    If extraction fails, the error is logged and processing continues.
    """
    client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")

    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as out_f:
            processed_count = sum(1 for _ in out_f)

    with open(input_file, "r", encoding="utf-8") as in_f:
        total_lines = sum(1 for _ in in_f)

    try:
        with open(input_file, "r", encoding="utf-8") as in_f, \
             open(output_file, "a", encoding="utf-8") as out_f:

            # Skip already processed lines.
            for _ in range(processed_count):
                next(in_f)

            with tqdm(total=total_lines, initial=processed_count, desc="Processing") as pbar:
                for line in in_f:
                    data = json.loads(line.strip())
                    if "text" not in data:
                        # Log and skip records without 'text'
                        print("Skipping record: missing 'text' field.")
                        pbar.update(1)
                        continue

                    # Preserve the original text.
                    original_text = data["text"]
                    user_prompt = template.format(text=original_text)

                    try:
                        response = client.chat.completions.create(
                            model="deepseek-ai/DeepSeek-R1",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant"},
                                {"role": "user", "content": user_prompt},
                            ],
                            stream=True
                        )
                    except Exception as api_error:
                        print(f"API Error during call: {api_error}")
                        response = None

                    if response is not None:
                        try:
                            message = response.choices[0].message
                            api_reasoning = message.reasoning_content
                            if not api_reasoning:
                                print("reasoning_content not found in the message object:")
                                print(message)
                                api_reasoning = "ERROR: reasoning_content missing"
                            else:
                                api_reasoning = api_reasoning.strip()
                        except Exception as inner_err:
                            print(f"Error extracting reasoning: {inner_err}")
                            print("Full response object for inspection:")
                            print(response)
                            api_reasoning = "ERROR: Failed to extract reasoning"
                    else:
                        api_reasoning = "ERROR: Failed to get response from API"

                    # Add reasoning and original text fields.
                    data["reasoning"] = api_reasoning
                    data["text_result"] = original_text
                    data["original_text"] = original_text
                    del data["text"]

                    json.dump(data, out_f, ensure_ascii=False)
                    out_f.write("\n")
                    pbar.update(1)
    except Exception as e:
        print(f"Error: {e}")

def load_template(template_file):
    """Loads the template file."""
    with open(template_file, "r", encoding="utf-8") as file:
        return file.read()

def calculate_output_filename(input_file):
    """Generates the output filename based on the input file name."""
    base_name = os.path.splitext(input_file)[0]
    return f"{base_name}_processed.jsonl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON-lines with DeepSeek API.")
    parser.add_argument("--input_file", required=True, help="Input JSON-lines file.")
    parser.add_argument("--template_file", default="../templates/deepseek_template.txt", help="Template file.")
    args = parser.parse_args()

    api_key = os.getenv("DEEP_INFRA")
    if not api_key:
        raise EnvironmentError("DeepSeekApi environment variable not set.")

    template_content = load_template(args.template_file)
    output_file = calculate_output_filename(args.input_file)

    process_file(
        input_file=args.input_file,
        template=template_content,
        output_file=output_file,
        api_key=api_key
    )

