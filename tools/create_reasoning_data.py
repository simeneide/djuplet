import os
import json
from openai import OpenAI
from tqdm import tqdm
import argparse

def process_file(input_file, template, output_file, api_key):
    """
    Processes a JSON-lines input file using the DeepSeek API and saves the output.
    """
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as output:
            processed_count = sum(1 for _ in output)

    with open(input_file, "r", encoding="utf-8") as input_f:
        total_lines = sum(1 for _ in input_f)

    try:
        with open(input_file, "r", encoding="utf-8") as input_f, \
             open(output_file, "a", encoding="utf-8") as output_f:

            for _ in range(processed_count):
                next(input_f)

            with tqdm(total=total_lines, initial=processed_count, desc="Processing") as pbar:
                for line in input_f:
                    data = json.loads(line.strip())

                    if "text" not in data:
                        raise ValueError("Missing 'text' field.")

                    user_prompt = template.format(text=data["text"])

                    try:
                        response = client.chat.completions.create(
                            model="deepseek-reasoner",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant"},
                                {"role": "user", "content": user_prompt},
                            ],
                            stream=False
                        )
                        reasoning = response.choices[0].message.content.strip()
                    except Exception as api_error:
                        print(f"API Error: {api_error}")
                        reasoning = "ERROR: Failed to get response from API"

                    # Store the raw reasoning output
                    data["reasoning"] = reasoning
                    
                    json.dump(data, output_f, ensure_ascii=False)
                    output_f.write("\n")
                    pbar.update(1)

    except Exception as e:
        print(f"Error: {e}")

def load_template(template_file):
    """Loads the template file."""
    with open(template_file, "r", encoding="utf-8") as file:
        return file.read()

def calculate_output_filename(input_file):
    """Generates the output filename."""
    base_name = os.path.splitext(input_file)[0]
    return f"{base_name}_processed.jsonl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON-lines with DeepSeek API.")
    parser.add_argument("--input_file", required=True, help="Input JSON-lines file.")
    parser.add_argument("--template_file", default="deepseek_template.txt", help="Template file.")
    args = parser.parse_args()

    api_key = os.getenv("DeepSeekApi")
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
