#!/usr/bin/env python3
import os
import json
import argparse
import threading
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

# Thread-local storage for the API client.
_thread_local = threading.local()

def get_client(api_key):
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
    return _thread_local.client

def accumulate_stream_response(response):
    full_text = ""
    for chunk in response:
        delta = chunk.choices[0].delta
        full_text += delta.get("content", "")
    full_text = full_text.strip()
    if full_text.startswith("<think>"):
        if "</think>" in full_text:
            reasoning_part, final_answer = full_text.split("</think>", 1)
            reasoning = reasoning_part.replace("<think>", "").strip()
            final_answer = final_answer.strip()
        else:
            reasoning = full_text.replace("<think>", "").strip() + " [INCOMPLETE: missing </think> tag]"
            final_answer = ""
    else:
        reasoning = ""
        final_answer = full_text
    return final_answer, reasoning

def process_record(line, template, api_key, stream_output):
    try:
        data = json.loads(line.strip())
    except json.JSONDecodeError:
        print(line.strip())
        return None
    if "text" not in data:
        return None
    original_text = data["text"]
    user_prompt = template.format(text=original_text)
    client = get_client(api_key)
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": user_prompt},
            ],
            stream=stream_output,
            max_tokens=2000
        )
    except Exception:
        print(json.JSONDecodeError)
        final_answer = "ERROR: Failed to get response from API"
        api_reasoning = "ERROR: Failed to get response from API"
    else:
        try:
            if stream_output:
                final_answer, api_reasoning = accumulate_stream_response(response)
            else:
                message = response.choices[0].message
                final_answer = message.content.strip() if message.content else ""
                if hasattr(message, "reasoning_content") and message.reasoning_content:
                    api_reasoning = message.reasoning_content.strip()
                else:
                    if final_answer.startswith("<think>"):
                        if "</think>" in final_answer:
                            reasoning_part, final_answer = final_answer.split("</think>", 1)
                            api_reasoning = reasoning_part.replace("<think>", "").strip()
                            final_answer = final_answer.strip()
                        else:
                            api_reasoning = final_answer.replace("<think>", "").strip() + " [INCOMPLETE: missing </think> tag]"
                            final_answer = ""
                    else:
                        api_reasoning = ""
        except Exception:
            print(Exception)
            final_answer = "ERROR: Failed to extract final answer"
            api_reasoning = "ERROR: Failed to extract reasoning"
    data["original_text"] = original_text
    data["text_result"] = final_answer
    data["reasoning"] = api_reasoning
    if "text" in data:
        del data["text"]
    return json.dumps(data, ensure_ascii=False)

def process_file_parallel(input_file, template, output_file, api_key, stream_output, num_workers, write_immediately):
    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as out_f:
            processed_count = sum(1 for _ in out_f)
    with open(input_file, "r", encoding="utf-8") as in_f:
        lines = in_f.readlines()
    lines_to_process = lines[processed_count:]
    total = len(lines_to_process)
    if total == 0:
        return
    if write_immediately:
        out_file = open(output_file, "a", encoding="utf-8")
    else:
        results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for processed in tqdm(
            executor.map(process_record, lines_to_process, repeat(template), repeat(api_key), repeat(stream_output)),
            total=total,
            desc="Processing"
        ):
            if processed is not None:
                if write_immediately:
                    out_file.write(processed + "\n")
                    out_file.flush()
                else:
                    results.append(processed)
    if not write_immediately:
        with open(output_file, "a", encoding="utf-8") as out_f:
            for record in results:
                out_f.write(record + "\n")
    else:
        out_file.close()

def load_template(template_file):
    with open(template_file, "r", encoding="utf-8") as file:
        return file.read()

def calculate_output_filename(input_file):
    base_name = os.path.splitext(input_file)[0]
    return f"{base_name}_processed.jsonl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON-lines with DeepSeek API in parallel.")
    parser.add_argument("--input_file", required=True, help="Input JSON-lines file.")
    parser.add_argument("--template_file", default="../templates/deepseek_template_norwegian.txt", help="Template file.")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode for the API call.")
    parser.add_argument("--processes", type=int, default=10, help="Number of parallel workers (default: 10).")
    parser.add_argument("--immediate", action="store_true", help="Write output immediately after processing each record.")
    args = parser.parse_args()
    api_key = os.getenv("DEEP_INFRA")
    if not api_key:
        raise EnvironmentError("DEEP_INFRA environment variable not set.")
    template_content = load_template(args.template_file)
    output_file = calculate_output_filename(args.input_file)
    process_file_parallel(
        input_file=args.input_file,
        template=template_content,
        output_file=output_file,
        api_key=api_key,
        stream_output=args.stream,
        num_workers=args.processes,
        write_immediately=args.immediate
    )

