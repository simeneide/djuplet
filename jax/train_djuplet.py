import re
import random
from datasets import Dataset, load_dataset
from jax import lax
from jax import numpy as jnp
from transformers import AutoTokenizer
import easydel as ed
from jiwer import wer

repo_id = "pere/llama3.2-3B-chat-reasoning-norwegian"
#repo_id = "qwen/qwen2-0.5b-instruct"
total_batch_size = 4
num_return_sequences = 4
max_prompt_length = 512
max_completion_length = 1024
max_sequence_length = max_completion_length + max_prompt_length

processor = AutoTokenizer.from_pretrained(repo_id)

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    repo_id,
    sharding_axis_dims=(1, -1, 1, 1),
    auto_shard_model=True,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=lax.Precision.DEFAULT,
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_dtype=jnp.bfloat16,
        attn_softmax_dtype=jnp.float32,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
        freq_max_position_embeddings=max_sequence_length,
        mask_max_position_embeddings=max_sequence_length,
    ),
    quantize_tensors=False,
    quantization_method=ed.EasyDeLQuantizationMethods.NONE,
)

#SYSTEM_PROMPT = """
#Respond in the following format:
#<think>
#...
#</think>
#<answer>
#...
#</answer>
#"""

SYSTEM_PROMPT = ""

XML_COT_FORMAT = """\
<think>
{reasoning}
</think>
<answer>
{answer}
</answer>
"""


def split_llama3_text(text: str) -> tuple[str, str]:
    """
    Splits a raw llama3-chat formatted text into prompt and answer parts.
    Assumes that the assistant’s block starts with:
      <|start_header_id|>assistant<|end_header_id|>
    and that its content ends at the next <|eot_id|>.
    
    Returns:
        prompt: Everything before the assistant marker.
        answer: The assistant block (trimmed up to the first <|eot_id|> after the marker).
    """
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
    parts = text.split(assistant_marker, 1)
    if len(parts) < 2:
        # If no assistant block is found, return the whole text as prompt.
        return text.strip(), ""
    prompt = parts[0].strip()
    remainder = parts[1].strip()
    # Remove any trailing end-of-text marker from the assistant part.
    answer = remainder.split("<|eot_id|>", 1)[0].strip()
    return prompt, answer

def get_norwegian_questions(split="train") -> Dataset:
    data = load_dataset("pere/reasoning_chat_norwegian")[split]
    data = data.map(
        lambda x: {
            "prompt": split_llama3_text(x["text"])[0],
            "answer": split_llama3_text(x["text"])[1],
        }
    )
    return data

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def debug_reward_func(prompts, completions, batch, **kwargs) -> list[float]:
    # With roughly a 1 in 10 chance, print all completions for debugging.
    if random.randint(1, 100) == 100:
        for i, completion in enumerate(completions):
            print(f"Completion {i}:")
            print(completion[0])
            print("-" * 40)
    # Return a neutral reward to avoid influencing training.
    return [0.0 for _ in completions]

def correctness_reward_func(prompts, completions, batch, **kwargs) -> list[float]:
    # Extract the assistant's final answer from the generated completions.
    extracted_responses = [extract_xml_answer(c) for c in completions]

    # Decode the reference answers from token ids and extract the text within the <answer> tags.
    decoded_answers = processor.batch_decode(batch["answer_ids"])
    extracted_answers = [extract_xml_answer(a) for a in decoded_answers]

    # Replicate each reference answer to match the number of returned sequences.
    replicated_answers = extracted_answers * num_return_sequences

    # Compare the generated responses with the replicated reference answers.
    return [2.0 if response == reference else 0.0 for response, reference in zip(extracted_responses, replicated_answers)]


def int_reward_func(completions, **kwargs) -> list[float]:
    extracted_responses = [extract_xml_answer(c) for c in completion]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, c) for c in completions]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, c) for c in completions]
    return [0.5 if match else 0.0 for match in matches]

def wer_reward_func(prompts, completions, batch, **kwargs) -> list[float]:
    """
    Computes a reward based on the Word Error Rate (WER) between the generated answer and the ground truth answer.
    The final reward is calculated as: r = min(1.0, max(0.0, 1 - error_rate)).
    
    Both generated and ground truth answers are extracted using the extract_xml_answer function to ensure only the text
    inside <answer> and </answer> is considered.
    """
    # Extract the generated responses and isolate the final answer.
    generated_answers = [extract_xml_answer(c) for c in completions]
    
    # Decode the ground truth answers from token ids, then extract only the text within the <answer> tags.
    decoded_answers = processor.batch_decode(batch["answer_ids"])
    ground_truth_answers = [extract_xml_answer(a) for a in decoded_answers]
    
    # Replicate each ground truth answer to match the number of returned sequences.
    replicated_ground_truth = ground_truth_answers * num_return_sequences
    
    rewards = []
    breakpoint()
    for gen, gt in zip(generated_answers, replicated_ground_truth):
        # Compute error rate using jiwer's wer function.
        error_rate = wer(gen, gt)
        # Calculate reward ensuring it is within the range [0.0, 1.0].
        reward = min(1.0, max(0.0, 1 - error_rate))
        rewards.append(reward)
    return rewards

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c) for c in completions]


arguments = ed.GRPOConfig(
    save_directory="/home/perk/djuplet/jax/output",
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_train_epochs=1,
    total_batch_size=total_batch_size,
    log_steps=10,
    use_wandb=False,
    report_steps=5,
    save_steps=500,
    progress_bar_type="json",
    save_optimizer_state=False,
    do_eval=False,
)

train_dataset = get_norwegian_questions("train")
test_dataset = get_norwegian_questions("test")
vinference = ed.vInference(
    model=model,
    processor_class=processor,
    generation_config=ed.vInferenceConfig(
        bos_token_id=processor.bos_token_id,
        eos_token_id=processor.eos_token_id,
        pad_token_id=processor.pad_token_id,
        do_sample=True,
        max_new_tokens=max_completion_length,
        streaming_chunks=max_completion_length,
        top_k=10,
        top_p=0.95,
        num_return_sequences=num_return_sequences,
    ),
    seed=84,
)

vinference.precompile(total_batch_size, max_prompt_length)


def data_tokenize_fn(batch, tokenizer, tools):
    # Set the pad token on the tokenizer if it isn't already set.
    if tokenizer.pad_token is None:
        # Use the first token from eos_token if it's a list, otherwise use eos_token directly.
        tokenizer.pad_token = tokenizer.eos_token[0] if isinstance(tokenizer.eos_token, list) else tokenizer.eos_token
    # Debug print
    # print(f"DEBUG: batch['answer'] type: {type(batch['answer'])}, value: {batch['answer']}")

    if not isinstance(batch["answer"], (str, list)):  # Ensure answer is a valid type
        raise TypeError(f"Unexpected format for 'answer': {type(batch['answer'])}") 

    ids = tokenizer(
        batch["prompt"],
        return_tensors="np",
        padding="max_length",
        padding_side="left",
        max_length=arguments.max_prompt_length,
        truncation=True,
        add_special_tokens=False,
    )
    ans = tokenizer(
        batch["answer"],
        return_tensors="np",
        padding="max_length",
        padding_side="left",
        max_length=arguments.max_prompt_length,
        truncation=True,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    ids.update({"answer_ids": ans["input_ids"]})
    return ids


trainer = ed.GRPOTrainer(
    model=model,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        wer_reward_func,
        debug_reward_func,
    ],
    processing_class=processor,
    eval_dataset=test_dataset,
    train_dataset=train_dataset,
    arguments=arguments,
    vinference=vinference,
    data_tokenize_fn=data_tokenize_fn,
)

trainer.train()
