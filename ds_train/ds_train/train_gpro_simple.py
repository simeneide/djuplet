#%%# train_grpo.py
from datasets import load_dataset, load_from_disk
from trl import GRPOConfig, GRPOTrainer

import re
import random
from jiwer import wer
import os

ds_summary = load_dataset("json", data_dir="article_summaries/")

prompt_template = f"Suggest a good summary for the article in list form. Write in the same language as the article. Think step by step, and show your work in enclosed <think> tags and return the final answer in enclosed <answer> tags, like , like <think> your thinking </think>\n<answer>['first point', 'second point', ...]</answer>. The article:"

def generate_prompt(article_text_all, original_summary):

    prompt = f"""{prompt_template}\n{article_text_all}\n<think>"""

    return {"prompt": prompt, "original_summary" : str(original_summary)}

dataset = ds_summary.map(lambda x: generate_prompt(x["article_text_all"], x["summary"])).select_columns(["prompt", "original_summary"])


test_completion = """The article mentions several key points about Victoria Haglund and her viral moment at school. First, it details how she won the Best Entrance award, which she attributes partly to her appearance in a ballroom costume. Next, it confirms the context of how she initially didn't get close to ballet, which became a significant part of her experience. We also note her experience with hosing named Nino, and how it was eventually introduced into her outfit for the event. The most striking feature is the viral popularity of the video, and the impact it has had on her. Given these points, let's list them out accordingly.
</think>
<answer>
['Victoria Haglund won the Best Entrance award', 'Her involvement in ballet was pivotal to her experience', 'Her hosing, named Nino, was introduced into her outfit for the event', 'Video of her red in the ballroom costume became viral, seeing over 178,000 views', 'It brought her much positive reaction from others']
</answer>"""

test_original_summary = """['Mikael, 39, sköts till döds framför sin 12-åriga son när han konfronterade ett ungdomsgäng i Skärholmen, Stockholm.',
  'Mikael hade tidigare kontaktat polisen om gängen. Han ville inte att hans son skulle växa upp omgiven av narkotika och kriminalitet.',
  'Hans svåger ifrågasätter hur det kommit sig att ungdomar i området går runt beväpnade.']"""
completions = [test_completion]*2
original_summary = [test_original_summary]*2

def format_reward_func(completions, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
    Returns:
        list[float]: Reward scores
    # Example completion: " dette er </think> tull <answer>1</answer> </answer> </answer>"
    """
    rewards = []

    for completion in completions:
        try:
            # Prepend synthetic <think> as it's already part of the prompt to ease matching
            completion = "<think>" + completion
            
            # Check if the format is correct using regex
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>[\s\S]*?<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            
            # Determine reward based on format match: 1.0 if valid, 0.0 if not
            if match is None or len(match.groups()) != 2:
                reward = 0.0
                num_groups = 0
            else:
                reward = 1.0
                num_groups = len(match.groups())
            
            # Optionally log a sample with its reward and number of match groups (10% chance)
            if random.random() < 0.1:
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "format_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(f"format reward: {reward}, match_groups={num_groups}\n")
                    f.write(completion)
            if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(f"wer reward: {r}\n")
                    f.write(completion)
            rewards.append(reward)
        except Exception:
            rewards.append(0.0)
    return rewards

def word_error_rate_reward_func(completions, original_summary, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, ground_truth in zip(completions, original_summary):
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards.append(0.0)
            continue
        # Extract the "answer" part from the completion
        generated_answer = match.group(1).strip()
        # Extract all numbers from the equation

        error_rate = wer(generated_answer, ground_truth)
        r = min(1.0, max(0.0, 1- error_rate))
        rewards.append(r)
        # Check if the equation is correct and matches the ground truth
        
        if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
            os.makedirs("completion_samples", exist_ok=True)
            log_file = os.path.join("completion_samples", "completion_samples.txt")
            with open(log_file, "a") as f:
                f.write(f"\n\n==============\n")
                f.write(f"wer reward: {r}\n")
                f.write(completion)

    return rewards

base_model ="Qwen/Qwen2.5-3B-Instruct"
output_dir = "results_summary"
training_args = GRPOConfig(output_dir=output_dir, logging_steps=2, use_vllm=True, vllm_device="cuda:1", per_device_train_batch_size=8, gradient_accumulation_steps=3, vllm_gpu_memory_utilization=0.3)
trainer = GRPOTrainer(
    model=base_model,
    reward_funcs=[word_error_rate_reward_func, format_reward_func],
    args=training_args,
    train_dataset=dataset['train'],
)
trainer.train()