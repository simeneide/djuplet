#%%
import logging
import os
from dataclasses import dataclass, fields
from datetime import datetime
import logging
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import random
import re 
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
import yaml
#%%

########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Helper functions
########################
from langdetect import detect, DetectorFactory, detect_langs
def language_reward(completions, **kwargs):
    """
    Checks whether the thinking part of the completion is in Norwegian.
    Assumes that the generated string starts with "<think>".
    
    Args:
        completions = [" dette er en stor tanke </think> tull <answer>1</answer> </answer> </answer>"] #Generated completions.
    
    Returns:
        list[float]: Reward scores (1.0 if Norwegian, otherwise 0.0).
    """
    DetectorFactory.seed = 0  # for reproducible results
    
    rewards = []
    for text in completions:
        try:
            completion = "<think>" + text
            # Extract the content within the <think>...</think> tags
            match = re.search(r"<think>([\s\S]*?)</think>", completion, re.DOTALL)
            if match:
                think_content = match.group(1).strip()
                if think_content:
                    detected_lang = detect(think_content)
                    rewards.append(1.0 if detected_lang == "no" else 0.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

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
                log_file = os.path.join("completion_samples", "completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n============== format_reward={reward}, match_groups={num_groups} ==============\n")
                    f.write(completion)
            
            rewards.append(reward)
        except Exception:
            rewards.append(0.0)
    return rewards
    

def equation_reward_func(completions, target, nums, **kwargs):
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
    for completion, gt, numbers in zip(completions, target, nums):
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards.append(0.0)
            continue
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            rewards.append(0.0)
            continue
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
           rewards.append(0.0)
           continue
        
        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(gt)) < 1e-5:
            rewards.append(1.0)
            if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)
        else:
            rewards.append(0.0)
      except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards

from jiwer import wer
def corrupt_reward_func(completions, original_text, **kwargs):
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
    for completion, ground_truth in zip(completions, original_text):
      try:
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
        if r==1.0:
            if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)

      except Exception as e:
            print(e)
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards

def corrupt_reward_binary_func(completions, original_text, **kwargs):
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
    for completion, ground_truth in zip(completions, original_text):
      try:
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
        score = min(1.0, max(0.0, 1- error_rate))
        
        # Check if the equation is correct and matches the ground truth
        if score>=0.98:
            rewards.append(1.0)
            if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_completion_samples_binary.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)
        else:
            rewards.append(0.0)

      except Exception as e:
            print(e)
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    dataset = load_dataset(script_args.dataset_id_or_path)
    # select a random subset of 50k samples
    #dataset = dataset.shuffle(seed=42).select(range(50000))

    #####################
    # Prepare and format dataset
    #####################

    # Read prompt template
    template_path = "../../templates/prompt_template_norwegian.txt"
    with open(template_path, 'r') as file:
        prompt_template = file.read()


    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(corrupt, original_text):

        prompt = f"""{prompt_template} {corrupt} <think> """

        return {"prompt": prompt, "corrupt": corrupt, "original_text": original_text}

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x["corrupt"], x["original_text"]))

    #########################
    # Instantiate DPO trainer
    #########################

    trainer = GRPOTrainer(
      model=model_args.model_name_or_path,
      processing_class=tokenizer,
      reward_funcs=[format_reward_func, corrupt_reward_func, corrupt_reward_binary_func, language_reward],
      args=training_args,
      train_dataset=dataset['train'],
      eval_dataset=dataset['validation'],
      peft_config=get_peft_config(model_args),
    )


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def load_config(yaml_file: str) -> dict:
    """Load configuration from a YAML file."""
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def filter_dataclass_args(dataclass_cls, config):
    """
    Filters a configuration dictionary and returns only the keys
    that are valid fields for the given dataclass.
    """
    allowed_fields = {field.name for field in fields(dataclass_cls)}
    return {key: value for key, value in config.items() 
            if key in allowed_fields and value is not None}

if __name__ == "__main__":
    #config = load_config("norsk_zero.yaml")
    config = load_config("norsk_warmstart.yaml")

    model_args = ModelConfig(**filter_dataclass_args(ModelConfig, config))
    script_args = ScriptArguments(**filter_dataclass_args(ScriptArguments, config))
    training_args  = GRPOConfig(**filter_dataclass_args(GRPOConfig, config))
    # Run the main training loop
    grpo_function(model_args, script_args, training_args)

# %%
