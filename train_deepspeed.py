import logging
import os
import json
import argparse
import numpy as np
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    IntervalStrategy
)
import torch
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
import numpy as np
from datasets import Dataset
import wandb
from experiment_lib import *

import math
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from datasets import load_dataset, Dataset

# Configure Python's logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Tracking runs with wandb
wandb_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_key)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument(
    "--config_file", type=str, required=True, help="Path to the configuration file"
)
parser.add_argument(
    "--deepspeed_config",
    type=str,
    required=False,
    help="Path to the DeepSpeed config file",
)
args = parser.parse_args()

# Load configuration files
with open(args.config_file, "r") as f:
    config = json.load(f)

with open(args.deepspeed_config, "r") as f:
    deepspeed_config = json.load(f)
    print(json.dumps(deepspeed_config, indent=4))

# Set default device
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

logger.info(f"Default device: {DEFAULT_DEVICE}")

# Set seed before loading the model
set_seed(config["seed"])

# output dir for trained models, and training/eval files
output_dir = os.path.join("finetuned", config["dataset_dir"], config["experiment_name"])
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load dataset
dataset_path = os.path.join(
    config["dataset_dir"], config["dataset_name"] + "." + config["language"]
)

eval_percentage = config["validation_split_percentage"]
split_set_to_train_val(eval_percentage, dataset_path, output_dir, config)

logger.info("Created train.csv and validation.csv files.")

data_files = {}
if config["train_file"] is not None:
    data_files["train"] = config["train_file"]
if config["validation_file"] is not None:
    data_files["validation"] = config["validation_file"]

# take file extension of the training file, else of the validation file
# do you need this?
extension = (
    config["validation_file"].split(".")[-1]
    if config["validation_file"] is not None
    else config["validation_file"].split(".")[-1]
)
if extension == "txt":
    extension = "text"
# Create hugging face dataset objects for training and validation
datasets = load_dataset(extension, data_files=data_files)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config["model"])
model = AutoModelForCausalLM.from_pretrained(config["model"])
# Move model to GPU if possible
# model.to(DEFAULT_DEVICE).half().eval()

# Preprocess datasets
column_names = datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]


def tokenize_function(examples):
    return tokenizer(examples[text_column_name])


tokenized_datasets = datasets.map(
    tokenize_function, batched=True, remove_columns=column_names, num_proc=4
)

block_size = tokenizer.model_max_length


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=4,
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    do_train=True,
    num_train_epochs=1,
    per_device_train_batch_size=15,
    per_device_eval_batch_size=15,
    logging_steps=100,
    logging_dir="./logs",
    save_strategy=IntervalStrategy.EPOCH,
    eval_strategy=IntervalStrategy.EPOCH,
    gradient_accumulation_steps=deepspeed_config["gradient_accumulation_steps"],
    evaluation_strategy="no",
    learning_rate=deepspeed_config["optimizer"]["params"]["lr"],
    warmup_steps=8,
    fp16=True,
    deepspeed=args.deepspeed_config,
)

train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=default_data_collator,
)

# Training
train_result = trainer.train(resume_from_checkpoint=None)

metrics = train_result.metrics

metrics["train_samples"] = len(train_dataset)   

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Evaluation
metrics = trainer.evaluate()
metrics["eval_samples"] = len(eval_dataset)
perplexity = math.exp(metrics["eval_loss"])
metrics["perplexity"] = perplexity

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# Save model
trainer.save_model(output_dir)

logger.info("====== Done ======")
