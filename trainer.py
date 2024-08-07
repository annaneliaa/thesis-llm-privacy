from pynvml import *
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset
import torch
import logging
from IPython.display import display
import os
import argparse
import json
from experiment_lib import load_constants_from_config

# Configure Python's logging in Jupyter notebook
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))

# Set up logger
logger = logging.getLogger()
handler = JupyterHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument(
    "--config_file", type=str, required=True, help="Path to the configuration file"
)
parser.add_argument("--epochs", type=int, required=False, help="Number of epochs to train for")

args = parser.parse_args()

# Load configuration files
with open(args.config_file, "r") as f:
    config = json.load(f)

(
    ROOT_DIR, 
    DATASET_DIR, 
    SOURCE_DIR, 
    DATASET_NAME, 
    EXPERIMENT_NAME, 
    NUM_TRIALS, 
    PREFIX_LEN, 
    SUFFIX_LEN, 
    PREPREFIX_LEN, 
    LANGUAGE, 
    SPLIT, 
    EXAMPLE_TOKEN_LEN, 
    SOURCE_FILE, 
    BATCH_SIZE, 
    MODEL_NAME, 
    TRAIN_FILE, 
    VAL_FILE, 
    VAL_SPLIT, 
    SEED
) = load_constants_from_config(config)

# Change to .env later
# This is the dir on Habrok where I store all models actively in use
HF_CACHE_DIR = "/scratch/s4079876"

# Set up trainer
output_dir = os.path.join(HF_CACHE_DIR, "finetuned", DATASET_DIR, EXPERIMENT_NAME)

logger.info("Saving trained model to %s", output_dir)

# Set default device
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

logger.info(f"Default device: {DEFAULT_DEVICE}")

# Functions for insight in GPU usage
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

logger.info("==== Starting trainer script ====")

logger.info("Experiment name %s", EXPERIMENT_NAME)

logger.info("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
    DEFAULT_DEVICE
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set the padding token to the EOS token
tokenizer.pad_token = tokenizer.eos_token

print("Model max length:", tokenizer.model_max_length)

# Load the training and validation sets
# Read and tokenize training dataset
with open(TRAIN_FILE, "r") as f:
    train = f.readlines()

tokenized_sentences = tokenizer(train, padding=True, truncation=True, return_tensors="pt")
# the sentences are lists of token ids
print("Number of sentences:", len(tokenized_sentences["input_ids"]))

# Read and tokenize evaluation dataset
with open(VAL_FILE, "r") as f:
    val = f.readlines()

tokenized_eval_sentences = tokenizer(val, padding=True, truncation=True, return_tensors="pt")
print("Number of validation sentences:", len(tokenized_eval_sentences["input_ids"]))

# Training set up
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

class SentencesDataset(Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
        }
        return item

# Instantiate the datasets
dataset = SentencesDataset(
    tokenized_sentences["input_ids"], tokenized_sentences["attention_mask"]
)
eval_dataset = SentencesDataset(
    tokenized_eval_sentences["input_ids"], tokenized_eval_sentences["attention_mask"]
)

# Training args for model 
default_args = {
    "output_dir": output_dir,
    "evaluation_strategy": "steps",
    "eval_steps": 1000,
    # save steps is a high number to avoid overflow of storage disk on Habrok (we dont want to store all intermediate versions of the model)
    "save_steps": 10000,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    # default is 1, unless specified on command line
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "fp16": True,
    "optim": "adafactor",
}

if args.epochs:
    default_args["num_train_epochs"] = args.epochs

training_args = TrainingArguments(**default_args)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

logger.info("Training model for %d epochs", training_args.num_train_epochs)
result = trainer.train()
logger.info("Training finished.")

print_summary(result)

logger.info("Saving model to %s", output_dir)
# Save model and tokenizer
trainer.save_model(
    os.path.join(output_dir)
)  # Save the model to the output directory
tokenizer.save_pretrained(
    os.path.join(output_dir)
)  # Save the tokenizer to the same directory

logger.info("==== End of trainer script ====")
