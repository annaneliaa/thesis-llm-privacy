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
from util_lib import *

# Configure Python's logging in Jupyter notebook
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set up logger
logger = initLogger()

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
    PREPROCESSING,
    PREPROCESSING_SUFFIX,
    NORMALIZATION,
    BATCHING,
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
    VAL_SPLIT, 
    SEED
) = load_constants_from_config(config)

# Change to .env later
# This is the dir on Habrok where I store all models actively in use
HF_CACHE_DIR = "/scratch/s5202841"

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

tokenizer = initTokenizer(MODEL_NAME)
logger.info("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
    DEFAULT_DEVICE
)
model.resize_token_embeddings(len(tokenizer))

print("Model max length:", tokenizer.model_max_length)

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

# Training args for model 
default_args = {
    "output_dir": output_dir,
    "eval_strategy": "steps",
    "eval_steps": 250,
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
    "per_device_train_batch_size": 8,
    #"learning_rate": 1e-04,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "fp16": True,
    "optim": "adafactor",
}

if args.epochs:
    default_args["num_train_epochs"] = args.epochs

# Load the training and validation sets
source_dir = get_source_directory(SOURCE_DIR, DATASET_DIR, LANGUAGE, PREPROCESSING, NORMALIZATION, EXAMPLE_TOKEN_LEN)
if BATCHING:
    train = torch.load(os.path.join(source_dir, "train-" + LANGUAGE + ".pt"))
else:
    train = torch.load(os.path.join(source_dir, "train-nb-" + LANGUAGE + ".pt"))
    train = [train]
val = torch.load(os.path.join(source_dir, "validation-" + LANGUAGE + ".pt"))

print("Number of validation sentences:", len(val["input_ids"]))

# Instantiate the validation dataset
eval_dataset = SentencesDataset(
    val["input_ids"], val["attention_mask"]
)
# if the input is not in batches, wrap the input. The following loop will simply run for one iteration
if not BATCHING:
    train = [train]
# initialize the trainer
training_args = TrainingArguments(**default_args)
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)
# train the model on all batches of training data
for i,tokenized_sentences in enumerate(train):
    dataset = SentencesDataset(
        tokenized_sentences["input_ids"], tokenized_sentences["attention_mask"]
    )
    trainer.train_dataset = dataset
    logger.info("Training model for %d epochs on batch %d", training_args.num_train_epochs, i)
    result = trainer.train()
    print_summary(result)

logger.info("Training finished.")


logger.info("Saving model to %s", output_dir)
# Save model and tokenizer
trainer.save_model(
    os.path.join(output_dir)
)  # Save the model to the output directory
tokenizer.save_pretrained(
    os.path.join(output_dir)
)  # Save the tokenizer to the same directory

logger.info("==== End of trainer script ====")