import logging
import os
import json
import argparse
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    IntervalStrategy,
    default_data_collator
)
import torch
from torch.utils.data import random_split
import numpy as np
from datasets import Dataset

torch.manual_seed(42)

# Configure Python's logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config["model"])
model = AutoModelForCausalLM.from_pretrained(config["model"])
# Move model to GPU if possible
model.to(DEFAULT_DEVICE).half().eval()

# Load tokenized dataset
dataset_base = os.path.join(
    config["source_dir"],
    config["dataset_dir"],
    config["language"],
    str(config["example_token_len"]),
    config["model"],
    "train_dataset.npy",
)

tokenized_dataset = np.load(dataset_base).astype(np.int64)
attention_masks = (tokenized_dataset != 0)
attention_masks = torch.tensor(attention_masks)
logger.info("Loaded dataset: %s and created attentio masks", dataset_base)

output_dir = os.path.join("finetuned", config["experiment_name"])
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# all sentence are same lenght for now
# find the longest sequence in the dataset
# max_len = max([len(seq) for seq in tokenized_dataset])
# logger.info("Max sequence length:%s", max_len)

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

# Convert the numpy array to a Hugging Face Dataset
dataset = Dataset.from_dict({
    "input_ids": tokenized_dataset.tolist(),
    "attention_mask": attention_masks.tolist()
})

print(dataset[0])

train_size = int(len(dataset) * 0.9)
eval_size = len(dataset) - train_size

# Create a range of indices to map to exids later
indices = list(range(len(dataset)))

# Split the indices along with the dataset
train_indices, eval_indices = random_split(indices, [train_size, eval_size])
# Convert the indices to numpy arrays for easier indexing
train_indices = np.array(train_indices)
eval_indices = np.array(eval_indices)

# Create the train and eval datasets using the indices
train_dataset = dataset[train_indices]
eval_dataset = dataset[eval_indices]

# Save indices to a file
with open(os.path.join(output_dir, "indices.json"), "w") as f:
    json.dump(
        {"train_indices": train_indices.tolist(), 
         "eval_indices": eval_indices.tolist()}, f
    )
            
# train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

train_batch = np.stack(train_dataset, axis=0)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # data_collator={
    #     "input_ids": lambda data: torch.tensor(data["input_ids"], dtype=torch.int64),
    #     "attention_mask": lambda data: torch.tensor(data["attention_mask"]),
    # }
    data_collator=default_data_collator
)

# trainer.train()

# Save the model
trainer.save_model(output_dir)
logger.info("Training completed and model saved.")
