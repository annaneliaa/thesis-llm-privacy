import numpy as np
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

default_args = {
    "output_dir": "finetune",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "fp16": True,
    "optim": "adafactor",
}

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

# Set default device
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

logger.info(f"Default device: {DEFAULT_DEVICE}")

# def print_gpu_utilization():
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print(f"GPU memory occupied: {info.used//1024**2} MB.")

# def print_summary(result):
#     print(f"Time: {result.metrics['train_runtime']:.2f}")
#     print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
#     print_gpu_utilization()

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(
    DEFAULT_DEVICE
)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Set the padding token to the EOS token
tokenizer.pad_token = tokenizer.eos_token

print("Model max length:", tokenizer.model_max_length)

# Load the dataset
dataset_file = "EMEA/train.txt"
with open(dataset_file, "r") as f:
    ds = f.readlines()

tokenized_sentences = tokenizer(ds, padding=True, truncation=True)
# the sentences are lists of token ids
print("Number of sentences:", len(tokenized_sentences["input_ids"]))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors="np"
)


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


# Instantiate the dataset
dataset = SentencesDataset(
    tokenized_sentences["input_ids"], tokenized_sentences["attention_mask"]
)

training_args = TrainingArguments(**default_args)
trainer = Trainer(
    model=model, args=training_args, train_dataset=dataset, data_collator=data_collator
)

result = trainer.train()
logger.info("Training finished.")
logger.info(result.metrics)
