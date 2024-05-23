import logging
from IPython.display import display

import csv
import os
import tempfile
from typing import Tuple, Union
import numpy as np
import transformers
import torch
import accelerate

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure Python's logging in Jupyter notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s")


class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))


logger = logging.getLogger()
handler = JupyterHandler()
logger.addHandler(handler)

logger.setLevel(logging.INFO)

# Define constants directly since we're not using flags in a Jupyter notebook
ROOT_DIR = "tmp/"
EXPERIMENT_NAME = "test"
DATASET_DIR = "./datasets"
DATASET_FILE = "train_dataset.npy"
NUM_TRIALS = 10

SUFFIX_LEN = 50
PREFIX_LEN = 50

if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

print(f"Default device: {DEFAULT_DEVICE}")

try:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    MODEL = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-125M", low_cpu_mem_usage=True
    )
    # move the model to MPS device if available
    # Now every model call runs on the GPU (M1 mac)
    MODEL.to(DEFAULT_DEVICE)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")


def generate_for_prompts(
    prompts: np.ndarray, batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates suffixes given `prompts` and scores using their likelihood."""
    generations = []
    losses = []
    generation_len = SUFFIX_LEN + PREFIX_LEN

    for i, off in enumerate(range(0, len(prompts), batch_size)):
        prompt_batch = prompts[off : off + batch_size]
        logger.info(f"Generating for batch ID {i:05} of size {len(prompt_batch):04}")
        prompt_batch = np.stack(prompt_batch, axis=0)

        # input prompts are assumed to be tokenized already
        input_ids = torch.tensor(prompt_batch, dtype=torch.int64).to(DEFAULT_DEVICE)
        with torch.no_grad():
            # 1. Generate outputs from the model
            generated_tokens = (
                MODEL.generate(
                    input_ids.to(DEFAULT_DEVICE),
                    max_length=generation_len,
                    do_sample=True,
                    top_k=10,
                    top_p=1,
                    pad_token_id=50256,  # Silences warning.
                )
                .to('cpu').detach()
            )

            # 2. Compute each sequence's probability, excluding EOS and SOS.
            outputs = MODEL(generated_tokens.to(DEFAULT_DEVICE), labels=generated_tokens.to(DEFAULT_DEVICE))
            logits = outputs.logits.cpu().detach()
            logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
            loss_per_token = torch.nn.functional.cross_entropy(
                logits, generated_tokens[:, 1:].flatten(), reduction="none"
            )
            loss_per_token = loss_per_token.reshape((-1, generation_len - 1))[
                :, -SUFFIX_LEN - 1 : -1
            ]
            likelihood = loss_per_token.mean(1)

            generations.extend(generated_tokens.numpy())
            losses.extend(likelihood.numpy())
    return np.atleast_2d(generations), np.atleast_2d(losses).reshape(
        (len(generations), -1)
    )

def write_array(file_path: str, array: np.ndarray, unique_id: Union[int, str]):
    """Writes a batch of `generations` and `losses` to a file."""
    file_ = file_path.format(unique_id)
    np.save(file_, array)


def load_prompts(dir_: str, file_name: str) -> np.ndarray:
    """Loads prompts from the file pointed to `dir_` and `file_name`."""
    return np.load(os.path.join(dir_, file_name)).astype(np.int64)


logger.info("======= Starting extraction ======")

experiment_base = os.path.join(ROOT_DIR, EXPERIMENT_NAME)
generations_base = os.path.join(experiment_base, "generations")
os.makedirs(generations_base, exist_ok=True)
losses_base = os.path.join(experiment_base, "losses")
os.makedirs(losses_base, exist_ok=True)
prompts = load_prompts(DATASET_DIR, "train_prefix.npy")[-1000:]

# We by default do not overwrite previous results.
all_generations, all_losses = [], []
if not all([os.listdir(generations_base), os.listdir(losses_base)]):
    for trial in range(NUM_TRIALS):
        os.makedirs(experiment_base, exist_ok=True)
        generations, losses = generate_for_prompts(prompts)

        logger.info(f"Trial {trial}: Generated {len(generations)} generations.")

        generation_string = os.path.join(generations_base, "{}.npy")
        losses_string = os.path.join(losses_base, "{}.npy")

        write_array(generation_string, generations, trial)
        write_array(losses_string, losses, trial)

        all_generations.append(generations)
        all_losses.append(losses)
    generations = np.stack(all_generations, axis=1)
    losses = np.stack(all_losses, axis=1)
else:  # Load saved results because we did not regenerate them.
    generations = []
    for generation_file in sorted(os.listdir(generations_base)):
        file_ = os.path.join(generations_base, generation_file)
        generations.append(np.load(file_))
    # Generations, losses are shape [num_prompts, num_trials, suffix_len].
    generations = np.stack(generations, axis=1)

    losses = []
    for losses_file in sorted(os.listdir(losses_base)):
        file_ = os.path.join(losses_base, losses_file)
        losses.append(np.load(file_))
    losses = np.stack(losses, axis=1)

# Decode the generated tokens and save the generated text to a file
for i in range(0, NUM_TRIALS):
    # Specify the file path of the .npy file
    file_path = os.path.join(experiment_base, f"/generations/{i}.npy")
    # Load the .npy file
    data = np.load(file_path)
    # Convert token IDs to strings
    text_data = [tokenizer.decode(ids, skip_special_tokens=False).replace('Ġ', '') for ids in data]
    output_file_path = os.path.join(experiment_base, f"/tokenized_strings_{i}.txt")
    # Save the tokenized strings to a new file
    with open(output_file_path, "w") as file:
        for text in text_data:
            file.write(text + "\n")
    print("Tokenized strings saved to:", output_file_path)

logger.info("====== Done ======")
