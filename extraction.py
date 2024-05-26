import logging
from IPython.display import display
import os
from typing import Tuple, Union
import numpy as np
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

BATCH_SIZE = 32

# Configure Python's logging in Jupyter notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s")

class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))

def generations_to_jsonl(output_file_path: str, generations: np.ndarray):
    """Converts the `generations` to a JSONL file at `path`."""
    with open(output_file_path, "w", encoding="utf-8", newline='') as file:
        exid = 0
        for row in generations:
            decoded_string = tokenizer.decode(row, skip_special_tokens=True).replace('Ġ', '').replace("Ä", '')
            line = decoded_string.strip()
            if not line:
                continue
            json_object = {"exid": exid, "text": line}
            json.dump(json_object, file, ensure_ascii=False)
            file.write("\n")
            exid += 1
    logger.info("Decoded strings saved to: %s", str(output_file_path))

def generate_for_prompts(prompts: np.ndarray, batch_size: int, suffix_len: int, prefix_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates suffixes given `prompts` and scores using their likelihood."""
    generations = []
    losses = []
    generation_len = suffix_len + prefix_len
    for i, off in enumerate(range(0, len(prompts), batch_size)):
        prompt_batch = prompts[off: off + batch_size]
        logger.info(f"Generating for batch ID {i:05} of size {len(prompt_batch):04}")
        prompt_batch = np.stack(prompt_batch, axis=0)
        input_ids = torch.tensor(prompt_batch, dtype=torch.int64).to(DEFAULT_DEVICE)
        with torch.no_grad():
            generated_tokens = (
                MODEL.generate(
                    input_ids.to(DEFAULT_DEVICE),
                    max_length=generation_len,
                    do_sample=True,
                    top_k=10,
                    top_p=1,
                    pad_token_id=50256,  # Silences warning.
                ).to('cpu').detach()
            )
            outputs = MODEL(generated_tokens.to(DEFAULT_DEVICE), labels=generated_tokens.to(DEFAULT_DEVICE))
            logits = outputs.logits.cpu().detach()
            logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
            loss_per_token = torch.nn.functional.cross_entropy(
                logits, generated_tokens[:, 1:].flatten(), reduction="none"
            )
            loss_per_token = loss_per_token.reshape((-1, generation_len - 1))[:, -suffix_len - 1: -1]
            likelihood = loss_per_token.mean(1)
            generations.extend(generated_tokens.numpy())
            losses.extend(likelihood.numpy())
    return np.atleast_2d(generations), np.atleast_2d(losses).reshape((len(generations), -1))

def write_array(file_path: str, array: np.ndarray, unique_id: Union[int, str]):
    """Writes a batch of `generations` and `losses` to a file."""
    file_ = file_path.format(unique_id)
    np.save(file_, array)

def load_prompts(dir_: str, file_name: str) -> np.ndarray:
    """Loads prompts from the file pointed to `dir_` and `file_name`."""
    return np.load(os.path.join(dir_, file_name)).astype(np.int64)

# Set up logger
logger = logging.getLogger()
handler = JupyterHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
    MODEL.to(DEFAULT_DEVICE)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    ROOT_DIR = config["root_dir"]
    EXPERIMENT_NAME = config["experiment_name"]
    DATASET_DIR = config["dataset_dir"]
    DATASET_FILE = config["dataset_file"]
    NUM_TRIALS = config["num_trials"]
    SUFFIX_LEN = config["suffix_len"]
    PREFIX_LEN = config["prefix_len"]
    SOURCE_DIR = config["source_dir"]
    LANGUAGE = config["language"]
    EXAMPLE_TOKEN_LEN = config["example_token_len"]

    logger.info("======= Starting extraction ======")
    logger.info("Number of trials: %d", NUM_TRIALS)

    experiment_base = os.path.join(ROOT_DIR, LANGUAGE, EXPERIMENT_NAME)
    generations_base = os.path.join(experiment_base, "generations")
    os.makedirs(generations_base, exist_ok=True)
    losses_base = os.path.join(experiment_base, "losses")
    os.makedirs(losses_base, exist_ok=True)
    prompts_base = os.path.join(SOURCE_DIR, LANGUAGE, str(EXAMPLE_TOKEN_LEN))
    prompts = load_prompts(prompts_base, "train_prefix.npy")[-1000:]

    all_generations, all_losses = [], []
    if not all([os.listdir(generations_base), os.listdir(losses_base)]):
        for trial in range(NUM_TRIALS):
            os.makedirs(experiment_base, exist_ok=True)
            generations, losses = generate_for_prompts(prompts, BATCH_SIZE, SUFFIX_LEN, PREFIX_LEN)
            input_tokens = prompts

            logger.info(f"Trial {trial}: Generated {len(generations)} generations.")

            generation_string = os.path.join(generations_base, "{}.npy")
            losses_string = os.path.join(losses_base, "{}.npy")

            write_array(generation_string, generations, trial)
            write_array(losses_string, losses, trial)

            all_generations.append(generations)
            all_losses.append(losses)

        generations = np.stack(all_generations, axis=1)
        losses = np.stack(all_losses, axis=1)
    else:
        generations = []
        for generation_file in sorted(os.listdir(generations_base)):
            file_ = os.path.join(generations_base, generation_file)
            generations.append(np.load(file_))
        generations = np.stack(generations, axis=1)

        losses = []
        for losses_file in sorted(os.listdir(losses_base)):
            file_ = os.path.join(losses_base, losses_file)
            losses.append(np.load(file_))
        losses = np.stack(losses, axis=1)

    for i in range(0, NUM_TRIALS):
        file_path = os.path.join(experiment_base, f"generations/{i}.npy")
        data = np.load(file_path)
        logger.info("Data shape: %s", str(data.shape))

        output_file_path = os.path.join(experiment_base, f"decoded/decoded_strings_trial_{i}.jsonl")
        output_dir = os.path.dirname(output_file_path)
        os.makedirs(output_dir, exist_ok=True)
        generations_to_jsonl(output_file_path, data)

    logger.info("====== Done ======")
