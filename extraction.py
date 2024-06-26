import logging
from IPython.display import display
import os
from typing import Tuple, Union
import numpy as np
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from experiment_lib import *

# Configure Python's logging in Jupyter notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))

# Set up logger
logger = logging.getLogger()
handler = JupyterHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info("Parsing arguments...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
parser.add_argument("--model_dir", type=str, required=False, help="Path to the directory with the saved model")
parser.add_argument("--cache_dir", type=str, required=False, help="Path to the cache directory")

args = parser.parse_args()

with open(args.config_file, 'r') as f:
    config = json.load(f)

# Load constants from config
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

HGmodel = MODEL_NAME

if args.model_dir:
    # Path to finetuned model is provided
    MODEL_NAME = args.model_dir
    logger.info(f"Model directory provided: {MODEL_NAME}")
    logger.info("Executing extraction on finetuned model.")
else:
    logger.info("Model directory not provided, using default model specified in config.")

# Set default device
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

logger.info(f"Default device: {DEFAULT_DEVICE}")

if args.cache_dir:
    cache_dir = args.cache_dir
else:
    # Get cache dir from .env
    cache_dir = "/scratch/s4079876"

# Load model and tokenizer
try:
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    logger.info("Loading model...")
    MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True, cache_dir=cache_dir)
    # move model to GPU
    MODEL.to(DEFAULT_DEVICE)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

logger.info("Experiment name: %s", EXPERIMENT_NAME)
logger.info("Language: %s", LANGUAGE)
logger.info("Model: %s", MODEL_NAME)


def generate_for_prompts(prompts: np.ndarray, batch_size: int, suffix_len: int, prefix_len: int, preprefix_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates suffixes given `prompts` and scores using their likelihood."""
    generations = []
    losses = []
    # Include preprefix length in generation length
    generation_len = preprefix_len + prefix_len + suffix_len
    for i, off in enumerate(range(0, len(prompts), batch_size)):
        prompt_batch = prompts[off: off + batch_size]
        # logger.info(f"Generating for batch ID {i:05} of size {len(prompt_batch):04}")
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

def write_array(file_path: str, array: np.ndarray, unique_id):
    """Writes a batch of `generations` and `losses` to a file."""
    # Convert unique_id to string before formatting file path
    file_ = file_path.format(str(unique_id))
    np.save(file_, array)
    
# # Load prompts
# If the experiment is done with context (preprefix), load the preprefix prompts
# and concatenate them with the prompts
def load_prompts(dir_: str, file_name: str, preprefix_len: int, split: int) -> np.ndarray:
    """Loads prompts from the file pointed to `dir_` and `file_name`."""
    prompts = np.load(os.path.join(dir_, file_name)).astype(np.int64)
    
    if preprefix_len == 0:
        return prompts 
    
    else:
        preprefix_prompts = np.load(os.path.join(dir_, f"{split}_preprefix.npy")).astype(np.int64)
        prompts = np.concatenate((preprefix_prompts, prompts), axis=1)

        return prompts

def main(): 
    logger.info("======= Starting extraction ======")

    logger.info("Number of trials: %d", NUM_TRIALS)

    logger.info("Creating paths...")
    experiment_base = os.path.join(ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME)
    generations_base = os.path.join(experiment_base, "generations")
    os.makedirs(generations_base, exist_ok=True)
    losses_base = os.path.join(experiment_base, "losses")
    os.makedirs(losses_base, exist_ok=True)
    prompts_base = os.path.join(SOURCE_DIR, DATASET_DIR, LANGUAGE, str(EXAMPLE_TOKEN_LEN), HGmodel)

    logger.info("Loading prompts from numpy file")
    prompts = load_prompts(prompts_base, SPLIT + "_prefix.npy", PREPREFIX_LEN, SPLIT)

    # all_generations, all_losses = [], []

    # if experiment is not done before, generate new data
    if not all([os.listdir(generations_base), os.listdir(losses_base)]):
        for trial in range(NUM_TRIALS):
            os.makedirs(experiment_base, exist_ok=True)
            generations, losses = generate_for_prompts(prompts, BATCH_SIZE, SUFFIX_LEN, PREFIX_LEN, PREPREFIX_LEN)

            logger.info(f"Trial {trial}: Generated {len(generations)} generations.")

            generation_string = os.path.join(generations_base, "{}.npy")
            losses_string = os.path.join(losses_base, "{}.npy")

            write_array(generation_string, generations, trial)
            write_array(losses_string, losses, trial)

        #     all_generations.append(generations)
        #     all_losses.append(losses)

        # generations = np.stack(all_generations, axis=1)
        # losses = np.stack(all_losses, axis=1)
    else:
        # we do not overwrite old results
        logger.info("Experiment done before, loading previously generated data...")
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
        
    logger.info("====== Done ======")

if __name__ == "__main__":
    main()