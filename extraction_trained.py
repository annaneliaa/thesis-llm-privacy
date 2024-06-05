import logging
from IPython.display import display
import os
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from experiment_lib import *
from happytransformer import HappyGeneration, GENSettings
from typing import Tuple, Union
import torch
import numpy as np

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

logger.info("Parsing arguments...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument(
    "--config_file", type=str, required=True, help="Path to the configuration file"
)
args = parser.parse_args()

with open(args.config_file, "r") as f:
    config = json.load(f)

# Set default device
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

logger.info(f"Default device: {DEFAULT_DEVICE}")

# For saving results
ROOT_DIR = config["root_dir"]
# Name of the dataset
DATASET_DIR = config["dataset_dir"]
# Directory where the .npy files of the dataset are stored
SOURCE_DIR = config["source_dir"]
# Name of the experiment
EXPERIMENT_NAME = config["experiment_name"]
# Name of the dataset
DATASET_FILE = config["dataset_file"]
# Number of trials
NUM_TRIALS = config["num_trials"]
# Length of the prefix
PREFIX_LEN = config["prefix_len"]
# Length of the suffix
SUFFIX_LEN = config["suffix_len"]
# Preprefix length
PREPREFIX_LEN = config["preprefix_len"]
# Language of the scenario (EN/NL)
LANGUAGE = config["language"]
# Number of tokens in the complete sequences
EXAMPLE_TOKEN_LEN = config["example_token_len"]
# Batch size for feeding prompts to the model
BATCH_SIZE = config["batch_size"]
# Name of the model to use
model = config["model"]


model_dir = os.path.join("models", DATASET_DIR, LANGUAGE, EXPERIMENT_NAME)

#NOTE: Can this be optimized? doesnt seem ideal
# Load model and tokenizer
try:
    # Load the pretrained model for HappyTransformers
    happy_gen = HappyGeneration(model_type="GPT-NEO", model_name=model_dir)
    # Load the base model for calculating the loss
    MODEL = AutoModelForCausalLM.from_pretrained(model)
    logger.info("Model loaded successfully.")
    tokenizer = AutoTokenizer.from_pretrained(model)
    logger.info("Tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

logger.info("Experiment name: %s", config["experiment_name"])
logger.info("Language: %s", config["language"])
logger.info("Model: %s", config["model"])

# modified function for happy transformer trained model
def generate_for_prompts(
    prompts: np.ndarray, batch_size: int, suffix_len: int, prefix_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates suffixes given `prompts` and scores using their likelihood."""
    generations = []
    losses = []
    generation_len = suffix_len + prefix_len
    args = GENSettings(
        max_length=generation_len, do_sample=True, top_k=10, top_p=1
    )

    for i, off in enumerate(range(0, len(prompts), batch_size)):

        prompt_batch = prompts[off : off + batch_size]
        logger.info(f"Generating for batch ID {i:05} of size {len(prompt_batch):04}")
        prompt_batch = np.stack(prompt_batch, axis=0)
        input_ids = torch.tensor(prompt_batch, dtype=torch.int64).to(DEFAULT_DEVICE)

        with torch.no_grad():
            result = happy_gen.generate_text(input_ids, args=args).to("cpu").detach()
            
            # Convert generated text to tokens
            generated_tokens = tokenizer.encode(result.text, return_tensors="pt").to(DEFAULT_DEVICE)

            # Evaluate output with Pytorch model
            outputs = MODEL(
                generated_tokens.to(DEFAULT_DEVICE),
                labels=generated_tokens.to(DEFAULT_DEVICE),
            )
            logits = outputs.logits.cpu().detach()
            logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
            loss_per_token = torch.nn.functional.cross_entropy(
                logits, generated_tokens[:, 1:].flatten(), reduction="none"
            )
            loss_per_token = loss_per_token.reshape((-1, generation_len - 1))[
                :, -suffix_len - 1 : -1
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

def main(): 
    logger.info("======= Starting extraction ======")

    logger.info("Number of trials: %d", NUM_TRIALS)

    logger.info("Creating paths...")
    experiment_base = os.path.join(ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME)
    generations_base = os.path.join(experiment_base, "generations")
    os.makedirs(generations_base, exist_ok=True)
    losses_base = os.path.join(experiment_base, "losses")
    os.makedirs(losses_base, exist_ok=True)
    prompts_base = os.path.join(SOURCE_DIR, DATASET_DIR, LANGUAGE, str(EXAMPLE_TOKEN_LEN), model)

    logger.info("Loading prompts from numpy file")
    prompts = load_prompts(prompts_base, "train_prefix.npy")

    all_generations, all_losses = [], []

    # if experiment is not done before, generate new data
    if not all([os.listdir(generations_base), os.listdir(losses_base)]):
        for trial in range(NUM_TRIALS):
            os.makedirs(experiment_base, exist_ok=True)
            generations, losses = generate_for_prompts(prompts, BATCH_SIZE, SUFFIX_LEN, PREFIX_LEN)

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

    logger.info("Decoding model generations to JSONL...")
    # Path to exids of the dataset
    exids = os.path.join(SOURCE_DIR, DATASET_DIR, "csv", "common_exids-"+str(EXAMPLE_TOKEN_LEN)+".csv")
    for i in range(0, NUM_TRIALS):
        file_path = os.path.join(experiment_base, f"generations/{i}.npy")
        data = np.load(file_path)
        logger.info("Data shape: %s", str(data.shape))

        output_file_path = os.path.join(experiment_base, f"decoded/decoded_strings_trial_{i}.jsonl")
        output_dir = os.path.dirname(output_file_path)
        os.makedirs(output_dir, exist_ok=True)
        generations_to_jsonl(output_file_path, data, tokenizer, exids)

    logger.info("====== Done ======")

if __name__ == "__main__":
    main()
