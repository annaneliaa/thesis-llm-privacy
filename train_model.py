import logging
from IPython.display import display
import os
import json
import argparse
from transformers import AutoTokenizer
from experiment_lib import *
from happytransformer import HappyGeneration, GENTrainArgs, GENSettings
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
DATASET_NAME = config["dataset_name"]
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

# Load model and tokenizer
try:
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    logger.info("Loading model...")
    MODEL = HappyGeneration("GPT-NEO", model)
    # move model to GPU
    MODEL.to(DEFAULT_DEVICE)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

# logger.info("Experiment name: %s", config["experiment_name"])
logger.info("Language: %s", config["language"])
logger.info("Model: %s", config["model"])


# # modified function for happy transformer trained model
# def generate_for_prompts(
#     prompts: np.ndarray, batch_size: int, suffix_len: int, prefix_len: int
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Generates suffixes given `prompts` and scores using their likelihood."""
#     generations = []
#     losses = []
#     generation_len = suffix_len + prefix_len
#     args = GENSettings(
#         max_length=generation_len, do_sample=True, top_k=10, top_p=1, pad_token_id=50256
#     )

#     for i, off in enumerate(range(0, len(prompts), batch_size)):
#         prompt_batch = prompts[off : off + batch_size]
#         logger.info(f"Generating for batch ID {i:05} of size {len(prompt_batch):04}")
#         prompt_batch = np.stack(prompt_batch, axis=0)
#         input_ids = torch.tensor(prompt_batch, dtype=torch.int64).to(DEFAULT_DEVICE)
#         input_ids.to(DEFAULT_DEVICE),
#         with torch.no_grad():
#             generated_tokens = MODEL.generate(input_ids, args=args).to("cpu").detach()
#             outputs = MODEL(
#                 generated_tokens.to(DEFAULT_DEVICE),
#                 labels=generated_tokens.to(DEFAULT_DEVICE),
#             )
#             logits = outputs.logits.cpu().detach()
#             logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
#             loss_per_token = torch.nn.functional.cross_entropy(
#                 logits, generated_tokens[:, 1:].flatten(), reduction="none"
#             )
#             loss_per_token = loss_per_token.reshape((-1, generation_len - 1))[
#                 :, -suffix_len - 1 : -1
#             ]
#             likelihood = loss_per_token.mean(1)
#             generations.extend(generated_tokens.numpy())
#             losses.extend(likelihood.numpy())
#     return np.atleast_2d(generations), np.atleast_2d(losses).reshape(
#         (len(generations), -1)
#     )

def main():
    # default epochs is 3
    args = GENTrainArgs(num_train_epochs=1)

    training_set = 
    # train the model
    logger.info("Training model on dataset %s", DATASET_NAME)
    try:
        MODEL.train(DATASET_NAME, args=args)
        logger.info("Model trained successfully.")
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

    output_dir = os.path.join("models", DATASET_DIR, LANGUAGE, EXPERIMENT_NAME)
    # save the model
    logger.info("Saving model...")
    MODEL.save(output_dir)

if __name__ == "__main__":
    main()