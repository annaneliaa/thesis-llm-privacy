import os
import logging
import argparse
import json
import torch
import random
import math
from transformers import set_seed, AutoModelForCausalLM
from scipy.stats import skewnorm
import numpy as np
import matplotlib.pyplot as plt
from util_lib import *
from experiment_lib import compute_losses_per_batch
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up logger
logger = initLogger()

logger.info("Parsing arguments...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
parser.add_argument("--insertions", type=int, required=False, help="The number of times the canary is inserted")
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

set_seed(SEED)

# Set default device
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

logger.info(f"Default device: {DEFAULT_DEVICE}")

try:
    logger.info("Loading trained model...")
    MODEL = AutoModelForCausalLM.from_pretrained(get_model_directory(DATASET_DIR, EXPERIMENT_NAME), low_cpu_mem_usage=True, cache_dir=cache_dir)
    # move model to GPU
    MODEL.to(DEFAULT_DEVICE)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models or tokenizer: {e}")
    raise

tokenizer = initTokenizer(MODEL_NAME)
pad_token_id = tokenizer.pad_token_id

SAMPLE_SIZE = 100000


def sample_canaries(prefix: str, suffix: str):
    logger.info("Sampling canary variants")
    digit_amount = len(suffix)
    number_range = 10**digit_amount - 1
    sample_suffixes = [f"{random.randint(0, number_range):0{digit_amount}}" for i in range(SAMPLE_SIZE)]
    sample_sentences = [prefix + " " + sample_suffix for sample_suffix in sample_suffixes]
    tokenized = tokenizer(sample_sentences, max_length=512, padding=True, truncation=True, return_tensors="pt")
    logger.info("Computing losses for canary variants")
    losses = compute_losses_per_batch(MODEL, [tokenized], DEFAULT_DEVICE, BATCH_SIZE, digit_amount)[0]
    # TODO potentially add further processing steps
    return losses

def main():
    logger.info("===== Starting canary attack =====")
    with open(os.path.join(DATASET_DIR, "canary", f"-{LANGUAGE}.json"), "r") as f:
            canary = json.load(f)
    digit_amount = len(canary["suffix"])
    losses = sample_canaries(canary["prefix"], canary["suffix"])
    shape, location, scale = skewnorm.fit(losses)
    tokenized_canary = tokenizer([canary["prefix"] + " " + canary["suffix"]], max_length=512, padding=True, truncation=True, return_tensors="pt")
    logger.info("Calculating loss for canary")
    loss_canary = compute_losses_per_batch(MODEL, [tokenized_canary], DEFAULT_DEVICE, BATCH_SIZE, digit_amount)[0][0]
    # This logarithm computes with base e
    logger.info("Calculating canary exposure")
    exposure = -math.log(skewnorm.ppf(loss_canary, shape, location, scale))
    dir = get_canary_result_directory(ROOT_DIR, DATASET_DIR, EXPERIMENT_NAME)
    
    logger.info("Saving results...")
    torch.save(loss_canary, os.path.join(dir, "canary-losses.pt"))
    with open(os.path.join(dir, "exposure.txt"), "w") as f:
        f.write(f"The exposure is {exposure}")
    # TODO add other statistic here
    # plot the pdf of the approximation, plot the sampling
    x = np.linspace(location - 5*scale, location + 5*scale, 500)
    pdf = skewnorm.pdf(x,shape,location,scale)
    plt.figure(figsize=(8, 6))
    plt.plot(x, pdf, label = "PDF", color = "orange")
    plt.hist(loss_canary, bins=500, density = True, alpha = 0.6, color="blue", label = "Histogram of samples")
    plt.axvline(loss_canary, color="red", linestyle="--", label="The loss of the canary", linewidth=1)
    plt.xlabel("Exposure")
    plt.ylabel("Probability density")
    plt.title(f"Canary attack {EXPERIMENT_NAME}")
    plt.legend(loc = "upper left")
    plt.grid()
    plt.savefig(os.path.join(dir, "plot.png"), dpi = 300, bbox_inches = "tight")
    logger.info("===== Canary attack done =====")

if __name__ == "__main__":
    main()