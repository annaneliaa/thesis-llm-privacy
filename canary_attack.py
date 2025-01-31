import os
import logging
import argparse
import json
import torch
import random
import math
from transformers import set_seed, AutoModelForCausalLM
from scipy.stats import skewnorm
from util_lib import *
from data_lib import tokenize_prompts_in_batches
from experiment_lib import compute_losses_per_batch
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up logger
logger = initLogger()

logger.info("Parsing arguments...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
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

SAMPLE_SIZE = 1000000


def sample_canaries(prefix: str, suffix: str, prefix_len):
    logger.info("Sampling canary variants")
    digit_amount = len(suffix)
    # generate the canary candidates used for sampling
    number_range = 10**digit_amount - 1
    sample_suffixes = [f"{random.randint(0, number_range):0{digit_amount}}" for i in range(SAMPLE_SIZE)]
    sample_sentences = [prefix + " " + sample_suffix for sample_suffix in sample_suffixes]
    # Tokenize the sentences. Because the numbers might be split into a different amount of tokens, this is done in batches where every batch has uniform length (without padding/truncation)
    logger.info("Tokenizing canary variants")
    tokenized = tokenize_prompts_in_batches(tokenizer, {i: sample_sentences[i] for i in range(len(sample_sentences))}, 1)
    # Now compute the losses per batch
    logger.info("Computing losses for canary variants")
    losses_mean = compute_losses_per_batch(MODEL, tokenized, DEFAULT_DEVICE, BATCH_SIZE, prefix_len)
    losses = []
    # The previous losses are means, but we want sums. Hence, multiply by the suffix lengths, which differ per batch.
    for i, batch in enumerate(losses_mean):
        suffix_len = len(tokenized[i]["input_ids"][0]) - prefix_len
        losses.extend([suffix_len*loss for loss in batch])
    return losses

def main():
    logger.info("===== Starting canary attack =====")
    # Load the canary
    with open(os.path.join(DATASET_DIR, "canary" + f"-{LANGUAGE}.json"), "r") as f:
            canary = json.load(f)
    
    prefix_len = len(tokenizer.encode(canary["prefix"]))
    # Sample the losses of potential canaries, and fit a skew normal distribution for that sample
    sample_losses = sample_canaries(canary["prefix"], canary["suffix"], prefix_len)
    shape, location, scale = skewnorm.fit(sample_losses)
    
    # Tokenize the canary, then calculate its loss
    tokenized_canary = tokenizer([canary["prefix"] + " " + canary["suffix"]], max_length=512, padding=True, truncation=True, return_tensors="pt")
    logger.info("Calculating loss for canary")
    loss_canary = compute_losses_per_batch(MODEL, [tokenized_canary], DEFAULT_DEVICE, BATCH_SIZE, prefix_len)
    loss_canary = loss_canary[0][0]*(len(tokenized_canary["input_ids"][0]) - prefix_len)
    
    # Calculate the exposure of the actual canary based on the approximated distribution and the loss of the canary
    logger.info("Calculating canary exposure")
    # This logarithm computes with base e
    exposure = -1*math.log(skewnorm.cdf(loss_canary, shape, location, scale))
    
    # Save the results, plot the distribution
    logger.info("Saving results...")
    dir = get_canary_result_directory(ROOT_DIR, DATASET_DIR, EXPERIMENT_NAME)
    torch.save(sample_losses, os.path.join(dir, "canary-losses.pt"))
    stats = {}
    stats["location"] = location
    stats["scale"] = scale
    stats["shape"] = shape
    stats["loss"] = loss_canary
    stats["exposure"] = exposure
    with open(os.path.join(dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent = 4)
    
    logger.info("===== Canary attack done =====")

if __name__ == "__main__":
    main()