import logging
from IPython.display import display
import os
from typing import Tuple, Union
import numpy as np
import torch
import json
import argparse
from transformers import AutoModelForCausalLM
from util_lib import *
from experiment_lib import compute_losses_per_batch

# Configure Python's logging in Jupyter notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up logger
logger = initLogger()

logger.info("Parsing arguments...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
parser.add_argument("--cache_dir", type=str, required=False, help="Path to the cache directory")

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

HGmodel = MODEL_NAME

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
    cache_dir = get_cache_directory()

MODEL_NAME = get_model_directory(DATASET_DIR, EXPERIMENT_NAME)

# Load models and tokenizer
tokenizer = initTokenizer(HGmodel)
try:
    logger.info("Loading trained model...")
    MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True, cache_dir=cache_dir)
    # move model to GPU
    MODEL.to(DEFAULT_DEVICE)
    logger.info("Model loaded successfully.")
    logger.info("Loading untrained model...")
    MODEL_UNTRAINED = AutoModelForCausalLM.from_pretrained(HGmodel, low_cpu_mem_usage=True, cache_dir=cache_dir)
    # The model has to account for the padding token which was introduced, for the trained model this was done in the trainer
    MODEL_UNTRAINED.resize_token_embeddings(len(tokenizer))
    MODEL_UNTRAINED.config.pad_token_id = tokenizer.pad_token_id
    # move model to GPU
    MODEL_UNTRAINED.to(DEFAULT_DEVICE)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models or tokenizer: {e}")
    raise

logger.info("Experiment name: %s", EXPERIMENT_NAME)
logger.info("Language: %s", LANGUAGE)
logger.info("Model: %s", MODEL_NAME)

# Executes a membership inference attack with the passed prompts by comparing the perplexity of two models, in this case a trained (MODEL_NAME, specified in flag),
# and untrained instance (HGModel, specified in config)
# Output: A list of numpy dictionaries, where every dictionary corresponds to one batch in the input data, and contains the ratio of perplexity
# between extraction on the trained and untrained model for every sentence in the batch, mapped to the sentence ids they correspond to.
def mia_comp(prompts: list, batch_size: int):
    # Compute the losses for the trained model
    logger.info("Computing losses for trained model.")
    losses_trained = compute_losses_per_batch(MODEL, prompts, DEFAULT_DEVICE, batch_size)
    losses_trained_npy = [np.array(losses) for losses in losses_trained]
    # This is a small optimization: If the losses have been calculated for another experiment, then we simply load them.
    # The assumption here is that the first experiment is run with one epoch of training, s.t. the experiment name does not have the appendix -ex where x is the number of epochs of training.
    path_untrained = os.path.join(get_mia_result_directory(ROOT_DIR, DATASET_DIR, EXPERIMENT_NAME)[:-3], "losses_untrained.pt")
    if not os.path.exists(path_untrained):
        logger.info("Computing losses for untrained model.")
        losses_untrained = compute_losses_per_batch(MODEL_UNTRAINED, prompts, DEFAULT_DEVICE, batch_size)
        losses_untrained_npy = [np.array(losses) for losses in losses_untrained]
    else:
        logger.info("Losses for untrained model already calculated. Loading losses...")
        losses_untrained_npy = torch.load(path_untrained)
    logger.info("Computing ratio of losses.")
    # Both trained and untrained have the same amount of batches, and the same amount of losses in every batch. 
    # Hence, we can simply calculate the ratio of their perplexity
    perplexity_ratio = [np.exp(losses_untrained_npy[i] - losses_trained_npy[i]) for i in range(len(losses_trained_npy))]
    # remap the perplexity scores to the sentence ids
    # in the case that the data is somehow misconfigured, we save the perplexity ratio
    dict_ratio = []
    for batch_nr in range(len(perplexity_ratio)):
        try:
            dict_ratio.append({
                prompts[batch_nr]["sentence_ids"][i]: perplexity_ratio[batch_nr][i] for i in range(len(perplexity_ratio[batch_nr]))
                })
        except IndexError as e:
            print("Indexing error in batch {batch_nr}! This indicates that something has gone wrong in the ordering of data.")
            print(f"len(sentence_ids): {len(prompts[batch_nr]['sentence_ids'])}, len(perplexity_ratio[{batch_nr}]): {len(perplexity_ratio[batch_nr])}")
            print("Saving the perplexity ratio")
            torch.save(perplexity_ratio, os.path.join(get_mia_result_directory(ROOT_DIR, DATASET_DIR, EXPERIMENT_NAME), "-ratio.pt"))
            raise e
    return dict_ratio, losses_trained_npy, losses_untrained_npy
    
    
def main():
    logger.info("====== Starting membership inference attack ======")
    # Get and create directories
    result_dir = get_mia_result_directory(ROOT_DIR, DATASET_DIR, EXPERIMENT_NAME)
    source_dir = get_source_directory(SOURCE_DIR, DATASET_DIR, LANGUAGE, PREPROCESSING, NORMALIZATION, EXAMPLE_TOKEN_LEN)
    os.makedirs(result_dir, exist_ok=True)
    # Get the prompts
    if BATCHING:
        prompts = torch.load(os.path.join(source_dir, "train-" + LANGUAGE + ".pt"))
    else:
        prompts = [torch.load(os.path.join(source_dir, "train-nb-" + LANGUAGE + ".pt"))]
    # Do the membership inference attack and save the results
    mia_results, losses_trained, losses_untrained = mia_comp(prompts, BATCH_SIZE)
    logger.info("Saving results...")
    # Save the losses for potential analysis later on
    torch.save(losses_trained, os.path.join(result_dir, "losses_trained.pt"))
    torch.save(losses_untrained, os.path.join(result_dir, "losses_untrained.pt"))
    if BATCHING:
        torch.save(mia_results, os.path.join(result_dir, "mia.pt"))    
    else:
        mia_results = mia_results[0]
        torch.save(mia_results, os.path.join(result_dir, "mia-nb.pt"))  
    logger.info("====== Membership inference attack done! ======")

if __name__ == "__main__":
    main()