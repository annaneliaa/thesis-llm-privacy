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
from experiment_lib import *
from data_lib import tokenize_prompts_in_batches

# Configure Python's logging in Jupyter notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up logger
logger = initLogger()

logger.info("Parsing arguments...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory with the saved model")
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
    cache_dir = "/scratch/s5202841"

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
    # move model to GPU
    MODEL_UNTRAINED.to(DEFAULT_DEVICE)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models or tokenizer: {e}")
    raise

logger.info("Experiment name: %s", EXPERIMENT_NAME)
logger.info("Language: %s", LANGUAGE)
logger.info("Model: %s", MODEL_NAME)

# Calculates the likelihood for a list of losses, but only for non-padding tokens (as indicated by the attention masks)
# Input: The list of lists of losses and their corresponding attention masks
# Output: The list of likelihoods (1 per list of losses)
def calculate_likelihoods(loss_per_token_2d, attention_masks_2d):
    likelihoods = []
    # filter out the losses of padding tokens by applyting the attention_masks. Then calculate the mean of the losses
    for i,sentence_logits in enumerate(loss_per_token_2d):
        sentence_mask = attention_masks_2d[i].bool()
        non_padded_losses = sentence_logits[sentence_mask]
        likelihoods.append(torch.mean(non_padded_losses))
    return likelihoods

# Input: Takes in a list of prompt batches with uniform size, where every batch in the list has a field "attention_mask" and a 
# field "input_ids", which are lists of tokenized sentences/their attention masks.
# Returns a list of prompt losses per batch (shape: (batch_amt, batch_prompt_amt))
def compute_losses_per_batch(model: AutoModelForCausalLM, prompts_list: list, batch_size: int):
    losses = []
    for prompts in prompts_list:
        # will temporarily hold the losses for this batch of prompts
        batch_losses = []
        # seperate attention masks and input ids. They are both 2d tensors.
        attention_masks = prompts["attention_mask"]
        input_ids = prompts["input_ids"]

        generation_len = len(input_ids[0])

        for i, off in enumerate(range(0, len(input_ids), batch_size)):
            # Get the data for the current batch, and realign it
            prompt_batch = input_ids[off:off+batch_size]
            attention_masks_batch = attention_masks[off:off+batch_size]
            # TODO: Prompt batch would be a tensor, can we call np.stack on a tensor (same for attention masks)?
            prompt_batch = np.stack(prompt_batch, axis=0)
            attention_masks_batch = np.stack(attention_masks_batch, axis=0)
            input_ids_batch = torch.tensor(prompt_batch, dtype=torch.int64).to(DEFAULT_DEVICE)

            with torch.no_grad():
                # Pass through the model to obtain the logits
                outputs = model(input_ids_batch.to(DEFAULT_DEVICE), labels=input_ids_batch.to(DEFAULT_DEVICE))
                # Store the logits (shape: (batch_size, sequence_length, vocab_size), sequence length is the length of each prompt)
                logits = outputs.logits.cpu().detach()
                # reshape logits into shape (batch_size * (sequence_length-1), vocab_size)
                logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
                # calculate the loss per token by taking the cross_entropy, returned shape is (batch_size*(sequence_length-1))
                loss_per_token = torch.nn.functional.cross_entropy(
                    logits, input_ids_batch[:, 1:].flatten(), reduction="none"
                )
                # Reshape to get an array of shape (batch_size, sequence_length-1) (so every row represents one prompt)
                # Then calculate the likelihood for each row (sentence), and append the resulting array to batch_losses
                batch_losses.extend(calculate_likelihoods(loss_per_token.reshape((-1, generation_len - 1))))
        # concatenate all the loss scores for this batch of prompts of equal length, and append it to the list of losses per prompt batch
        losses.append(batch_losses)
    return losses


# Executes a membership inference attack with the passed prompts by comparing the perplexity of two models, in this case a trained (MODEL_NAME, specified in flag),
# and untrained instance (HGModel, specified in config)
# Output: A list of numpy dictionaries, where every dictionary corresponds to one batch in the input data, and contains the ratio of perplexity
# between extraction on the trained and untrained model for every sentence in the batch, mapped to the sentence ids they correspond to.
def mia_comp(prompts, batch_size: int):
    # Compute the losses for the 
    losses_trained = compute_losses_per_batch(MODEL, prompts, batch_size)
    losses_untrained = compute_losses_per_batch(MODEL_UNTRAINED, prompts, batch_size)
    # Make the losses in every batch a numpy array to calculate the perplexity
    losses_trained_npy = [loss.numpy() for loss in losses_trained]
    losses_untrained_npy = [loss.numpy() for loss in losses_untrained]
    # Both trained and untrained have the same amount of batches, and the same amount of losses in every batch. 
    # Hence, we can simply calculate the ratio of their perplexity
    perplexity_ratio = [np.exp(losses_trained_npy[i] - losses_untrained_npy[i]) for i in range(len(losses_trained_npy))]
    # remap the perplexity scores to the sentence ids
    dict_ratio = [
        {prompts[batch_nr]["sentence_ids"][i]: perplexity_ratio[batch_nr][i] for i in range(len(perplexity_ratio[batch_nr]))}
          for batch_nr in range(len(perplexity_ratio))
        ]
    return dict_ratio
    
    
def main():
    logger.info("====== Starting membership inference attack ======")
    # Get and create directories
    experiment_base = os.path.join(ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME)
    source_dir = get_source_directory(SOURCE_DIR, DATASET_DIR, LANGUAGE, PREPROCESSING, NORMALIZATION, EXAMPLE_TOKEN_LEN)
    os.makedirs(experiment_base, exist_ok=True)
    # Get the prompts
    prompts = torch.load(os.path.join(source_dir, "train-" + LANGUAGE + ".pt"))
    if not BATCHING:
        prompts = [prompts]
    # Do the membership inference attack and save the results
    mia_results = mia_comp(prompts, BATCH_SIZE)
    if not BATCHING:
        mia_results = mia_results[0]
    torch.save(mia_results, os.path.join(experiment_base, "mia.pt"))    

if __name__ == "__main__":
    main()