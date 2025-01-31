import os
import logging
import argparse
import json
import torch
import math
from transformers import set_seed, AutoModelForCausalLM
from scipy.stats import skewnorm, kstest
import numpy as np
import matplotlib.pyplot as plt
from util_lib import *
from experiment_lib import compute_losses_per_batch
from plot_lib import *
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up logger
logger = initLogger()

logger.info("Parsing arguments...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
parser.add_argument(
    "--eval_mode", type=str, required=False, help="Determines what will be evaluated, default (not provided) is evaluation of a single experiment, epochs evaluates the same experiment along different epochs, models compares the finding of different models for the same amount of training epochs"
)
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


# Set default device
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

logger.info(f"Default device: {DEFAULT_DEVICE}")

SAMPLE_SIZE = 1000000

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

def evaluate_experiment():
    logger.info(f"Evaluating experiment {EXPERIMENT_NAME}")
    logger.info("Loading data...")
    dir = get_canary_result_directory(ROOT_DIR, DATASET_DIR, EXPERIMENT_NAME)
    sample_losses = torch.load(os.path.join(dir, "canary-losses.pt"))
    #with open(os.path.join(dir, "stats.json")) as f:
        #stats = json.load(f)
    # TODO this code is temporary:
    with open(os.path.join(DATASET_DIR, "canary" + f"-{LANGUAGE}.json"), "r") as f:
        canary = json.load(f)
    prefix_len = len(tokenizer.encode(canary["prefix"]))
    # Tokenize the canary, then calculate its loss
    tokenized_canary = tokenizer([canary["prefix"] + " " + canary["suffix"]], max_length=512, padding=True, truncation=True, return_tensors="pt")
    logger.info("Calculating loss for canary")
    loss_canary = compute_losses_per_batch(MODEL, [tokenized_canary], DEFAULT_DEVICE, BATCH_SIZE, prefix_len)
    loss_canary = loss_canary[0][0]*(len(tokenized_canary["input_ids"][0]) - prefix_len)
    
    # Calculate the exposure of the actual canary based on the approximated distribution and the loss of the canary
    logger.info("Calculating canary exposure")
    # End of temporary code
    # Fit the data into a skew-normal distribution, and create the data for plotting
    shape, location, scale = skewnorm.fit(sample_losses)
    x = np.linspace(location - 5*scale, location + 5*scale, 500)
    pdf = skewnorm.pdf(x, shape, location, scale)
    # This logarithm computes with base e
    exposure = -1*math.log(skewnorm.cdf(loss_canary, shape, location, scale))
    # Plot the results
    logger.info("Plotting exposure")
    plt.figure(figsize=(8, 6))
    plt.plot(x, pdf, label = "PDF", color = "orange")
    plt.hist(sample_losses, bins=500, density = True, alpha = 0.6, color="blue", label = "Histogram of samples")
    plt.axvline(loss_canary, color="black", linestyle="--", label="The loss of the canary", linewidth=1)
    plt.xlabel("Log-perplexity")
    plt.ylabel("Probability density")
    plt.title(f"Canary attack {EXPERIMENT_NAME}")
    plt.legend(loc = "upper left")
    plt.grid()
    plt.savefig(os.path.join(dir, "plot.png"), dpi = 300, bbox_inches = "tight")

    # Perform Kolmogorov-Smirnov goodness of fit test
    logger.info("Performing Kolmogorov-Smirnov goodness of fit test")
    cdf = lambda x: skewnorm(x, shape, location, scale)
    _, p_ks = kstest(sample_losses, cdf)

    # Save the parameters in json format for easy usability
    logger.info("Saving results")
    stats = {}
    stats["location"] = location
    stats["scale"] = scale
    stats["shape"] = shape
    stats["loss"] = loss_canary
    stats["exposure"] = exposure
    stats["p_ks"] = p_ks
    with open(os.path.join(dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent = 4)
    
def evaluate_insertions():
    logger.info("Evaluating all canary attacks")
    dir = get_canary_result_directory(ROOT_DIR, DATASET_DIR, "").strip("/")
    experiment_names = ["en-100-nat-1.3B-can-I", "en-100-nat-125M-can-I", "en-100-nat-2.7B-can-I", "nl-100-nat-1.3B-can-I", "nl-100-nat-125M-can-I", "nl-100-nat-2.7B-can-I"]
    markers = get_colors()
    colors = get_markers()
    folders_all = sorted(os.listdir(dir))
    x = [], y_exposure = [], y_p_ks = []
    logger.info("Retrieving data")
    for i,experiment_name in enumerate(experiment_names):
        folders_experiment = [f for f in folders_all if f.startswith(experiment_name)]
        insertions = []
        exposure = []
        p_ks = []
        for folder in folders_experiment:
            name = os.path.basename(folder.strip("/"))
            insertions.append(int(name[len(experiment_name):]))
            with open(os.path.join(dir, "stats.json")) as f:
                stats = json.load(f)
            exposure.append(stats["exposure"])
            p_ks.append(stats["p_ks"])
        x[i] = insertions
        y_exposure[i] = exposure
        y_p_ks[i] = p_ks
    
    logger.info("Plotting results")
    # Plot the exposure values
    experiments_len_half = (int) (len(experiment_names / 2))
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    for i, experiment_name in enumerate(experiment_names):
        ax[0].plot(x[i], y_exposure[i], label = experiment_name, color = colors[i // experiments_len_half], marker = markers[i % experiments_len_half])
    set_up_plot(ax[0], "Exposure of canary attacks", "Number of insertions", "Exposure")
    fig.savefig(os.path.join(dir, "plot_exposures.png"))
    # Plot the goodness of fit data
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    for i, experiment_name in enumerate(experiment_names):
        ax[0].plot(x[i], y_p_ks[i], label = experiment_name, color = colors[i // experiments_len_half], marker = markers[i % experiments_len_half])
    confidence_level = 0.05
    ax[0].axvline(x = confidence_level, color = "black", linestyle = "--", label = "Confidence level")
    set_up_plot(ax[0], "P-values of Kolmogorov-Smirnov tests", "Number of insertions", "K-S p-value")
    fig.savefig(os.path.join(dir, "plot_K-S.png"))

def main():
    logger.info("===== Starting canary evaluation =====")
    if not args.eval_mode:
        evaluate_experiment()
    elif args.eval_mode == "insertions":
        evaluate_insertions()
    else:
        logger.info("Unknown evaluation mode. No evaluation")
    
    logger.info("===== Canary evaluation done =====")

if __name__ == "__main__":
    main()