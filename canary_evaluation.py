import os
import logging
import argparse
import json
import torch
import math
from transformers import set_seed, AutoModelForCausalLM
from scipy.stats import skewnorm, kstest, chisquare
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
    "--eval_mode", type=str, required=False, help="Determines what will be evaluated, default (not provided) is evaluation of a single experiment, insertions evaluates the all experiments in one plot"
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

SAMPLE_SIZE = 1000000

# Evaluates a single experiment by plotting the sampling, its approximation, and the canary loss.
# Additionally, performs both a Kolmogorov-Smirnov and a chi-square goodness of fit test
def evaluate_experiment():
    logger.info(f"Evaluating experiment {EXPERIMENT_NAME}")
    logger.info("Loading data...")
    dir = get_canary_result_directory(ROOT_DIR, DATASET_DIR, EXPERIMENT_NAME)
    # load the data
    sample_losses = torch.load(os.path.join(dir, "canary-losses.pt"))
    with open(os.path.join(dir, "stats.json")) as f:
        stats = json.load(f)
    loss_canary = stats["loss"]
    shape = stats["shape"]
    location = stats["location"]
    scale = stats["scale"]
    
    x = np.linspace(location - 5*scale, location + 5*scale, 500)
    pdf = skewnorm.pdf(x, shape, location, scale)
    # Plot the results
    logger.info("Plotting exposure")
    plt.figure(figsize=(8, 6))
    plt.plot(x, pdf, label = "PDF", color = "orange")
    bin_amt = (int) (math.sqrt(SAMPLE_SIZE))
    plt.hist(sample_losses, bins=bin_amt, density = True, alpha = 0.6, color="blue", label = "Histogram of samples")
    plt.axvline(loss_canary, color="black", linestyle="--", label="The loss of the canary", linewidth=1)
    plt.xlabel("Log-perplexity")
    plt.ylabel("Probability density")
    plt.title(f"Canary attack {EXPERIMENT_NAME}")
    plt.legend(loc = "upper left")
    plt.grid()
    plt.savefig(os.path.join(dir, "plot.png"), dpi = 300, bbox_inches = "tight")

    # Perform Kolmogorov-Smirnov goodness of fit test
    logger.info("Performing Kolmogorov-Smirnov goodness of fit test")
    params = (shape, location, scale)
    cdf = lambda x: skewnorm.cdf(x, *params)
    _, p_ks = kstest(sample_losses, cdf)

    # Perform chi-square goodness of fit test
    logger.info("Performing chi-square goodness of fit test")
    observed, bin_edges = np.histogram(sample_losses, bins=bin_amt)
    expected = np.zeros_like(observed, dtype=float)
    for i in range(bin_amt):
        low = skewnorm.cdf(bin_edges[i], shape, loc=location, scale=scale)
        high = skewnorm.cdf(bin_edges[i+1], shape, loc=location, scale=scale)
        expected[i] = (high-low) * len(sample_losses)
    # Normalize expected counts to match the sum of observed counts
    expected = expected * (observed.sum() / expected.sum())
    _, p_chi = chisquare(f_obs=observed, f_exp=expected)

    # Save the parameters in json format for easy usability
    logger.info("Saving results")
    stats["p_ks"] = p_ks
    stats["p_chi"] = p_chi
    with open(os.path.join(dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent = 4)

# Evaluates all experiments
def evaluate_insertions():
    logger.info("Evaluating all canary attacks")
    # Get all the folders
    dir = get_canary_result_directory(ROOT_DIR, DATASET_DIR, "")
    experiment_names = ["en-100-nat-1.3B-can-I", "en-100-nat-125M-can-I", "en-100-nat-2.7B-can-I", "nl-100-nat-1.3B-can-I", "nl-100-nat-125M-can-I", "nl-100-nat-2.7B-can-I"]
    folders_all = sorted(os.listdir(dir))
    x, y_exposure, y_loss, y_p_ks, y_p_chi = [], [], [], [], []
    logger.info("Retrieving data")
    # Read all data necessary: The number of insertions, exposure, loss, and test statistics
    for i,experiment_name in enumerate(experiment_names):
        # Get all folders corresponding to the experiment
        folders_experiment = [f for f in folders_all if f.startswith(experiment_name)]
        insertions, exposure, loss, p_ks, p_chi = [], [], [], [], []
        # For every folder, retrieve the data
        for folder in folders_experiment:
            name = os.path.basename(folder.strip("/"))
            insertions.append(int(name[len(experiment_name):]))
            with open(os.path.join(dir, folder, "stats.json")) as f:
                stats = json.load(f)
            exposure.append(stats["exposure"])
            loss.append(stats["loss"])
            p_ks.append(stats["p_ks"])
            p_chi.append(stats["p_chi"])
        # Sort the entries based on the number of insertions
        sorted_x = np.argsort(insertions)
        x.append(np.array(insertions)[sorted_x])
        y_exposure.append(np.array(exposure)[sorted_x])
        y_loss.append(np.array(loss)[sorted_x])
        y_p_ks.append(np.array(p_ks)[sorted_x])
        y_p_chi.append(np.array(p_chi)[sorted_x])
    
    logger.info("Plotting results")
    colors = get_colors()
    markers = get_markers()
    # Plot the exposure and loss values
    experiments_len_half = (int) (len(experiment_names) / 2)
    fig, ax = plt.subplots(1,2,figsize=(16,6))
    plt.subplots_adjust(right=0.75)
    for i, experiment_name in enumerate(experiment_names):
        ax[0].plot(x[i], y_exposure[i], label = experiment_name[:-6], color = colors[i // experiments_len_half], marker = markers[i % experiments_len_half])
        ax[1].plot(x[i], y_loss[i], label = experiment_name[:-6], color = colors[i // experiments_len_half], marker = markers[i % experiments_len_half])
    set_up_plot(ax[0], "Exposure of canary attacks", "Number of insertions", "Exposure")
    set_up_plot(ax[1], "Loss of canaries", "Number of insertions", "Loss")
    ax[0].legend(loc="upper left", bbox_to_anchor=(1, 0))
    ax[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(os.path.join(dir, "plot_exposures_losses.png"))
    # Plot the goodness of fit data
    fig, ax = plt.subplots(1,2,figsize=(16,6))
    for i, experiment_name in enumerate(experiment_names):
        ax[0].plot(x[i], y_p_ks[i], label = experiment_name[:-6], color = colors[i // experiments_len_half], marker = markers[i % experiments_len_half])
        ax[1].plot(x[i], y_p_chi[i], label = experiment_name[:-6], color = colors[i // experiments_len_half], marker = markers[i % experiments_len_half])
    confidence_level = 0.05
    ax[0].axhline(y = confidence_level, color = "black", linestyle = "--", label = "Confidence level")
    ax[1].axhline(y = confidence_level, color = "black", linestyle = "--", label = "Confidence level")
    set_up_plot(ax[0], "P-values of Kolmogorov-Smirnov tests", "Number of insertions", "K-S p-value")
    set_up_plot(ax[1], "P-values of Chi-square tests", "Number of insertions", "Chi^2 p-value")
    fig.savefig(os.path.join(dir, "plot_tests.png"))

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