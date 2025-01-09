import logging
import argparse
import json
import torch
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from util_lib import *

# Configure Python's logging in Jupyter notebook
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set up logger
logger = initLogger()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument(
    "--config_file", type=str, required=True, help="Path to the configuration file"
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

tokenizer = initTokenizer(MODEL_NAME)
pad_token_id = tokenizer.pad_token_id

def convert_to_dict(dict_list: list):
    res = {}
    for d in dict_list:
        res.update(d)
    return res

def write_stats(results: list, file: str):
    mean = np.mean(results)
    median = np.median(results)
    std = np.std(results, ddof=1)
    p25 = np.percentile(results, 25)
    p50 = np.percentile(results, 50)
    p75 = np.percentile(results, 75)
    
    with open(file, "a") as f:
        f.write(f"Mean: {mean}\n")
        f.write(f"Median: {median}\n")
        f.write(f"Standard deviation: {std}\n")
        f.write(f"25, 50 and 75 Percentiles: {p25} {p50} {p75}\n")

def evaluate_results(results: list, sentence_lengths: list, dir: str, plotting = True):
    # Make a scatter plot of the results based on the sentence_lengths
    # Write all sorts of statistical data
    file = os.path.join(dir, "stats.txt")
    write_stats(results, file)
    increased_perplexity_amt = 0
    for result in results:
        if result > 1: 
            increased_perplexity_amt += 1
    with open(file, "a") as f:
        f.write(f"Percentage of sentences with ratio greater than 1: {increased_perplexity_amt/len(results)}\n")
    # Gather and write stats for all results that are above the 75 percentile
    percentiles = [75,90]
    for percentile in percentiles:
        lengths_over_p = []
        p = np.percentile(results, percentile)
        for i in range(len(results)):
            if results[i] > p:
                lengths_over_p.append(sentence_lengths[i])
        with open(file, "a") as f:
            f.write(f"---- Stats for upper {100-percentile} percentile ----\n")
        write_stats(lengths_over_p, file)

    # If no plots are desired, simply return
    if not plotting:
        return
    # Otherwise, make a scatter plot mapping the loss ratio to the sentence lengths
    plt.figure(figsize=(8, 6))
    plt.scatter(sentence_lengths, results, c='blue', s=10, alpha=0.7)
    plt.xlabel("Sentence length (tokenized)")
    plt.ylabel("Loss ratio")
    plt.title(f"Membership inference attack {EXPERIMENT_NAME}")
    plt.grid()
    plt.savefig(os.path.join(dir, "plot.png"), dpi = 300, bbox_inches = "tight")
    # Make a plot with only the lower 90 percentile, and with the upper 10 percentile
    _, top_init = plt.ylim
    for percentile in percentiles:
        p = np.percentile(results, percentile)
        plt.ylim(bottom=p, top = top_init)
        plt.title(f"Membership inference attack {EXPERIMENT_NAME}: Results in the highest {100-percentile} percentile")
        plt.savefig(os.path.join(dir, f"plot_over_p{percentile}.png"), dpi = 300, bbox_inches = "tight")
        plt.ylim(bottom = 0, top=p)
        plt.title(f"Membership inference attack {EXPERIMENT_NAME}: Results in the lower {percentile} percentile")
        plt.savefig(os.path.join(dir, f"plot_under_p{percentile}.png"), dpi = 300, bbox_inches = "tight")


def main():
    logger.info("===== Evaluating experiment %s =====", EXPERIMENT_NAME)
    data_dir = get_data_directory(DATASET_DIR, PREPROCESSING, NORMALIZATION, EXAMPLE_TOKEN_LEN)
    with open(os.path.join(data_dir, DATASET_NAME + "." + LANGUAGE), "r") as f:
        dataset = f.readlines()
    res_dir = get_mia_result_directory(ROOT_DIR, DATASET_DIR, EXPERIMENT_NAME)
    # generate some stats for the losses obtained
    losses_trained = torch.load(os.path.join(res_dir, "losses_trained.pt"))
    write_stats([item for batch in losses_trained for item in batch], os.path.join(res_dir, "losses_trained_stats.txt"))
    losses_untrained = torch.load(os.path.join(res_dir, "losses_untrained.pt"))
    write_stats([item for batch in losses_untrained for item in batch], os.path.join(res_dir, "losses_untrained_stats.txt"))
    # analyze the results of the mia
    results_list = torch.load(os.path.join(res_dir, "mia.pt"))
    results = convert_to_dict(results_list)
    sentence_lengths = [min(len(tokenizer.encode(dataset[key])), 512) for key in results.keys()]
    evaluate_results(list(results.values()), sentence_lengths, res_dir)
    # analyze stats for each batch individually, no plotting done for every batch
    stats_file = os.path.join(res_dir, "stats.txt")
    prev = 0
    for i, result in enumerate(results_list):
        with open(stats_file, "a") as f:
            f.write(f"\n---- Stats for batch {i} ----\n")
        evaluate_results(list(result.values()), sentence_lengths[prev:prev+len(result.values())], res_dir, False)
        prev += len(result.values())
    logger.info("===== Done! =====")

if __name__ == "__main__":
    main()