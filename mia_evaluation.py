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

def evaluate_results(results: list, sentence_lengths: list, dir: str):
    # Make a scatter plot of the results based on the sentence_lengths
    plt.figure(figsize=(8, 6))
    plt.scatter(sentence_lengths, results, c='blue', s=10, alpha=0.7)
    plt.xlabel("Setence length (tokenized)")
    plt.ylabel("Loss ratio")
    plt.title(f"Membership inference attack {EXPERIMENT_NAME}")
    plt.grid()
    plt.savefig(os.path.join(dir, "plot.png"), dpi = 300, bbox_inches = "tight")
    # Write all sorts of statistical data
    file = os.path.join(dir, "stats.txt")
    write_stats(results, file)
    increased_perplexity_amt = 0
    for result in results:
        if result > 1: 
            increased_perplexity_amt += 1
    with open(file, "a") as f:
        f.write(f"Percentage of sentences with ratio greater than 1: {increased_perplexity_amt/len(results)}\n")
    # Make a plot with only the lower 75 percentile, and with the upper 25 percentile
    p75 = np.percentile(results, 75)
    plt.ylim(bottom=p75)
    plt.title(f"Membership inference attack {EXPERIMENT_NAME}: Results in the highest 25 percentile")
    plt.savefig(os.path.join(dir, "plot_over_p75.png"), dpi = 300, bbox_inches = "tight")
    plt.ylim(bottom = 0, top=p75)
    plt.title(f"Membership inference attack {EXPERIMENT_NAME}: Results in the lower 75 percentile")
    plt.savefig(os.path.join(dir, "plot_under_p75.png"), dpi = 300, bbox_inches = "tight")
    # Gather and write stats for all results that are above the 75 percentile
    lengths_over_p75 = []
    for i in range(len(results)):
        if results[i] > p75:
            lengths_over_p75.append(sentence_lengths[i])
    write_stats(lengths_over_p75, os.path.join(dir, "stats_lengths_over_p75.txt"))


def main():
    logger.info("===== Evaluating experiment %s =====", EXPERIMENT_NAME)
    data_dir = get_data_directory(DATASET_DIR, PREPROCESSING, NORMALIZATION, EXAMPLE_TOKEN_LEN)
    with open(os.path.join(data_dir, DATASET_NAME + "." + LANGUAGE), "r") as f:
        dataset = f.readlines()
    res_dir = get_mia_result_directory(ROOT_DIR, DATASET_DIR, EXPERIMENT_NAME)
    # generate some stats for the losses obtained
    losses_trained = torch.load(os.path.join(res_dir, "losses_trained.pt"))
    write_stats(losses_trained, os.path.join(res_dir, "losses_trained_stats.text"))
    losses_untrained = torch.load(os.path.join(res_dir, "losses_untrained.pt"))
    write_stats(losses_untrained, os.path.join(res_dir, "losses_untrained_stats.text"))
    # analyze the results of the mia
    results = torch.load(os.path.join(res_dir, "mia.pt"))
    results = convert_to_dict(results)
    sentence_lengths = [len(tokenizer.encode(dataset[key])) for key in results.keys()]
    evaluate_results(results.values(), sentence_lengths, res_dir)
    logger.info("===== Done! =====")

if __name__ == "__main__":
    main()