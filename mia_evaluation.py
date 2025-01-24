import logging
import argparse
import json
import torch
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from util_lib import *
from plot_lib import *

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
parser.add_argument(
    "--eval_mode", type=str, required=False, help="Determines what will be evaluated, default (not provided) is evaluation of a single experiment, epochs evaluates the same experiment along different epochs, models compares the finding of different models for the same amount of training epochs"
)

parser.add_argument(
    "--epochs", type=int, required=False, help="If analyzing epochs, specifies the amount of epochs"
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

def write_stats(results: list, file: str, percentiles = True):
    # First write some basic stats
    mean = np.mean(results)
    median = np.median(results)
    std = np.std(results, ddof=1)
    with open(file, "a") as f:
        f.write(f"Mean: {mean}\n")
        f.write(f"Median: {median}\n")
        f.write(f"Standard deviation: {std}\n")
    
    if not percentiles:
        return
    # Write the percentiles, if that is desired.
    p25 = np.percentile(results, 25)
    p50 = np.percentile(results, 50)
    p75 = np.percentile(results, 75)
    p90 = np.percentile(results, 90)
    p99 = np.percentile(results, 99)
    with open(file, "a") as f:
        f.write(f"25, 50, 75, 90, and 99 Percentiles: {p25} {p50} {p75} {p90} {p99}\n")

# Write all sorts of statistical data
def evaluate_results(results: list, sentence_lengths: list, dir: str):
    file = os.path.join(dir, "stats.txt")
    write_stats(results, file)
    increased_perplexity_amt = 0
    for result in results:
        if result > 1: 
            increased_perplexity_amt += 1
    with open(file, "a") as f:
        f.write(f"Percentage of sentences with ratio greater than 1: {increased_perplexity_amt/len(results)}\n")
    # Gather and write stats for all results that are above the 75 percentile
    percentiles = [75,90,99]
    for percentile in percentiles:
        lengths_over_p = []
        p = np.percentile(results, percentile)
        for i in range(len(results)):
            if results[i] > p:
                lengths_over_p.append(sentence_lengths[i])
        with open(file, "a") as f:
            f.write(f"---- Stats for sentence length of upper {100-percentile} percentile in perplexity ratio ----\n")
        write_stats(lengths_over_p, file, False)

# evaluate the experiment that is specified in the EXPERIMENT_NAME
def evaluate_experiment():
    logger.info("===== Evaluating experiment %s =====", EXPERIMENT_NAME)
    data_dir = get_data_directory(DATASET_DIR, PREPROCESSING, NORMALIZATION, EXAMPLE_TOKEN_LEN)
    with open(os.path.join(data_dir, DATASET_NAME + "." + LANGUAGE), "r") as f:
        dataset = f.readlines()
    res_dir = get_mia_result_directory(ROOT_DIR, DATASET_DIR, EXPERIMENT_NAME)
    res_dir_no_epoch = get_mia_result_directory(ROOT_DIR, DATASET_DIR, EXPERIMENT_NAME, False)
    
    # generate some stats for the losses obtained
    losses_trained = torch.load(os.path.join(res_dir, "losses_trained.pt"))
    write_stats([item for batch in losses_trained for item in batch], os.path.join(res_dir, "losses_trained_stats.txt"))
    if not os.path.exists(os.path.join(res_dir_no_epoch, "losses_untrained_stats.txt")):
        losses_untrained = torch.load(os.path.join(res_dir_no_epoch, "losses_untrained.pt"))
        write_stats([item for batch in losses_untrained for item in batch], os.path.join(res_dir_no_epoch, "losses_untrained_stats.txt"))
    
    # analyze the results of the mia
    results_list_dict = torch.load(os.path.join(res_dir, "mia.pt"))
    results = convert_to_dict(results_list_dict)
    sentence_lengths = [min(len(tokenizer.encode(dataset[key])), 512) for key in results.keys()]
    torch.save(sentence_lengths, os.path.join(res_dir, "sentence_lengths.pt"))
    results_list = list(results.values())
    evaluate_results(results_list, sentence_lengths, res_dir)
    means = [np.mean(list(result.values())) for result in results_list_dict]
    medians = [np.median(list(result.values())) for result in results_list_dict]
    plot_results_scatter(results_list, sentence_lengths, means, medians, res_dir)
    
    # analyze stats for each batch individually, no plotting done for every batch
    stats_file = os.path.join(res_dir, "stats.txt")
    prev = 0
    for i, result in enumerate(results_list_dict):
        with open(stats_file, "a") as f:
            f.write(f"\n---- Stats for batch {i} ----\n")
        evaluate_results(list(result.values()), sentence_lengths[prev:prev+len(result.values())], res_dir)
        prev += len(result.values())

# Evaluate all experiments that have run for the specified amount of epochs by plotting their means and medians
# Output: One plot of all means for experiments that have run for the specified amount of epochs, and one such plot for the medians.
# They are stored in the result directory in the folder E*
def evaluate_epochs(epochs: int):
    dir = get_mia_result_directory(ROOT_DIR, DATASET_DIR, "", True)
    dir = dir[:-1]
    folder_suffix = f"-E{epochs}"
    result_dir = os.path.join(dir, f"E{epochs}")
    # Filter all directories for the given epoch. They all have a suffix folder_suffix
    folders = os.listdir(dir)
    folders = [f for f in folders if f.endswith(folder_suffix)]
    # plot the means and medians
    plot_means_medians(folders, dir, result_dir, f"{epochs} of training")

# Evaluate all experiments of a certain model, for both languages (e.g. all experiments on the 125M model with specified pretraining)
# Output: One plot of all means for experiments that have run for the specified amount of epochs, and one such plot for the medians.
# Additionally, we compare the the change in losses between different epochs of training (mia only compares between some epoch of training and untrained) to see how the increase increases with more training.
# These results are stored with names that end on _epochs_comp.png. All results are in a folder in the results directory that speicifies the experiment, e.g. (100-nat-125M)
def evaluate_model():
    languages = ["en,nl"]
    experiment = EXPERIMENT_NAME[2:-3]
    experiment_names = [lang + experiment for lang in languages]
    dir = get_mia_result_directory(ROOT_DIR, DATASET_DIR, "", True)
    dir = dir[:-1]
    folders_all = os.listdir(dir)
    folders = []
    for experiment_name in experiment_names:
        folders.extend([f for f in folders_all if f.startswith(experiment_name)])
    # plot all ratios
    plot_means_medians(folders, dir, os.path.join(dir, EXPERIMENT_NAME[2:]), f"the {experiment} experiment")

    # set up the plot
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    set_up_plot(ax[0], "Perplexity ratios comparing different epochs of training ", "Sentence length (tokenized)", "Perplexity ratio")
    # plot all ratio increases, so compare the loss from 1 epoch of training to 2 epochs, from 2 to 4, and so on
    # seperate the folders per language, then sort them 
    folders_lang = []
    for experiment_name in experiment_names:
        folders_lang.append([f for f in folders_all if f.startswith(experiment_name)])
    folders_lang = [sorted(f) for f in folders_lang]

    # for the folders for both languages do
    for folders_spec in folders_lang:
        result_0 = torch.load(os.path.join(dir, folders_spec[0], "mia.pt"))
        plotting_means_medians(ax, result_0, f"ratio of {folders_spec[i]} to untrained model", )
        for i in range(1, len(folders_spec)):
            # load two results (remember, the folders are sorted), and compute their ratio. e.g. if results are for e1 and e2, we get the ratio e1/e2
            results1 = torch.load(os.path.join(dir, folders_spec[i-1], "mia.pt"))
            results2 = torch.load(os.path.join(dir, folders_spec[i], "mia.pt"))
            result_ratio = []
            for i in range(len(results1)):
                result_ratio.append({j: results2[i][j] / results1[i][j] for j in results1[i].keys()})
            
            # Now compute and plot the means and medians
            plotting_means_medians(ax, result_ratio, f"ratio of {folders_spec[i]} / {folders_spec[i-1]}", )

    ax[0].figure.savefig(os.path.join(dir, experiment, "means_epochs_comp.png"), bbox_inches="tight")
    ax[1].figure.savefig(os.path.join(dir, experiment, "medians_epochs_comp.png"), bbox_inches="tight")

def main():
    if not args.eval_mode:
        evaluate_experiment()
    elif args.eval_mode == "epochs":
        evaluate_epochs(args.epochs)
    elif args.eval_mode == "model":
        evaluate_model()
    else:
        logger.info("===== Unknwon evaluation mode. Terminating. =====")
        return
    logger.info("===== Done! =====")

if __name__ == "__main__":
    main()