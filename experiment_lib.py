import os
import json
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import torch
from torch.utils.data import random_split

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0

def generate_exid_list(file_path):
    exids = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                exids.append(line.strip())
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
    return exids

# Function to generate a jsonlines version of model output
# input here is a numpy array of tokenized data (using token IDs)
def generations_to_jsonl(output_file_path: str, data: np.ndarray, tokenizer, exids_file_path):
    """Converts the tokenized data to a JSONL file at `path`."""

    if os.path.exists(exids_file_path) and os.path.getsize(exids_file_path) > 0:
        exids = generate_exid_list(exids_file_path)
    else:
        print(f"File {exids_file_path} does not exist or is empty.")

    with open(output_file_path, "w", encoding="utf-8", newline='') as file:
        index = 0
        
        for row in data:
            exid = exids[index]
            # Convert token IDs to strings
            # replace token space character with empty strin
            decoded_string = tokenizer.decode(row, skip_special_tokens=True).replace('Ġ', '')
            line = decoded_string.strip()

            # Create a JSON object with a "text" field containing the line
            json_object = {"exid": exid,
                           "text": line}

            # Write the JSON object to the output file as a single line
            json.dump(json_object, file, ensure_ascii=False)
            file.write("\n")
            index += 1

    print("Decoded strings saved to: %s", str(output_file_path))

# Function to generate a jsonlines version of scores for each example, for each trial
def losses_to_jsonl(output_file_path: str, data: np.ndarray, exids):
    """Converts tokenized losses to a JSONL file at `path`."""
    index = 0

    with open(output_file_path, "w", encoding="utf-8", newline='') as file:
        # loop over all rows in the trial
        for row in data:     
            # get the exid of the example from list       
            exid = int(exids[index])

            # scores are ordered
            # convert to native python float
            score = row[0].item()
    
            # Create a JSON object with a "text" field containing the line
            json_object = {"exid": exid,
                           "loss": score}

            # Write the JSON object to the output file as a single line
            json.dump(json_object, file, ensure_ascii=False)
            file.write("\n")
            index += 1

    print("Decoded losses saved to: %s", str(output_file_path))

# Function to merge bleu scores over different trials of one example sentence with exid curr_exid
# Using binary search to speed up the search when dealing with large datasets
# returns a list of dicts with keys "trial" and "score"

def merge_scores_or_losses(directory, trial_file_pattern, num_trials, curr_exid, logger, is_loss):
    # to store scores or losses
    scores = []
    # loop over all trials
    for i in range(num_trials):
        # get the file path
        trial_file = os.path.join(directory, trial_file_pattern + f"{i}.jsonl")
        if not os.path.exists(trial_file):
            logger.warning(f"File {trial_file} not found.")
            continue  # Skip if trial file doesn't exist
        with open(trial_file, 'r') as f:
            # use binary search to find exid in file
            lines = f.readlines()
            scores_found = False
            low = 0
            high = len(lines) - 1
            logger.debug(f"Searching for exid {curr_exid} in {trial_file} (lines {low}-{high})")
            while low <= high:
                mid = (low + high) // 2
                try:
                    obj = json.loads(lines[mid])
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON at line {mid} in {trial_file}: {e}")
                    break
                if int(obj["exid"]) == curr_exid:
                    if is_loss:
                        score_obj = {"trial": i, "loss": obj["loss"]}
                    else:
                        score_obj = {"trial": i, "score": obj["score"]}
                    scores.append(score_obj)
                    scores_found = True
                    break
                elif int(obj["exid"]) < curr_exid:
                    low = mid + 1
                else:
                    high = mid - 1
            if scores_found:
                logger.debug(f"Found exid {curr_exid} in trial {i}")
                # example found, move on to next file
                continue
    return scores

def sort_losses(losses):
    return sorted(losses, key=lambda x: x["loss"], reverse=True)

# The function sort_bleu_scores(scores) expects a list of dicts
# each dictionary has a key called "score"
# sorts this list of dicts based on the value of "score" in DESCENDING order
def sort_bleu_scores(scores):
    return sorted(scores, key=lambda x: x["score"], reverse=True)

def read_bleu_scores(file_path):
    scores = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            scores.append(data['score'])
    return scores

def plot_bleu_distribution(root_dir, experiment_name, scores, trial, num_trials, num_bins=10):
    plt.figure(figsize=(10, 6))
    
    # Compute histogram
    counts, bins = np.histogram(scores, bins=num_bins, range=(0, 1))
    
    # Plot histogram as bar chart
    plt.bar(bins[:-1], counts, width=(bins[1] - bins[0]), edgecolor='black', align='edge')

    # add a grid on image
    plt.grid(True)

    # Set titles and labels
    plt.title('Distribution of BLEU Scores')
    plt.xlabel('BLEU Score')
    plt.ylabel('Frequency')
    
    plt.xticks(np.linspace(0, 1, num_bins + 1))

    # create a directory
    plots_dir = os.path.join(root_dir, experiment_name, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # save the file
    plt.savefig(os.path.join(plots_dir, f"bleu_distribution_trial_{trial}.png"))
    
    # Show plot
    plt.show()

# get the shape of a numpy array
def get_shape(arr):
    if isinstance(arr, np.ndarray):
        return arr.shape
    return None

def load_constants_from_config(config):
    # For saving results
    ROOT_DIR = config["root_dir"]
    # Name of the dataset
    DATASET_DIR = config["dataset_dir"]
    # Directory where the .npy files of the dataset are stored
    SOURCE_DIR = config["source_dir"]
    # Name of the dataset
    DATASET_NAME = config["dataset_name"]
    # Name of the experiment
    EXPERIMENT_NAME = config["experiment_name"]
    # Number of trials
    NUM_TRIALS = config["num_trials"]
    # Language of the scenario (EN/NL)
    LANGUAGE = config["language"]
    # Split the dataset into train and eval
    SPLIT = config["split"]
    # Length of the suffix
    SUFFIX_LEN = config["suffix_len"]
    # Length of the prefix
    PREFIX_LEN = config["prefix_len"]
    # Number of tokens in the complete sequences
    EXAMPLE_TOKEN_LEN = config["example_token_len"]
    # Preprefix length
    PREPREFIX_LEN = config["preprefix_len"]
    # Name of the tokenized .npy file of the dataset
    SOURCE_FILE = config["source_file"]
    # Batch size for feeding prompts to the model
    BATCH_SIZE = config["batch_size"]
    # Name of the model to use
    MODEL_NAME = config["model"]
    TRAIN_FILE = config["train_file"]
    VAL_FILE = config["validation_file"]
    VAL_SPLIT = config["validation_split_percentage"]
    SEED = config["seed"]

    return (ROOT_DIR, DATASET_DIR, SOURCE_DIR, DATASET_NAME, EXPERIMENT_NAME, NUM_TRIALS, PREFIX_LEN, SUFFIX_LEN, PREPREFIX_LEN, LANGUAGE, SPLIT, EXAMPLE_TOKEN_LEN, SOURCE_FILE, BATCH_SIZE, MODEL_NAME, TRAIN_FILE, VAL_FILE, VAL_SPLIT, SEED)



def text_to_csv(dir, train_file, val_file):
    with open(train_file, encoding='utf-8') as txtfile:
        all_text = txtfile.read()
    with open(os.path.join(dir, 'train.csv'), mode='w', encoding='utf-8') as csv_file:
        fieldnames = ['text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'text': all_text})


    with open(val_file, encoding='utf-8') as txtfile:
        all_text = txtfile.read()
    with open(os.path.join(dir, 'validation.csv'), mode='w', encoding='utf-8') as csv_file:
        fieldnames = ['text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'text': all_text})

def split_set_to_train_val(eval_percentage, output_dir, dataset_path, language):
    train_out_file = os.path.join(output_dir, "train-" + language + ".txt")
    val_out_file = os.path.join(output_dir, "evaluation-" + language + ".txt")
    indices_file = os.path.join(output_dir, "split_indices-" + language + ".json")

    # Check if the files already exist
    if os.path.exists(train_out_file) and os.path.exists(val_out_file) and os.path.exists(indices_file):
        print("Files already exist. Skipping computation.")
        return

    # Load the dataset
    with open(dataset_path, "r") as f:
        dataset = f.readlines()

    train_size = int(len(dataset) * (1- eval_percentage))
    eval_size = len(dataset) - train_size

    # Create a range of indices to map to exids later
    indices = list(range(len(dataset)))

    # Split the indices along with the dataset
    train_indices, eval_indices = random_split(indices, [train_size, eval_size])

    # Create the train and eval datasets using the indices
    train_dataset = [dataset[i] for i in train_indices]
    eval_dataset = [dataset[i] for i in eval_indices]

    # Save train and eval datasets to files
    with open(train_out_file, "w") as f:
        f.writelines(train_dataset)

    with open(val_out_file, "w") as f:
        f.writelines(eval_dataset)

    # Convert Subset objects to lists
    train_indices = train_indices.indices
    eval_indices = eval_indices.indices

    # Save indices to file in JSON format
    with open(indices_file, "w") as f:
        json.dump({"train": train_indices, "eval": eval_indices}, f)
