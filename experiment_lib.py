import os
import json
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import torch
from torch.utils.data import random_split
from transformers import AutoModelForCausalLM

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
# and a list of exids from the original (training) dataset
def generations_to_jsonl(output_file_path: str, data: np.ndarray, tokenizer, exids):
    """Converts the tokenized data to a JSONL file at `path`."""

    with open(output_file_path, "w", encoding="utf-8", newline='') as file:
        index = 0
        
        for row in data:
            exid = exids[index]
            # Convert token IDs to strings
            # replace token space character with empty string
            decoded_string = tokenizer.decode(row, skip_special_tokens=True).replace('Ġ', '')
            line = decoded_string.strip()

            # Create a JSON object with a "text" field containing the line
            json_object = {"exid": exid,
                           "text": line}

            # Write the JSON object to the output file as a single line
            json.dump(json_object, file, ensure_ascii=False)
            file.write("\n")
            index += 1

    print("Decoded strings saved to:", str(output_file_path))

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
def sort_scores(scores):
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

def calculate_perplexity(losses: dict):
    # Parse the JSON object
    # The object is of the form { "losses": [{"trial": 0, "loss": 0.0}]}
    # Initialize an empty list for perplexities
    perplexities = []

    for trial in losses:
        # Calculate the perplexity as exp(loss)
        perplexity = np.exp(trial['loss'])
        # Create a new dictionary with the trial number and the calculated perplexity
        perplexities.append({'trial': trial['trial'], 'perplexity': perplexity})
    
    return perplexities

# Calculates the likelihood for a list of losses, but only for non-padding tokens (as indicated by the attention masks)
# Input: The list of lists of losses and their corresponding attention masks
# Output: The list of likelihoods (1 per list of losses)
def calculate_likelihoods(loss_per_token_2d, attention_masks_2d):
    likelihoods = []
    # filter out the losses of padding tokens by applyting the attention_masks. Then calculate the mean of the losses
    for i,sentence_logits in enumerate(loss_per_token_2d):
        sentence_mask = attention_masks_2d[i].bool()
        non_padded_losses = sentence_logits[sentence_mask]
        likelihoods.append(torch.mean(non_padded_losses).item())
    return likelihoods

# Input: Takes in a list of prompt batches with uniform size, where every batch in the list has a field "attention_mask" and a 
# field "input_ids", which are lists of tokenized sentences/their attention masks.
# Returns a list of prompt losses per batch (shape: (batch_amt, batch_prompt_amt))
def compute_losses_per_batch(model: AutoModelForCausalLM, prompts_list: list, default_device: str, batch_size: int, suffix_len = -1) -> list:
    losses = []
    for i, prompts in enumerate(prompts_list):
        print(f"Computing losses for batch {i}")
        # will temporarily hold the losses for this batch of prompts
        batch_losses = []
        # seperate attention masks and input ids. They are both 2d tensors.
        attention_masks = prompts["attention_mask"]
        input_ids = prompts["input_ids"]

        generation_len = len(input_ids[0])
        if suffix_len == -1:
            suffix_idx = 0
        else:
            suffix_idx = generation_len - suffix_len

        for j, off in enumerate(range(0, len(input_ids), batch_size)):
            print(f"{j}/{(int)(len(input_ids)/batch_size)}")
            # Get the data for the current batch, and realign it
            prompt_batch = input_ids[off:off+batch_size]
            input_ids_batch = torch.tensor(prompt_batch, dtype=torch.int64).to(default_device)
            attention_masks_batch = attention_masks[off:off+batch_size]

            with torch.no_grad():
                # Pass through the model to obtain the logits
                outputs = model(input_ids_batch, labels=input_ids_batch)
                # Store the logits (shape: (batch_size, sequence_length, vocab_size), sequence length is the length of each prompt)
                logits = outputs.logits.cpu().detach()
                # reshape logits into shape (batch_size * (sequence_length-1), vocab_size)
                logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
                # calculate the loss per token by taking the cross_entropy, returned shape is (batch_size*(sequence_length-1))
                loss_per_token = torch.nn.functional.cross_entropy(
                    logits, input_ids_batch[:, 1:].to('cpu').detach().flatten(), reduction="none"
                ).cpu()
                # Reshape to get an array of shape (batch_size, sequence_length-1) (so every row represents one prompt)
                # Then calculate the likelihood for each row (sentence), and append the resulting array to batch_losses
            batch_losses.extend(calculate_likelihoods(loss_per_token.reshape((-1, generation_len - 1))[:, suffix_idx:], attention_masks_batch[:, 1:]))
            # this is to not run out of gpu memory
            del outputs, logits, input_ids_batch
            torch.cuda.empty_cache()
        # concatenate all the loss scores for this batch of prompts of equal length, and append it to the list of losses per prompt batch
        losses.append(batch_losses)
    return losses