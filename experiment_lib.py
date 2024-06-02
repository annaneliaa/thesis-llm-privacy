import os
import json
import numpy as np
import matplotlib.pyplot as plt

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0

# Function to generate a jsonlines version of dataset
# input here is a numpy array of tokenized data (using token IDs)
def generations_to_jsonl(output_file_path: str, data: np.ndarray, tokenizer):
    """Converts the tokenized data to a JSONL file at `path`."""

    with open(output_file_path, "w", encoding="utf-8", newline='') as file:
        exid = 0
        for row in data:
            # Convert token IDs to strings
            # replace token space character with empty strin
            decoded_string = tokenizer.decode(row, skip_special_tokens=True).replace('Ġ', '')
            line = decoded_string.strip()

            # Skip empty lines
            if not line:
                continue

            # Create a JSON object with a "text" field containing the line
            json_object = {"exid": id,
                           "text": line}

            # Write the JSON object to the output file as a single line
            json.dump(json_object, file, ensure_ascii=False)
            file.write("\n")
            exid += 1

    print("Decoded strings saved to: %s", str(output_file_path))

# Function to merge bleu scores over different trials of one example sentence with exid curr_exid
# Using binary search to speed up the search when dealing with large datasets
# returns a list of dicts with keys "trial" and "score"
def merge_bleu_scores(directory, num_trials, curr_exid):
    scores = []
    for i in range(num_trials):
        trial_file = directory + f"bleu_scores_trial_{i}.jsonl"
        if not os.path.exists(trial_file):
            continue  # Skip if trial file doesn't exist
        with open(trial_file, 'r') as f:
            lines = f.readlines()
            scores_found = False
            low = 0
            high = len(lines) - 1
            while low <= high:
                mid = (low + high) // 2
                obj = json.loads(lines[mid])
                if obj["exid"] == curr_exid:
                    score_obj = {"trial": i, "score": obj["score"]}
                    scores.append(score_obj)
                    scores_found = True
                    break
                elif obj["exid"] < curr_exid:
                    low = mid + 1
                else:
                    high = mid - 1
            if scores_found:
                # example found, move on to next file
                continue
    return scores

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
