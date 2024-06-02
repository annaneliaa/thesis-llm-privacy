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

# 4. Plot distribution of scores

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
