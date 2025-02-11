import logging
from IPython.display import display
import os
import argparse
import json
from experiment_lib import load_constants_from_config
from torch.utils.data import random_split
from transformers import set_seed
import torch

# Configure Python's logging in Jupyter notebook
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))

# Set up logger
logger = logging.getLogger()
handler = JupyterHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
    TRAIN_FILE, 
    VAL_FILE,
    NPY_ARRAYS_BASE,
    VAL_SPLIT, 
    SEED
) = load_constants_from_config(config)

# set_seed(SEED)

# This script splits the full dataset into a training and validation set
# Performed for both langauges in the experiment to maintain a balanced version of the training and validation sets
languages = ["el", "es"]

def main():
    logger.info("==== Starting data train+val split script ====")

    # load the dataset
    data_set_base = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), DATASET_NAME)
    eval_percentage = VAL_SPLIT

    # Path where the split and validation datasets will be stored (same directory as the original dataset)
    output_dir = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN))

    # Step 1: split on indices and save them to a file
    indices_file = os.path.join(output_dir, "split_indices.json")

    # take the size of the first language as the size of the dataset
    # this can be the second one as well, the input datasets are already aligned and identical
    dataset_path = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), DATASET_NAME + f".{languages[0]}")
    # print("Dataset path:", dataset_path)
    with open(dataset_path, "r") as f:
        dataset = f.readlines()

    # Calculate the size of the training and validation sets
    train_size = int(len(dataset) * (1- eval_percentage))
    eval_size = len(dataset) - train_size

    # Create a range of indices to map to exids later
    indices = list(range(len(dataset)))

    # Split the indices along with the dataset
    logger.info("Splitting indices...")
    train_indices, eval_indices = random_split(indices, [train_size, eval_size])

    # Convert Subset objects to lists
    train_indices = train_indices.indices
    eval_indices = eval_indices.indices

    print("# of indices: ", len(train_indices)+ len(eval_indices))
    
    # Save indices to file in JSON format
    with open(indices_file, "w") as f:
        json.dump({"train": train_indices, "eval": eval_indices}, f)


    # Step 2: split the datasets using the indices and save them to files
    logger.info("Splitting datasets into train and validation sets...")
    for lang in languages:
        logger.info(f"Processing language: {lang}")
        train_out_file = os.path.join(output_dir, "train-" + lang + ".txt")
        val_out_file = os.path.join(output_dir, "validation-" + lang + ".txt")
        jsonl_out_file = os.path.join(data_set_base + f".{lang}-train.jsonl")
        indices_exist = os.path.exists(indices_file)
        text_files_exist = os.path.exists(train_out_file) and os.path.exists(val_out_file)

        # Split train and validation files only if they do not exist
        if not text_files_exist or not indices_exist:
            dataset_path = os.path.join(data_set_base + f".{lang}")
            # print("Dataset path:", dataset_path) 
            with open(dataset_path, "r") as f:
                dataset = f.readlines()
        
            # Create the train and eval datasets using the indices
            train_dataset = [dataset[i] for i in train_indices]
            eval_dataset = [dataset[i] for i in eval_indices]

            # Save train and eval datasets to files
            with open(train_out_file, "w") as f:
                f.writelines(train_dataset)

            with open(val_out_file, "w") as f:
                f.writelines(eval_dataset)
        else:
            logger.info(f"Text files for {lang} already exist. Skipping text split.")

        # Generate the JSONL file (always executed)
        if not os.path.exists(jsonl_out_file):
            logger.info(f"Creating JSONL file for training set: {jsonl_out_file}")
            dataset_path_jsonl = os.path.join(dataset_path + ".jsonl")
            # print("Dataset path:", dataset_path_jsonl)
            with open(dataset_path_jsonl, "r") as f, open(indices_file, "r") as idx_file:
                dataset_jsonl = f.readlines()
                train_indices = json.load(idx_file)["train"]
                # print("Train indices:", train_indices)
                # print("Dataset jsonl:", dataset_jsonl)

                with open(jsonl_out_file, "w") as out_file:
                    # Iterate over all train_indices to write JSONL objects
                    for index in train_indices:
                        # print("last index:", index)
                        json_obj = json.loads(dataset_jsonl[index])
                        json.dump(json_obj, out_file, ensure_ascii=False)
                        out_file.write("\n")
        else:
            logger.info(f"JSONL file for {lang} already exists. Skipping JSONL creation.")

    logger.info("==== Data train+val split script completed ====")

if __name__ == "__main__":
    main()