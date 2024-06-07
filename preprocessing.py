import argparse
import logging
from IPython.display import display
from transformers import AutoTokenizer
from data_lib import *

INF = float("inf")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))

# Set up logger
logger = logging.getLogger()
handler = JupyterHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info("Parsing arguments...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
args = parser.parse_args()

with open(args.config_file, 'r') as f:
    config = json.load(f)

# Directory of the dataset
DATASET_DIR = config["dataset_dir"]
# Name of the dataset files
DATASET_NAME = config["dataset_name"]
# Directory where the .npy files of the dataset are stored
SOURCE_DIR = config["source_dir"]
# Number of tokens in the complete sequences
EXAMPLE_TOKEN_LEN = config["example_token_len"]

# For dataprocessing we use the GPT-2 tokenizer
MODEL_NAME = "gpt2"
languages = ["en", "nl"]

# Load tokenizer
logger.info("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    logger.error(f"Error loading tokenizer: {e}")
    raise e

def main():
    # Input: Two parallel datasets where each line is a sentence, in english and dutch (or LANG1 and LANG2)
    # Assumption: the datasets are aligned, and will be used to train a model
    # To increase the number of long sentences in the dataset, we concatenate sentences < desired token length
    # Sentences that are long enough will not be concatenated
    # Concatenating of sentences will be performed on the smallest dataset of the two to ensure identical sentence pairs
    # Output: A JSONL version of both datasets, aligned such that the set of example IDs is the same for both languages
    
    logger.info("==== Starting data preprocessing script ====")
    logger.info("This may take a while depending on the size of the dataset...")

    # Load the datasets
    dataset_base = os.path.join(config["dataset_dir"], config["dataset_name"])

    # Count the number of tokens in each sentence for both datasets
    csv_output_file_pattern = os.path.join(SOURCE_DIR, DATASET_DIR, "csv")
    min_count = (INF, "")
    for lang in languages:
        input_file = os.path.join(dataset_base + "." + lang)
        output_file= os.path.join(csv_output_file_pattern, DATASET_NAME + "." + lang + ".csv")

        logger.info("Counting tokens for %s...", lang)
        generate_token_count_csv(input_file, output_file, tokenizer)
        
        count = count_large_entries(output_file, EXAMPLE_TOKEN_LEN)
        logger.info("Number of samples >= %s tokens in %s: %s", EXAMPLE_TOKEN_LEN, output_file, count) 
    
        if count < min_count[0]:
            min_count = (count, lang)
    
    smallest_set = min_count[1]

    # Concatenate sentences that are too short in the smallest dataset to ensure balance
    in_file = os.path.join(csv_output_file_pattern, DATASET_NAME + "." + smallest_set + ".csv") 
    out_file = os.path.join(SOURCE_DIR, DATASET_DIR, DATASET_NAME + "." + smallest_set + ".jsonl")

    # Output file is in JSONL format
    # Create new sentence pairs and record their new sizes
    process_train_data(in_file, out_file,EXAMPLE_TOKEN_LEN)

    new_sample_count = count_large_entries_json(out_file, EXAMPLE_TOKEN_LEN)
    logger.info("Concatenated sentences in %s to reach %s samples >= %s tokens", smallest_set, new_sample_count, EXAMPLE_TOKEN_LEN)
    
    # Create concatenated version for the datasets
    # JSONL format, aligned on example IDs
    logger.info("Generating JSONL for both languages...")
    for lang in languages:
        reformat_dataset(out_file, 
                         os.path.join(dataset_base + "." + lang),
                         os.path.join(dataset_base + "-c." + lang)
        )

    logger.info("==== Data preprocessing complete ====")
    
if __name__ == "__main__":
    main()