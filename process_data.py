import argparse
import logging
from IPython.display import display
from transformers import AutoTokenizer
from data_lib import *

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
    # This script will process the data and save it in a format that can be used by the model
    # The data will be tokenized to count the number of tokens in each sentence
    # Each sentence is assigned an example ID
    # We balance the English and Dutch datasets by only keeping sentences that are at least the desired token length in both languages
    # Output: A JSONL version of both datasets, aligned such that the set of example IDs is the same for both languages

    logger.info("==== Sarting data processing script ====")
    logger.info("This may take a while depending on the size of the dataset...")
    # Load the datasets
    dataset_base = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), DATASET_NAME)

    # Count the number of tokens in each sentence for both datasets
    # Count the number of sentences that are at least the desired token length
    # Filtering csv files on the basis of token length
    # Generate JSONL version of the datasets for inspection
    csv_output_file_pattern = os.path.join(SOURCE_DIR, DATASET_DIR, "csv", str(EXAMPLE_TOKEN_LEN))
    for lang in languages:
        input_file = os.path.join(dataset_base + "." + lang)
        output_file= os.path.join(csv_output_file_pattern, DATASET_NAME + "." + lang + ".csv")
        
        logger.info("Counting tokens for %s...", lang)
        generate_token_count_csv(input_file, output_file, tokenizer)
        
        count = count_large_entries(output_file, EXAMPLE_TOKEN_LEN)
        logger.info("Number of samples >= %s tokens in %s: %s", str(EXAMPLE_TOKEN_LEN), output_file, count) 

        # Filtering csv files on the basis of token length
        logger.info("Filtering sentences for %s...", lang)
        output_csv = os.path.join(SOURCE_DIR, DATASET_DIR, "csv", str(EXAMPLE_TOKEN_LEN), DATASET_NAME + "-" + str(EXAMPLE_TOKEN_LEN) + "." + lang + ".csv")
        filter_csv(output_file, output_csv, EXAMPLE_TOKEN_LEN)

        logger.info("Generating JSONL for %s...", lang)
        text_to_jsonlines(input_file, os.path.join(input_file + ".jsonl"))

    # Compute common example IDs
    csv_file_pattern = os.path.join(SOURCE_DIR, DATASET_DIR, "csv", str(EXAMPLE_TOKEN_LEN), DATASET_NAME + "-" + str(EXAMPLE_TOKEN_LEN) + ".")
    csv_file_lang1 = csv_file_pattern + languages[0] + ".csv"
    csv_file_lang2 = csv_file_pattern + languages[1] + ".csv"
    output_csv = os.path.join(SOURCE_DIR, DATASET_DIR, "csv", str(EXAMPLE_TOKEN_LEN), "common_exids-" + str(EXAMPLE_TOKEN_LEN) + ".csv")

    common_exids = find_common_exids(csv_file_lang1, csv_file_lang2)
    write_exids_to_file(common_exids, output_csv)
    logger.info(f"Common exids have been written to {output_csv}")

    # Filter the datasets to only include the common example IDs
    # Truncate sentences to the desired token length
    exid_list = read_common_exids(output_csv)
    logger.info("%s common example IDs found", len(exid_list))

    for lang in languages:
        input_file = os.path.join(dataset_base + "." + lang)
        trunc_json_file = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), DATASET_NAME + "-" + str(EXAMPLE_TOKEN_LEN) + "." + lang + ".jsonl")
        
        # Truncate sentences and put in JSONL format for string comparison after extraction
        trunc_json(os.path.join(input_file + ".jsonl"), trunc_json_file, EXAMPLE_TOKEN_LEN, exid_list, tokenizer)
        
        # JSONL version of the complete dataset is no longer needed
        os.remove(input_file + ".jsonl")
        
        #Make text version of jsonl version too, for model training
        extract_text_from_json(trunc_json_file, os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), DATASET_NAME + "-" + str(EXAMPLE_TOKEN_LEN) + "." + lang))

    logger.info("==== Data processing script completed ====")

if __name__ == "__main__":
    main()
