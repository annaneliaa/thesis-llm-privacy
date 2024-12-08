import argparse
import logging
from IPython.display import display
from transformers import AutoTokenizer
from data_lib import *
from experiment_lib import load_constants_from_config

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
    VAL_SPLIT, 
    SEED
) = load_constants_from_config(config)

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
    
    if (PREPROCESSING == False):
        logger.info("==== No preprocessing as specified in %s ====", args.config_file)
        return

    logger.info("==== Starting data preprocessing script ====")
    logger.info("This may take a while depending on the size of the dataset...")

    # Load the datasets
    # e.g. dataset_base = "europarl/europarl-v7.nl-en"
    dataset_base = os.path.join(DATASET_DIR, DATASET_NAME)
    # e.g. dataset_subdir = "europarl/100/europarl-v7.nl-en"
    dataset_subdir = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), DATASET_NAME)
    # Count the number of tokens in each sentence for both datasets
    # e.g. csv_output_file_pattern = "datasets/europarl/csv/100"
    csv_output_file_pattern = os.path.join(SOURCE_DIR, DATASET_DIR, "csv", str(EXAMPLE_TOKEN_LEN))
    min_count = (INF, "")
    for lang in languages:
        input_file = os.path.join(dataset_base + "." + lang)
        # Save the complete token count in a CSV file
        output_file= os.path.join(csv_output_file_pattern, DATASET_NAME + "." + lang + ".csv")

        logger.info("Counting tokens for %s...", lang)
        generate_token_count_csv(input_file, output_file, tokenizer)
        
        count = count_large_entries(output_file, EXAMPLE_TOKEN_LEN)
        logger.info("Number of samples >= %s tokens in %s: %s", str(EXAMPLE_TOKEN_LEN), output_file, count) 
    
        if count < min_count[0]:
            min_count = (count, lang)
    
    smallest_set = min_count[1]

    # Concatenate sentences that are too short in the smallest dataset to ensure balance
    # Use the file with complete token counts as input
    in_file = os.path.join(csv_output_file_pattern, DATASET_NAME + "." + smallest_set + ".csv") 
    # Save the concatenated sentences (as lists of exids) in a JSONL file
    out_file = os.path.join(csv_output_file_pattern, DATASET_NAME + "." + smallest_set + ".jsonl")

    # Make sure out_file exists
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # First filtering out all sentences that are already long enough
    logger.info("Filtering sentences that are >= %s tokens to %s...", str(EXAMPLE_TOKEN_LEN), out_file)
    filter_large_entries(in_file, out_file, EXAMPLE_TOKEN_LEN)

    # Concatenating step
    # Output file is in JSONL format
    # Create new sentence pairs and record their new sizes
    # Groups default is 12
    logger.info("Concatening remaining sentences to %s", out_file)
    group_sentences(in_file, out_file, groups=12)


    # Count the number of (concatenated) samples >= EXAMPLE_TOKEN_LEN in the concatenated dataset for the "smallest" language
    new_sample_count = count_large_entries_json(out_file, EXAMPLE_TOKEN_LEN)
    logger.info("Concatenated sentences in %s to reach %s samples >= %s tokens", smallest_set, new_sample_count, str(EXAMPLE_TOKEN_LEN))
    
    # Create concatenated version for the datasets
    # txt format, aligned on sentences
    # Save new formatted datasets in a sub directory indicating the EXAMPLE_TOKEN_LEN value
    logger.info("Generating concatenated .txt version of original dataset for both languages...")
    for lang in languages:
        reformat_dataset(out_file, 
                         os.path.join(dataset_base + "." + lang),
                         os.path.join(dataset_subdir + PREPROCESSING_SUFFIX + "." + lang)
        )

    logger.info("==== Data preprocessing complete ====")
    
if __name__ == "__main__":
    main()