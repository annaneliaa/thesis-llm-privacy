import argparse
import logging
import shutil
from IPython.display import display
from transformers import AutoTokenizer
from data_lib import *
from util_lib import *

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up logger
logger = initLogger()

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

# For dataprocessing we use the GPT-2 tokenizer
MODEL_NAME = "gpt2"
languages = ["en", "nl"]

# Load tokenizer
tokenizer = initTokenizer(model_name=MODEL_NAME)

def main():
    # Input: Two parallel datasets where each line is a sentence, in english and dutch (or LANG1 and LANG2)
    # This script will process the data and save it in a format that can be used by the model
    # The data will be tokenized to count the number of tokens in each sentence
    # Each sentence is assigned an example ID
    # We balance the English and Dutch datasets by only keeping sentences that are at least the desired token length in both languages, and also exist in both sets
    # Output: A JSONL version of both datasets, aligned such that the set of example IDs is the same for both languages

    logger.info("==== Starting data processing script ====")
    logger.info("This may take a while depending on the size of the dataset...")
    # dataset_base has the file path of the dataset minus the ending that indicates the language
    dataset_base = os.path.join(DATASET_DIR, DATASET_NAME)
    #output_file_pattern is the directory where the output databases are stored
    output_file_pattern = get_data_directory(DATASET_DIR, PREPROCESSING, NORMALIZATION, EXAMPLE_TOKEN_LEN)
    if (PREPROCESSING):
        dataset_base = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), DATASET_NAME + PREPROCESSING_SUFFIX)
    
    # If there is no normalization, we simply create a jsonl file with the existing datasets, copy the existing dataset to the appropriate directory, and terminate
    if (NORMALIZATION == False):
        for lang in languages:
            input_file = os.path.join(dataset_base + "." + lang)
            logger.info("Generating JSONL for %s...", lang)
            text_to_jsonlines(input_file, os.path.join(output_file_pattern, DATASET_NAME + "." + lang + ".jsonl"))
            shutil.copy(input_file, os.path.join(output_file_pattern, DATASET_NAME + "." + lang))
        logger.info("==== Done: No normalization as specified in %s ====", args.config_file)
        return
    

    # this is where temporarily created csv files are stored
    csv_output_file_pattern = os.path.join(SOURCE_DIR, DATASET_DIR, "csv", str(EXAMPLE_TOKEN_LEN))
    # Count the number of tokens in each sentence for both datasets
    # Count the number of sentences that are at least the desired token length
    # Filtering csv files on the basis of token length
    # Generate JSONL version of the datasets for inspection
    for lang in languages:
        input_file = os.path.join(dataset_base + "." + lang)
        output_file = os.path.join(csv_output_file_pattern, DATASET_NAME + "." + lang + ".csv")
        
        logger.info("Counting tokens for %s...", lang)
        generate_token_count_csv(input_file, output_file, tokenizer)
        
        count = count_large_entries(output_file, EXAMPLE_TOKEN_LEN)
        logger.info("Number of samples >= %s tokens in %s: %s", str(EXAMPLE_TOKEN_LEN), output_file, count) 

        # Filtering csv files on the basis of token length
        logger.info("Filtering sentences for %s...", lang)
        output_csv = os.path.join(csv_output_file_pattern, DATASET_NAME + "-" + str(EXAMPLE_TOKEN_LEN) + "." + lang + ".csv")
        filter_csv(output_file, output_csv, EXAMPLE_TOKEN_LEN)

        logger.info("Generating JSONL for %s...", lang)
        # Assigning NEW exids starting at 1
        text_to_jsonlines(input_file, os.path.join(output_file_pattern, DATASET_NAME + "." + lang + ".jsonl"))

    # Compute common example ID
    csv_file_pattern = os.path.join(csv_output_file_pattern, DATASET_NAME + "-" + str(EXAMPLE_TOKEN_LEN) + ".")
    csv_file_lang1 = csv_file_pattern + languages[0] + ".csv"
    csv_file_lang2 = csv_file_pattern + languages[1] + ".csv"
    output_csv = os.path.join(csv_output_file_pattern, "common_exids-" + str(EXAMPLE_TOKEN_LEN) + ".csv")

    common_exids = find_common_exids(csv_file_lang1, csv_file_lang2)
    write_exids_to_file(common_exids, output_csv)
    logger.info(f"Common exids have been written to {output_csv}")

    # Filter the datasets to only include the common example IDs
    # Truncate sentences to the desired token length
    exid_list = read_common_exids(output_csv)
    logger.info("%s common example IDs found", len(exid_list))

    for lang in languages:
        input_json_file = os.path.join(output_file_pattern, DATASET_NAME  + "." + lang + ".jsonl")
        trunc_json_file = os.path.join(output_file_pattern, DATASET_NAME + "-temp." + lang + ".jsonl")
        
        # Truncate sentences and put in JSONL format for string comparison after extraction
        trunc_json(input_json_file, trunc_json_file, EXAMPLE_TOKEN_LEN, exid_list, tokenizer)
        
        # JSONL version of the complete dataset is no longer needed, so overwrite it
        os.rename(trunc_json_file, input_json_file)
        
        # Make text version of jsonl version too, for model training
        extract_text_from_json(input_json_file, os.path.join(output_file_pattern, DATASET_NAME + "." + lang))

    logger.info("==== Data processing script completed ====")

if __name__ == "__main__":
    main()
