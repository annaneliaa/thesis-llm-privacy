import argparse
import json
from IPython.display import display
from transformers import AutoTokenizer
from experiment_lib import *
import wandb
import logging

# Configure Python's logging in Jupyter notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s")

class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))

def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    # For saving results
    ROOT_DIR = config["root_dir"]
    # Name of the dataset
    DATASET_DIR = config["dataset_dir"]
    # Directory where the .npy files of the dataset are stored
    SOURCE_DIR = config["source_dir"]
    # Name of the experiment
    EXPERIMENT_NAME = config["experiment_name"]
    # Number of trials
    NUM_TRIALS = config["num_trials"]
    # Length of the prefix
    PREFIX_LEN = config["prefix_len"]
    # Length of the suffix
    SUFFIX_LEN = config["suffix_len"]
    # Preprefix length
    PREPREFIX_LEN = config["preprefix_len"]
    # Language of the scenario (EN/NL)
    LANGUAGE = config["language"]
    # Number of tokens in the complete sequences
    EXAMPLE_TOKEN_LEN = config["example_token_len"]
    # Name of the model to use
    model = config["model"]

    tokenizer = AutoTokenizer.from_pretrained(model)

    # Initialize wandb
    wandb.init(
        project="thesis-llm-privacy",
        name="Evaluation BLEU Score - " + EXPERIMENT_NAME + " - " + model,
        config={
            'experiment_name': EXPERIMENT_NAME,
            "dataset": DATASET_DIR,
            "language": LANGUAGE,
            "token_len": EXAMPLE_TOKEN_LEN,
            "prefix_len": PREFIX_LEN,
            "num_trials": NUM_TRIALS
        })
        
    # Set up logger
    logger = logging.getLogger()
    handler = JupyterHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.inf("==== Starting evaluation ====")

    # Merge bleu scores over different trials of all examples
    logger.info("Loading list of example IDs for dataset %s...", DATASET_DIR)
    dataset_base = os.path.join(SOURCE_DIR, DATASET_DIR, "csv", "common_exids-" + EXAMPLE_TOKEN_LEN + ".csv")
    exids = []
    with open(dataset_base, 'r') as f:
        for line in f:
            exids.append(line.strip())
    f.close()
        
    # Create output file
    scores_base = os.path.join(ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME, "scores")
    os.makedirs(os.path.dirname(scores_base), exist_ok=True)
    output_file = scores_base + "complete_bleu_scores.jsonl"    

    # the exids list is sorted
    for exid in exids:
        logger.info("Processing example %s...", exid)
        # get all scores for this example
        scores = merge_bleu_scores(ROOT_DIR, NUM_TRIALS, exid)

        json_object = {"exid": exid, "scores": scores}

        # Write the JSON object to the output file as a single line
        with open(output_file, 'a') as file:
            json.dump(json_object, file, ensure_ascii=False)
            file.write("\n")
        file.close()
        logger.info("Merged BLEU scores for exid %s", exid)    

    # Sort the bleu scores of all examples
    logger.info("Sorting BLEU scores...")
    sorted_output_file = scores_base + "sorted_compl_bleu_scores.jsonl"
    with open(output_file, 'r') as f, open(sorted_output_file, 'w') as file:
            lines = f.readlines()
            for line in lines:
                obj = json.loads(line)
                sorted_scores = sort_bleu_scores(obj["scores"])
                # replace the scores with the sorted list
                sorted_obj = {"exid": obj["exid"], "scores": sorted_scores}
                json.dump(sorted_obj, file, ensure_ascii=False)
                file.write("\n")
    f.close()
    file.close()
    logger.info("Sorted BLEU scores saved to %s", sorted_output_file)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input from config file.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config_file)
