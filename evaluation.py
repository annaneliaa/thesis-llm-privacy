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

    # Set up logger
    logger = logging.getLogger()
    handler = JupyterHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("==== Starting evaluation ====")

    # Merge bleu scores over different trials of all examples
    logger.info("Loading list of example IDs for dataset %s...", DATASET_DIR)
    dataset_base = os.path.join(SOURCE_DIR, DATASET_DIR, "csv", "common_exids-" + str(EXAMPLE_TOKEN_LEN) + ".csv")
    exids = []
    with open(dataset_base, 'r') as f:
        for line in f:
            exids.append(line.strip())
    f.close()

    logger.info("Loaded %s example IDs", len(exids))
        
    # Create output file
    scores_base = os.path.join(ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME, "scores")
    os.makedirs(os.path.dirname(scores_base), exist_ok=True)
    output_file = os.path.join(scores_base, "complete_bleu_scores.jsonl")
    
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        trial_file_pattern = "bleu_scores_trial_"
        # the exids list is sorted
        for i in range(len(exids)):
            exid = exids[i]
            logger.info("Processing example %s...", exid)
            # get all scores for this example
            scores = merge_scores_or_losses(scores_base, trial_file_pattern, NUM_TRIALS, int(exid), logger, is_loss=False)

            json_object = {"exid": exid, "scores": scores}

            # Write the JSON object to the output file as a single line
            with open(output_file, 'a') as file:
                json.dump(json_object, file, ensure_ascii=False)
                file.write("\n")
            logger.info("Merged BLEU scores for exid %s", exid)   

        logger.info("All merged BLEU scores saved to %s", output_file)
    else:
        logger.info("Bleu scores for this experiment previously merged, skipping...")


    # Sort the bleu scores of all examples
    logger.info("Sorting BLEU scores...")
    sorted_output_file = os.path.join(scores_base, "sorted_compl_bleu_scores.jsonl")

    if os.path.exists(sorted_output_file) and os.path.getsize(sorted_output_file) > 0:
        logger.info("Output file %s already exists and is not empty, skipping...", sorted_output_file)
    else:
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
    
    # Decoding losses
    logger.info("Decoding losses...")
    losses_base = os.path.join(ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME, "losses")

    for i in range(NUM_TRIALS):
        decoded_losses_file = os.path.join(losses_base, f"decoded/decoded_losses_trial_{i}.jsonl")
        
        # If the file already exists, skip this iteration
        if os.path.exists(decoded_losses_file):
            logger.info("Decoded losses for trial %s already computed, skipping...", i)
            continue

        np_losses_file = os.path.join(losses_base, f"{i}.npy")
        data = np.load(np_losses_file)        
        output_dir = os.path.dirname(decoded_losses_file)
        os.makedirs(output_dir, exist_ok=True)
        losses_to_jsonl(decoded_losses_file, data, exids)

        logger.info("Decoded losses saved to %s", decoded_losses_file)

    # merge losses
    loss_output_file = os.path.join(losses_base, "decoded/complete_losses.jsonl")
    trial_file_pattern = "decoded/decoded_losses_trial_"

    # If the file already exists and is not empty, skip the rest of the code
    if os.path.exists(loss_output_file) and os.path.getsize(loss_output_file) > 0:
        logger.info("Output file %s already exists and is not empty, skipping...", loss_output_file)
    else:
        for exid in exids:
            logger.info("Processing example %s...", exid)
            # get all losses for this example over all trials
            losses = merge_scores_or_losses(losses_base, trial_file_pattern, NUM_TRIALS, int(exid), logger, is_loss=True)

            json_object = {"exid": exid, "losses": losses}

            # Write the JSON object to the output file as a single line
            with open(loss_output_file, 'a') as file:
                json.dump(json_object, file, ensure_ascii=False)
                file.write("\n")
            logger.info("Merged losses for exid %s", exid)
        
        logger.info("All merged losses saved to %s", loss_output_file)

    # Sort the losses of all examples
    logger.info("Sorting losses...")
    sorted_loss_output_file = os.path.join(losses_base, "decoded/sorted_compl_losses.jsonl")
    with open(loss_output_file, 'r') as f, open(sorted_loss_output_file, 'w') as file:
            lines = f.readlines()
            for line in lines:
                obj = json.loads(line)
                sorted_losses = sort_losses(obj["losses"])
                # replace the losses with the sorted list
                sorted_obj = {"exid": obj["exid"], "losses": sorted_losses}
                json.dump(sorted_obj, file, ensure_ascii=False)
                file.write("\n")
            f.close()
    logger.info("Sorted losses saved to %s", sorted_loss_output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input from config file.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config_file)
