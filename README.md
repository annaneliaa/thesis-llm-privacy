# LLM Privacy Attack Language Comparison

This repository contains the code and resources for several privacy attacks on Large Language Models (LLMs). Currently, three methodologies are implemented: Training data extraction attacks, membership inference attacks, and canary attacks. The attacks involve processing datasets, training models, and evaluating their performance using various metrics. They are meant to be performed on two languages to compare their memorization

## Setup

1. **Clone the repository:**
    ```sh
    git clone git@github.com:annaneliaa/thesis-llm-privacy.git
    cd thesis-llm-privacy
    ```

2. **Create a virtual environment and activate it:**
    ```sh
    python -m venv .env
    source .env/bin/activate  # On Windows use `.env\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

# Usage

All code I used (so the code exclusive the training data extraction attack) is meant to run as one pipeline. Do to the nature of the experiments, this is split up into different stages. Before running any scripts make sure to:
- Store two aligned datasets with the name specified in the `dataset_name` field and endings ".en" and ".nl" in the `dataset_dir` specified in the config file. 
- Change the `cache_dir` variable in `util_lib.py` to a directory that has sufficient (fast) memory.
After running all scripts, the results will be in the `root_dir` specified in the config file (at least for the membership inference and canary attacks)

## Configuration

The configuration file ([`config.json`]) contains various settings required for data processing, training, and evaluation. Ensure that the paths and parameters are correctly set before running the scripts. Some fields are explained below, all are described in the `load_constants_from_config` method in `util_lib.py`.

## Data Processing

The regular pre-processing pipeline involves running multiple scripts (see as examples the `tokenization.sh` scripts in the harbrok-scripts folder):

    
    python preprocessing.py --config_file config.json
    python process_data.py --config_file config.json
    python split_train_val.py --config_file config.json
    python tokenize_data.py --config_file config.json


- The `preprocessing.py` script concatenates data into larger sentences. The target length is the `EXAMPLE_TOKEN_LEN` specified in the config file. If this is not desired, it can be omitted. If that is the case, or alternatively, the `preprocessing` field in the config file must be set to false.
- The `process_data.py` script normalizes the dataset to the token length specified in `example_token_len`, i.e., it deletes sentences that are too short and truncates sentences that are too long. If this is not desired, set the `normalization` field in the confi file to false.
- The `split_train.py` script splits the dataset into disjoint training and validation sets. The share of sentences in the validation set is determined by the `VAL_SPLIT` field in the config file. Typically, it should be around 0.1.
- The `tokenize_data.py` script tokenizes the data set. If the `batching` field is set to true, this will be done in batches, otherwise the sentences are tokenized with uniform length. This script, other than the previous ones, needs to be run for each language individually (see the examples)

## Training

To train the model, run the `trainer.py` script:

    
    python trainer.py --config_file config.json --epochs <number_of_epochs>
    

## Training Data Extraction Attacks

Disclaimer: I have not spend any time looking into the evaluation steps of this attack methodology. I think that my changes should be compatible with the scripts, but I have not tried this. In any case, incompatibilities should be because of changed file paths, all of which can now be retrieved from `util_lib.py`. Hence, fixing them should be very straightforward. Additionally, it might be necessary to reduce the dataset size before extraction.

### Extraction
For the training data extraction attacks, run the `split_dataset.py` script at the end of the preprocessing, or in any case before the `extraction.py` script.

    
    python split_dataset.py --config_file config.json
    python extraction.py --config_file config.json --model_dir <path_to_model> --cache_dir <cache_dir>

- The `split_dataset.py` script splits the dataset into prefixes and suffixes. The respective length is specified in the `prefix_len`, `preprefix_len` and `suffix_len` fields in the config file.
- The `extraction.py` script first extracts strings of `suffix_len` from the provided model inputing the prefixes previously generated, and calculates the losses associated with the output strings. 


### Evaluation

There are multiple scripts to evaluate the training data extraction attacks. I am unaware of what they do, but the names are fairly descriptive

    
    python evaluation.py --config_file config.json --trained True
    python calculate_scores.py --config_file config.json
    python accuracy.py --config_file config.json
    

## Membership Inference Attacks
For the membership inference attack, run the following scripts after training:
### Extraction

    
    python mia.py --config_file config.json
This script will run the membership inference attack storing the loss of the untrained model, of the trained model, and the perplexity ratio. See two former ones are simply lists (or lists of lists if batching is activated), the latter is a (list of) dictionaries from sentence ids to to perplexity ratio. Optionally a cache directory can be specified via a --cache_dir flag.

### Evaluation

    
    python mia_evaluation.py --config_file config.json --eval_mode <mode> --epochs <int>
This script generates statistics and plots based on the membership inference attacks. Both eval_mode and epochs are optional flags. The default is to evaluate the experiment specified in the config file. If the eval_mode flag is set to "model", then there will be cross-experiment evaluation of all experiments run on the model specified in the config file. If it is set to "epoch", then there will be cross-experiment evaluation of all experiments for the number of epochs specified in the epochs flag.

## Canaray Attacks
For a canary attack, first store two json files (one for each language) in the directory specified in the `dataset_dir` field in the config file. These files must be named "canary-en.json" and "canary-nl.json", and contain a field "prefix" and a field "suffix" that in combination make up the canary sentence. For instance, the prefix could be "The social security number of Daniel Johann Seidel is", and the suffix could be "5683507". 
### Pre-processing
Run the following before any pre-processing:

    
    python canary_insertion.py --config_file config.json --dataset_name <str> --insertions <int>
The dataset_name is the name of the dataset in the `dataset_dir` directory. This should be distinct from the `dataset_name` in the config file, otherwise the original dataset is overwritten.
Additinally, add the --canaries_train flag when running the `split_train_val.py` script.
### Extraction
The extraction is done in the `canary_attack.py` script. Run it as follows: 

    
    python canary_attack.py --config_file config.json 
This script samples candidate canaries, tokenizes them and calculates their loss. It then approximates the distribution of the samples, calculates the loss of the actual canary, and computes the exposure. Exposure, canary loss, and distribution parameters are stored in json format.
### Evaluation
To evaluate the canary attacks, run

    
    python canary_attack.py --config_file config.json --eval_mode <mode>
The default eval_mode evaluates the experiment specified in the config files. It plots the extraction findings and performs goodness of fit tests. The other eval_mode is "insertions". It evaluates the exposure, loss, and goodness of fit of all experiments run so far. The experiment names are hard-coded in this script, if they change the script must be changed.

## Changing Languages
Changing the language of the experiments could essentially be done by a global search of the strings "en" and "nl", and replacing them with the new languages. Alternatively, one can simply run the experiments under a false language name, and adopt the plotting. Either way, it is advisable to make the language an easily changeable variable in the future. For instance, the languages could be stored in `util_lib` as is the cache directory.

## Filtering sentences per batch
To get a uniform distribution of the sentences in each batch, I advise to interfere in the `tokenize_prompts_in_batches` method, where these batches are created and it is easy to delete sentences from it. This could also be done later, but then there is some overhead from tokenize not needed sentences. Notice that the method is used by multiple scripts. 

## Logging

Logging is configured to display information in the console. You can adjust the logging level and format in each script as needed.