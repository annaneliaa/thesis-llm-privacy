# Thesis LLM Privacy

This repository contains the code and resources for the thesis project on LLM (Large Language Model) privacy. The project involves processing datasets, training models, and evaluating their performance using various metrics.

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

4. **Set up the environment:**
    ```
    conda env create -f torch-conda.yml
    conda activate <env-name>
    ```

## Usage

### Data Processing

To process the data, run the [`process_data.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fannavisman%2Fstack%2FRUG%2FCS%2FYear3%2Fthesis%2Fthesis-llm-privacy%2Fprocess_data.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/annavisman/stack/RUG/CS/Year3/thesis/thesis-llm-privacy/process_data.py") script with the appropriate configuration file:

    
    python process_data.py --config_file config.json
    

### Preprocessing

To preprocess the data, run the [`preprocessing.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fannavisman%2Fstack%2FRUG%2FCS%2FYear3%2Fthesis%2Fthesis-llm-privacy%2Fpreprocessing.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/annavisman/stack/RUG/CS/Year3/thesis/thesis-llm-privacy/preprocessing.py") script:

    
    python preprocessing.py --config_file config.json
    

### Training

To train the model, run the [`trainer.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fannavisman%2Fstack%2FRUG%2FCS%2FYear3%2Fthesis%2Fthesis-llm-privacy%2Ftrainer.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/annavisman/stack/RUG/CS/Year3/thesis/thesis-llm-privacy/trainer.py") script:

    
    python trainer.py --config_file config.json --epochs <number_of_epochs>
    

### Evaluation

To evaluate the model, run the [`evaluation.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fannavisman%2Fstack%2FRUG%2FCS%2FYear3%2Fthesis%2Fthesis-llm-privacy%2Fevaluation.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/annavisman/stack/RUG/CS/Year3/thesis/thesis-llm-privacy/evaluation.py") script:

    
    python evaluation.py --config_file config.json --trained True
    

### Calculate Scores

To calculate BLEU scores, run the [`calculate_scores.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fannavisman%2Fstack%2FRUG%2FCS%2FYear3%2Fthesis%2Fthesis-llm-privacy%2Fcalculate_scores.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/annavisman/stack/RUG/CS/Year3/thesis/thesis-llm-privacy/calculate_scores.py") script:

    
    python calculate_scores.py --config_file config.json
    

### Accuracy

To calculate accuracy, run the [`accuracy.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fannavisman%2Fstack%2FRUG%2FCS%2FYear3%2Fthesis%2Fthesis-llm-privacy%2Faccuracy.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/annavisman/stack/RUG/CS/Year3/thesis/thesis-llm-privacy/accuracy.py") script:

    
    python accuracy.py --config_file config.json
    

## Configuration

The configuration file ([`config.json`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fannavisman%2Fstack%2FRUG%2FCS%2FYear3%2Fthesis%2Fthesis-llm-privacy%2Fconfig.json%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/annavisman/stack/RUG/CS/Year3/thesis/thesis-llm-privacy/config.json")) contains various settings required for data processing, training, and evaluation. Ensure that the paths and parameters are correctly set before running the scripts.

## Logging

Logging is configured to display information in the console. You can adjust the logging level and format in each script as needed.