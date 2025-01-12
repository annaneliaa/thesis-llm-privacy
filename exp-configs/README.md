# Configuration files
The configuration files contain information shared in preprocessing, training, and extraction/attacks. The meaning of the variables is described in the `load_constants_from_config` method in util_lib.

## Naming convetions
Experiments are named after what they do. The general scheme is [lang]-[example_token_len / raw]-[example_token_len / nat]-[model_size]-e[number of epochs].
An example would be en-100-nat-1.3B, which represents the experiment in English with preprocessing to token length 100, without normalization, and on the model with 1.3B parameters. Notice that by convention if the number of epochs in the experiment is one, we omit the -e[number of epochs] at the end of the name. Doing so and running the experiment first on one epoch of training will result in a small time-complexity optimization (see the `mia_comp` function in `mia.py`)

## Variable values
- batch_size: This can be 64 for the experiments with 125M parameters. For all other models, the program will run out of GPU memory; hence 32 should be used for those.