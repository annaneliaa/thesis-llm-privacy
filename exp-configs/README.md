# Configuration files
The configuration files contain information shared in preprocessing, training, and extraction/attacks. The meaning of the variables is described in the `load_constants_from_config` method in util_lib.

## Naming convetions
- Experiment names: Experiments are named after what they do. The general scheme is [lang]-[example_token_len / raw]-[example_token_len / nat]-[model_size]-[experiment_type]-[E/I][number of epochs / number of insertions].
An example would be en-100-nat-1.3B-mia-E1, which represents the experiment in English with preprocessing to token length 100, without normalization, on the model with 1.3B parameters, and for two epochs of training. Notice that the mia experiments miss the experiment type, this should be fixed in the future 

## Variable values
- batch_size: This can be 64 for the models with 125M parameters, 32 for the 1.3B models, and 16 for the 2.7B models.