#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=experiment-tokenization{100}
#SBATCH --mem=32000
#SBATCH --gpus-per-node=a100:1

module purge
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/thesis-llm-privacy/.env/bin/activate

python ./canary_insertion.py --config_file exp-configs/EMEA/canary/i1/config-125M-nl.json --dataset_name EMEA-c --insertions 1
python ./preprocessing.py --config_file exp-configs/EMEA/canary/i1/config-125M-nl.json
python ./process_data.py --config_file exp-configs/EMEA/canary/i1/config-125M-nl.json
python ./split_train_val.py --config_file exp-configs/EMEA/canary/i1/config-125M-nl.json
python ./tokenize_data.py --config_file exp-configs/EMEA/canary/i1/config-125M-nl.json
python ./tokenize_data.py --config_file exp-configs/EMEA/canary/i1/config-125M-en.json

deactivate