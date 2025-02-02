#!/bin/bash
#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=experiment-full(100/NL/125M-i2)
#SBATCH --mem=16000
#SBATCH --gpus-per-node=a100:1

module purge
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/thesis-llm-privacy/.env/bin/activate

python ./trainer.py --config_file exp-configs/EMEA/canary/i2/config-125M-nl.json --epochs 1
python ./canary_attack.py --config_file exp-configs/EMEA/canary/i2/config-125M-nl.json
python ./canary_evaluation.py --config_file exp-configs/EMEA/canary/i2/config-125M-nl.json

deactivate