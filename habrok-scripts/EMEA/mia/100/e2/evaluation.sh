#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=evaluation-epochs-2
#SBATCH --mem=8000
#SBATCH --gpus-per-node=a100:1

module purge
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/thesis-llm-privacy/.env/bin/activate

python ./mia_evaluation.py --config_file exp-configs/EMEA/100/e2/config-125M-nl.json --eval_mode epoch --epochs 2

deactivate