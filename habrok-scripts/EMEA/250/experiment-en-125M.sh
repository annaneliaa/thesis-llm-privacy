#!/bin/bash
#SBATCH --time=26:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=experiment-full(250/EN/125M)
#SBATCH --mem=8000
#SBATCH --gpus-per-node=a100:1

module purge
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/venvs/torch/bin/activate

python ./trainer.py --config_file exp-configs/EMEA/250/config-125M-en.json

python ./split_dataset.py --config_file exp-configs/EMEA/250/config-125M-en.json

python ./extraction.py --config_file exp-configs/EMEA/250/config-125M-en.json --model_dir /scratch/s4079876/finetuned/EMEA/en-250-100-125M

deactivate