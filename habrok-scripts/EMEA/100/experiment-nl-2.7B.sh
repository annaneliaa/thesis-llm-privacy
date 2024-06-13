#!/bin/bash
#SBATCH --time=26:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=experiment-full(100/NL/2.7B)
#SBATCH --mem=32000
#SBATCH --gpus-per-node=a100:1

module purge
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/venvs/torch/bin/activate

python ./trainer.py --config_file exp-configs/EMEA/100/config-2.7B-nl.json

python ./split_dataset.py --config_file exp-configs/EMEA/100/config-2.7B-nl.json

python ./extraction.py --config_file exp-configs/EMEA/100/config-2.7B-nl.json --model_dir /scratch/s4079876/finetuned/EMEA/nl-100-100-2.7B

deactivate