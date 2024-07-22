#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=train-extract(150/nl/2.7B)
#SBATCH --mem=320000
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=train-extract-2.7B-model-%j.out

module purge
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source /scratch/s4079876/venvs/torch/bin/activate

python ./trainer.py --config_file exp-configs/EMEA/context/150/config-2.7B-nl.json --epoch 1

python ./extraction.py --config_file exp-configs/EMEA/context/150/config-2.7B-nl.json --model_dir /scratch/s4079876/finetuned/EMEA/context/nl-150-100-2.7B

deactivate