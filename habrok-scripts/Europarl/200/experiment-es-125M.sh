#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=experiment-full(200/ES/125M)
#SBATCH --mem=8000
#SBATCH --gpus-per-node=a100:1

module purge
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source .env/bin/activate

export PATH="/home1/s6153712/short-programming-llm-privacy/.env/bin:$PATH"
export PYTHONPATH="/home1/s6153712/short-programming-llm-privacy/.env/lib/python3.11/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/home1/s6153712/short-programming-llm-privacy/.env/lib:$LD_LIBRARY_PATH"

python ./trainer.py --config_file exp-configs/Europarl/200/config-125M-es.json
# CUDA_LAUNCH_BLOCKING=1 python ./extraction.py --config_file exp-configs/Europarl/200/config-125M-es.json --model_dir /scratch/s6153712/llm-privacy/hf_cache/finetuned/Europarl/es-200-100-125M --cache_dir /scratch/s6153712/llm-privacy/hf_cache/

deactivate