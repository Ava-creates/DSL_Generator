#!/bin/bash
#SBATCH --job-name=funsearch
#SBATCH --output=random_sampler_hf_%j.out
#SBATCH --error=random_sampler_hf_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --account=aip-lelis

#replace the venv path here
source carft/bin/activate

python -m funsearch.implementation.funsearch --model_type huggingface --function collect --spec_file  prompt_specifications/specification_jocelyn.txt