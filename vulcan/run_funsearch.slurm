#!/bin/bash
#SBATCH --job-name=funsearch_baseline
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=vulcan/log/random_sampler_hf_%j.out
#SBATCH --error=vulcan/log/random_sampler_hf_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --account=aip-lelis

##replace the venv path here
source new_dsl_env/bin/activate

##python -m funsearch.implementation.funsearch --model_type huggingface --function collect --spec_file  prompt_specifications/specification_jocelyn.txt

##python3 prog_synth_pipeline/program_synthesis.py prog_synth_pipeline/task_config.json

python3 -m funsearch.implementation.funsearch --spec_file prompt_specifications/specification_with_updated_nld_baseline.txt --func_file function_specific_prompts/make_arrow_base.txt --func_init functions_generated/baseline.py --model_type huggingface
