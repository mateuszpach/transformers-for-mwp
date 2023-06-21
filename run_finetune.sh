#!/bin/bash
#SBATCH --job-name=transformather
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --partition=student

cd $HOME/transformers-for-mwp
source transformers-for-mwp/bin/activate

python -u finetune.py