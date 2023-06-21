#!/bin/bash
#SBATCH --job-name=transformather
#SBATCH --qos=test
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=student

cd $HOME/transformers-for-mwp
python3 -m venv transformers-for-mwp
source transformers-for-mwp/bin/activate
pip install accelerate==0.20.1 evaluate pytesseract transformers datasets rouge-score nltk tensorboard wandb py7zr --upgrade