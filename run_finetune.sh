#!/bin/bash
#SBATCH --job-name=transformather
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --partition=student

cd $HOME/transformers-for-mwp
#source transformers-for-mwp/bin/activate
conda activate mwp

echo STARTING complex
python -u finetune.py --model=base --epochs=15 --strategy=complex
echo STARTING complex_deductive
python -u finetune.py --model=base --epochs=15 --strategy=complex_deductive
echo STARTING complex_deductive_natural_language
python -u finetune.py --model=base --epochs=15 --strategy=complex_deductive_natural_language
echo STARTING complex_simplify
python -u finetune.py --model=base --epochs=15 --strategy=complex_simplify
echo STARTING deductive
python -u finetune.py --model=base --epochs=15 --strategy=deductive

echo STARTING complex-large
python -u finetune.py --model=large --epochs=15 --strategy=complex
echo STARTING complex_deductive-large
python -u finetune.py --model=large --epochs=15 --strategy=complex_deductive
echo STARTING complex_deductive_natural_language-large
python -u finetune.py --model=large --epochs=15 --strategy=complex_deductive_natural_language
echo STARTING complex_simplify-large
python -u finetune.py --model=large --epochs=15 --strategy=complex_simplify
echo STARTING deductive-large
python -u finetune.py --model=large --epochs=15 --strategy=deductive

