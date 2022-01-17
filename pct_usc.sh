#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=72:00:00
#SBATCH --output output_16sa.out
#SBATCH --error output_16da.err


module purge
module load python3/anaconda/2020.02
module load cuda/9.0
source activate ../ml

export CUDA_VISIBLE_DEVICES=0

python ./main.py

