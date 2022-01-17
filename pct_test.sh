#!/bin/sh
#SBATCH --job-name=rx24
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1 ## run on one gpu
#SBATCH --exclude=node[178-238]
#SBATCH --output output_16sa_b64_lr50e4_0807.out
#SBATCH --error output_16sa_b64_lr50e4_0807.err
#SBATCH -p gpu-v100-16gb

#Load your modules first:
module load python3/anaconda/2020.02
module load gcc/7.3.0
module load cuda/10.1
source activate ../ml

# set to use first visible GPU in the machine
#export CUDA_VISIBLE_DEVICES=0

echo $CUDA_VISIBLE_DEVICES

python ./main.py --batch_size 64 --lr 0.005  --exp_name sa16_lr50e_4_b64_0809
