#!/bin/sh
#SBATCH --job-name=r24
#SBATCH --gres=gpu:1 ## run on one gpu
#SBATCH --output output_rx24.out
#SBATCH --error output_rx24.err
#SBATCH -p v100-32gb-hiprio

#Load your modules first: THIS is all done in the .bashrc file
#module load python3/anaconda/2020.02
#module load gcc/7.3.0
#module load cuda/10.1
#source activate ../ml

# set to use first visible GPU in the machine
echo "hostname is " 
hostname

echo -e "\nCuda Vis devices assigned by slurm is:"

# print out what GPU, slurm scheduler assigned, 0 or 1
echo $CUDA_VISIBLE_DEVICES  

python ./main.py --batch_size 64 --lr 0.002  --exp_name exp24_sa16_lr2e_3_b64_0820
