#!/bin/bash

#SBATCH --job-name big2rl
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

# available partition
# 1. cpu-share
# 2. gpu-share
# 3. himem-share

# selecting partition
#SBATCH -p cpu-share

# **** example ****
# To use 2 cpu cores and 2 gpu devices in a node
##SBATCH -N 1 -n 2 --gres=gpu:4
# use 4 cpu cores
#SBATCH -N 1 -n 4
# *****************

# select cores
#SBATCH -N 1 -n 1

# memory per node???
#SBATCH --mem=8G

# output file location
## SBATCH --output=/home/hcchengaa/ml-projects/big2-rl/slurm_report/%j.out

# module load python  # Lmod has detected error, unknown module python
# source activate big2rl  # No such file or directory

srun which python # confirm python version. This should be executed if we used 'conda activate big2rl' before

# testing basic function
# srun python helloworld.py

# # train with cpu
echo -e "\n\n\n Training"
cd ..
srun python train.py --actor_device_cpu --training_device cpu -pt 8 10 13

echo -e "\n\n\n Generating eval data"
srun python generate_eval_data.py

echo -e "\n\n\n Evaluating"
srun python evaluate.py

# wait
echo -e "Training done"
