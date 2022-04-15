#!/bin/bash

#SBATCH --job-name big2rl
#SBATCH --ntasks=1
SBATCH --time=12:00:00

# available partition
# 1. cpu-share
# 2. gpu-share
# 3. himem-share

# selecting partition
SBATCH -p gpu-share

# **** example ****
# To use 2 cpu cores and 2 gpu devices in a node
##SBATCH -N 1 -n 2 --gres=gpu:4
# use 4 cpu cores
##SBATCH -N 1 -n 4
# *****************

# select cores
SBATCH -N 1 -n 4 --gres=gpu:2

# memory per node???
#SBATCH --mem=8G

# output file location
## SBATCH --output=/home/hcchengaa/ml-projects/big2-rl/slurm_report/%j.out

module load python3
source activate big2rl

srun which python # confirm python version

# testing basic function
# srun python helloworld.py

# # train with cpu
echo -e "\n\n\n Training"
cd ..
# srun python3 train.py --actor_device_cpu --training_device cpu -pt 8 10 13
srun python3 train.py --gpu_devices 0,1 --num_actor_devices 1 --num_actors 10 --training_device 1 -pt 8 10 13

echo -e "\n\n\n Generating eval data"
srun python3 generate_eval_data.py

echo -e "\n\n\n Evaluating"
srun python3 evaluate.py

# wait
echo -e "Training done"