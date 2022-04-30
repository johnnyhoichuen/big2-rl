#!/bin/bash

#SBATCH --job-name big2rl
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

# available partition
# 1. cpu-share
# 2. gpu-share
# 3. himem-share

# selecting partition
#SBATCH -p gpu-share

# **** example ****
# To use 2 cpu cores and 2 gpu devices in a node
##SBATCH -N 1 -n 2 --gres=gpu:4
# use 4 cpu cores
##SBATCH -N 1 -n 4
# *****************

# select cores
#SBATCH -N 1 -n 4 --gres=gpu:2

# memory per node???
#SBATCH --mem=8G

# output file location
## SBATCH --output=/home/hcchengaa/ml-projects/big2-rl/slurm_report/%j.out
module load cuda

srun which python # confirm python version

# testing basic function
# srun python helloworld.py

# # train with cpu
echo -e "\n\n\nTraining"
cd ..
# if running on local machine, use:
# when playing against Charlesworth PPO, use -pt 14 16 18 (values must be distinct!) since the model was trained on ruleset of no increased penalty
# python3 train.py --actor_device_cpu --training_device cpu --opponent_agent ppo
# python3 generate_eval_data.py
# python3 evaluate.py
srun python train.py --gpu_devices 0,1 --num_actor_devices 1 --num_actors 10 --training_device 1 --opponent_agent ppo

echo -e "\n\n\nGenerating eval data"
srun python generate_eval_data.py

echo -e "\n\n\nEvaluating"
srun python evaluate.py

# wait
echo -e "\nTraining done"