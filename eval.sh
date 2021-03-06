#!/bin/bash

# for slurm only
#SBATCH --job-name big2-eval
#SBATCH --ntasks=1
#SBATCH -p cpu-share
#SBATCH -N 2 -n 2

function evaluate() {
    train_opponent=$1
    model=$2
    eval_opponent=$3
    frame_trained=$4
    echo "\n-------------------------\n\n"
    echo "evaluating: (training with: $train_opponent, using model: $model, with $frame_trained frames) vs ($eval_opponent)"
    # for local computer
     python evaluate.py --model_type $model --train_opponent $train_opponent --eval_opponent $eval_opponent --frames_trained $frame_trained

    # for slurm
#    srun python evaluate.py --model_type $model --train_opponent $train_opponent --eval_opponent $eval_opponent --frames_trained $frame_trained
#    echo "\n-------------------------\n"
}

# generate eval data once
python generate_eval_data.py --num_games 500
# srun python generate_eval_data.py --num_games 1000

 echo "============================================"
 echo "eval opponent = ppo"

 ## training with ppo and eval with ppo
 evaluate 'ppo' 'standard' 'ppo' 2009600
 evaluate 'ppo' 'residual' 'ppo' 2006400
 evaluate 'ppo' 'conv' 'ppo' 2009600
 evaluate 'ppo' 'convres' 'ppo' 2009600

 ## training with prior and eval with ppo
 evaluate 'prior' 'standard' 'ppo' 2009600
 evaluate 'prior' 'residual' 'ppo' 2595200
 evaluate 'prior' 'conv' 'ppo' 2009600
 evaluate 'prior' 'convres' 'ppo' 2009600

 ## training with random and eval with ppo
 evaluate 'random' 'standard' 'ppo' 2009600
 evaluate 'random' 'residual' 'ppo' 4851200
 evaluate 'random' 'conv' 'ppo' 2009600
 evaluate 'random' 'convres' 'ppo' 2009600










# echo "============================================ \n\n\n"
# echo "eval opponent = prior \n\n"

# ## training with ppo and eval with prior
# evaluate 'ppo' 'standard' 'prior' 2009600
# evaluate 'ppo' 'residual' 'prior' 2006400

# ## below 2 are buggy
# #evaluate 'ppo' 'conv' 'prior' 2009600
# #evaluate 'ppo' 'convres' 'prior' 2009600
# #  File "/Users/johnnycheng/Documents/Python Projects/Anaconda Projects/big2-rl/venv/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1498, in load_state_dict
# #    self.__class__.__name__, "\n\t".join(error_msgs)))
# #RuntimeError: Error(s) in loading state_dict for Big2ModelConv:
# #        Unexpected key(s) in state_dict: "lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0", "dense2.blocks.0.weight", "dense2.blocks.0.bias", "dense3.blocks.0.weight", "dense3.blocks.0.bias", "dense4.blocks.0.weight", "dense4.blocks.0.bias", "dense5.blocks.0.weight", "dense5.blocks.0.bias".
# #        size mismatch for dense1.weight: copying a param with shape torch.Size([512, 687]) from checkpoint, the shape in current model is torch.Size([512, 1071]).


# ## eval vs prior
# evaluate 'prior' 'standard' 'prior' 2009600
# evaluate 'prior' 'residual' 'prior' 2595200
# ## below 2 are buggy for the same reason
# #evaluate 'prior' 'conv' 'prior' 2009600
# #evaluate 'prior' 'convres' 'prior' 2009600

# evaluate 'random' 'standard' 'prior' 2009600
# evaluate 'random' 'residual' 'prior' 4851200
# # ## below 2 are buggy
# # #evaluate 'random' 'conv' 'prior' 2009600
# # #evaluate 'random' 'convres' 'prior' 2009600












#echo "============================================ \n\n\n"
#echo "eval opponent = random \n\n"
#
### training with ppo and eval with random
#evaluate 'ppo' 'standard' 'random' 2009600
#evaluate 'ppo' 'residual' 'random' 2006400
#evaluate 'ppo' 'conv' 'random' 2009600
#evaluate 'ppo' 'convres' 'random' 2009600
#
### training with prior and eval with random
#evaluate 'prior' 'standard' 'random' 2009600
#evaluate 'prior' 'residual' 'random' 2595200
#evaluate 'prior' 'conv' 'random' 2009600
#evaluate 'prior' 'convres' 'random' 2009600
#
### training with random and eval with random
#evaluate 'random' 'standard' 'random' 2009600
#evaluate 'random' 'residual' 'random' 4851200
#evaluate 'random' 'conv' 'random' 2009600
#evaluate 'random' 'convres' 'random' 2009600