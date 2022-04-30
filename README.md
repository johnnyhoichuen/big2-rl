# Big Two with Deep Reinforcement Learning

Big2-RL is a reinforcement learning framework for [Big Two](https://en.wikipedia.org/wiki/Big_two) (Cantonese: 鋤大弟), a four-player card-shedding game popular in many Southeast Asia countries played with a standard 52-card deck without jokers. Our framework uses multiprocessing and is heavily inspired by DouZero and TorchBeast (see Acknowledgements).

Each player's goal is to empty their hand of all cards before other players. Cards are shed (or played) through tricks consisting of specific hand types (singles, pairs, triples and five-card hands), and each player must either follow the trick by playing a higher-ranked hand of the same type as the person who led the trick, or pass. If all other players pass, the person who won the trick "leads" the next trick and chooses which hand type to play. When one player empties their hand (the winner), the remaining players are penalised based on the number of cards left in their hands, and these penalties are awarded to the winner.

In contrast to Dou Dizhu, which has clearly defined roles for each player, collaboration in Big Two is much more fluid. For instance, it is common for players to pass when the opponent preceding them has just played in the hopes of preserving higher ranked cards for later and having the opportunity to play second in case the player before them gets to lead the next trick. Conversely, players tend to play cards if the player after them has played because if the player following them leads the next trick, they will have to play last on the subsequent round. Additionally, Dou Dizhu has no additional penalty for having lots of unplayed cards, whereas Big Two is inherently more risky since although more cards usually means more manoeuvrability, it also incurs a higher penalty if they lose, and vice versa.

## Big Two Challenges

TODO

In this work, we explore a variety of algorithms and structures and evaluate their respective performances. Please read [our paper](TODO) for more details.

## Installation
The training code is designed for GPUs. Thus, you need to first install CUDA if you want to train models. You may refer to [this guide](https://docs.nvidia.com/cuda/index.html#installation-guides). For evaluation, CUDA is optional and you can use CPU for evaluation.

First, clone the repo with:
```
git clone https://github.com/johnnyhoicheng/big2-rl.git
```
Make sure you have python 3.6+ installed. Install dependencies:
```
cd big2-rl
pip3 install -r requirements.txt
```

## Training [TODO]

### Using SLURM
Use the following commands to schedule jobs in your computer cluster
sbcppo (='sbatch slurm_cpu_vs_ppo.sh')
sbcpri (='sbatch slurm_cpu_vs_prior.sh')
sbcran (='sbatch slurm_cpu_vs_rand.sh')
# sbg (='sbatch slurm_gpu.sh')
sq (='squeue')
lo (='./latest_slurm_out.sh')

###
To use GPU for training, run
```
python3 train.py
```
This will train DouZero on one GPU. To train DouZero on multiple GPUs. Use the following arguments.
*   `--gpu_devices`: what gpu devices are visible
*   `--num_actor_devices`: how many of the GPU devices will be used for simulation, i.e., self-play
*   `--num_actors`: how many actor processes will be used for each device
*   `--training_device`: which device will be used for training DouZero

For example, if we have 4 GPUs, where we want to use the first 3 GPUs to have 15 actors each for simulating and the 4th GPU for training, we can run the following command:
```
python3 train.py --gpu_devices 0,1,2,3 --num_actor_devices 3 --num_actors 15 --training_device 3
```
To use CPU training or simulation (Windows can only use CPU for actors), use the following arguments:
*   `--training_device cpu`: Use CPU to train the model
*   `--actor_device_cpu`: Use CPU as actors

For example, use the following command to run everything on CPU:
```
python3 train.py --actor_device_cpu --training_device cpu
```
The following command only runs actors on CPU:
```
python3 train.py --actor_device_cpu
```
For more customized configuration of training, see the following optional arguments: [TODO add the rest]
```
--xpid XPID           Experiment id (default: douzero)
--save_interval SAVE_INTERVAL
                      Time interval (in minutes) at which to save the model
--actor_device_cpu    Use CPU as actor device
--gpu_devices GPU_DEVICES
                      Which GPUs to be used for training
--num_actor_devices NUM_ACTOR_DEVICES
                      The number of devices used for simulation
--num_actors NUM_ACTORS
                      The number of actors for each simulation device
--training_device TRAINING_DEVICE
                      The index of the GPU used for training models. `cpu`
                	  means using cpu
--load_model          Load an existing model
--disable_checkpoint  Disable saving checkpoint
--savedir SAVEDIR     Root dir where experiment data will be saved
--total_frames TOTAL_FRAMES
                      Total environment frames to train for
--exp_epsilon EXP_EPSILON
                      The probability for exploration
--batch_size BATCH_SIZE
                      Learner batch size
--unroll_length UNROLL_LENGTH
                      The unroll length (time dimension)
--num_buffers NUM_BUFFERS
                      Number of shared-memory buffers for a given actor device
--num_threads NUM_THREADS
                      Number learner threads
--max_grad_norm MAX_GRAD_NORM
                      Max norm of gradients
--learning_rate LEARNING_RATE
                      Learning rate
--alpha ALPHA         RMSProp smoothing constant
--momentum MOMENTUM   RMSProp momentum
--epsilon EPSILON     RMSProp epsilon
```

## Evaluation [TODO]
The evaluation can be performed with GPU or CPU (GPU will be much faster). Pretrained model is available at [Google Drive](https://drive.google.com/drive/folders/1NmM2cXnI5CIWHaLJeoDZMiwt6lOTV_UB?usp=sharing) or [百度网盘](https://pan.baidu.com/s/18g-JUKad6D8rmBONXUDuOQ), 提取码: 4624. Put pretrained weights in `baselines/`. The performance is evaluated through self-play. We have provided pretrained models and some heuristics as baselines:
*   [random](big2_rl/evaluation/random_agent.py): agents that play randomly (uniformly)
*   SL (`baselines/sl/`): the pre-trained deep agents on human data
*   DouZero-ADP (`baselines/douzero_ADP/`): the pretrained DouZero agents with Average Difference Points (ADP) as objective
*   DouZero-WP (`baselines/douzero_WP/`): the pretrained DouZero agents with Winning Percentage (WP) as objective

### Step 1: Generate evaluation data
```
python3 generate_eval_data.py
```
Some important hyperparameters are as follows.
*   `--output`: where the pickled data will be saved
*   `--num_games`: how many random games will be generated, default 10000

### Step 2: Self-Play [TODO add the settings]
```
python3 evaluate.py
```
Some important hyperparameters are as follows.
* `--south`: which agent will play as the South player, which can be random, or the path of the pre-trained DMC model
* `--east`: which agent will play as the East player, which can be random, or the path of the pre-trained DMC model
* `--north`: which agent will play as the North player, which can be random, or the path of the pre-trained DMC model
* `--west`: which agent will play as the West player, which can be random, or the path of the pre-trained DMC model
* `--eval_data`: the pickle file that contains evaluation data
* `--num_workers`: how many subprocesses will be used to run evaluation data
* `--gpu_device`: which GPU to use. It will use CPU by default

For example, the following command evaluates performance of a DMC Agent trained for 200 frames against random agents in all other positions
```
python3 evaluate.py --south baselines/weights_200.ckpt --east random --west random --north random
```
The following command evaluates performance of DMC Agent trained for 200 frames at south and east positions, but trained for 4000 frames at north, and 10000 frames at west.
```
python3 evaluate.py --south baselines/weights_200.ckpt --east baselines/weights_200.ckpt --west baselines/weights_10000.ckpt --north baselines/weights_4000.ckpt
```
By default, our model will be saved in `big2rl_checkpoints/big2rl` every 20 minutes.

## Core Team
*   [Jasper Chow](https://github.com/jchow-ust)
*   [Johnny Cheng](https://github.com/johnnyhoichuen)

## Cite this Work
If you find this project helpful in your research, please cite our paper:

TODO

## Acknowledgments [TODO incomplete]
*   Zha, Daochen et al. “DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning.” ICML (2021).

