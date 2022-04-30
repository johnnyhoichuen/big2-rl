import os
import threading
import time
import timeit
import pprint
from collections import deque

import torch
from torch import multiprocessing as mp
from torch import nn

from big2_rl.deep_mc.file_writer import FileWriter
# from big2_rl.deep_mc.model import Big2Model
from big2_rl.deep_mc.model import Big2ModelResNet
from big2_rl.deep_mc.utils import get_batch, log, create_buffers, act

# only save the mean episode return of one position (observed player)
mean_episode_return_buf = deque(maxlen=100)

# selected activation function
activation = 'relu'


def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss


def learn(actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock=threading.Lock()):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')
    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf.append(torch.mean(episode_returns).to(device))

    with lock:
        # (obs_z) should be size (NL,4,208)
        # (obs_x) should ge size (NL,559)
        learner_outputs = model(obs_z, obs_x, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)  # compute loss
        mean_ep_ret = torch.mean(
            torch.stack([_r for _r in mean_episode_return_buf])).item()
        stats = {
            'mean_episode_return_actor': mean_ep_ret,
            'loss': loss.item(),
        }

        torch.autograd.set_detect_anomaly(True)

        # backpropagation and gradient clipping
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        # update actor models to have weights of the learner model
        for pos in actor_models.keys():
            actor_models[pos].load_state_dict(model.state_dict())
        return stats


def train(flags):
    """
    Based on the TorchBeast Monobeast implementation. Creates 'flags.num_buffers' many rollout buffers,
    each containing shared-memory tensors without batch dimension
    Create 2 shared queues 'free_queue' and 'full_queue' for each actor device which will communicate ints by Unix pipes
    Start 'flags.num_actor_devices' many actor processes, each with a copy of the environment,
    which dequeues an index from free_queue and writes a batch-slice (size 'flags.batch_size') with rollout data into
    buffers[index] then enqueues index to full_queue and dequeues the next index.

    Meanwhile, each actor device has many learner threads, which dequeues 'flags.batch_size' many indices
    from 'full_queue', stacks them together into batch and moves them to GPU, puts indices back into 'free_queue', then
    sends batch through model, compute losses, does backward pass, and updates weights.
    """

    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. \
            Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")

    # initialise FileWriter object with current experiment id, dictionary of cmdline flags,
    # and string corresponding to root directory where model weights will be saved
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    # initialise checkpoint path to where model weights will be periodically saved
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    print('ckpt path: ', checkpointpath)

    T = flags.unroll_length
    B = flags.batch_size

    # TODO
    # Initialize actor models
    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(
            flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    models = {}
    # create model on each actor device and moves it to shared memory
    for device in device_iterator:
        # model = Big2Model(device)
        model = Big2ModelResNet(device, activation=activation)
        model.share_memory()
        model.eval()  # actors shouldn't be training (ie receiving weight updates)
        models[device] = model

    # Initialize buffers
    buffers = create_buffers(flags, device_iterator)

    # Initialize queues
    actor_processes = []
    # parent process starts a fresh python interpreter process with 'spawn' on Linux and Windows
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}
    # create one free_queue and full_queue per actor device
    for device in device_iterator:
        _free_queue = ctx.SimpleQueue()
        _full_queue = ctx.SimpleQueue()
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    # Create learner model for training on the ONE training_device
    # learner_model = Big2Model(flags.training_device)
    learner_model = Big2ModelResNet(flags.training_device, activation=activation)

    # Create globally shared optimizer for all positions
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha
    )

    # Stat Keys (just 1 since all 4 positions will use the same model)
    stat_keys = [
        'mean_episode_return_actor',
        'loss'
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}

    # Load prior model and optimizer if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath,
            map_location="cuda:" + str(flags.training_device) if flags.training_device != "cpu" else "cpu"
        )
        learner_model.load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])

        for device in device_iterator:  # loads actor models
            models[device].load_state_dict(learner_model.state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        # log.info(f"Resuming preempted job, current stats:\n{stats}")

    # Starting actor processes on each actor device
    for device in device_iterator:
        # each actor process shares the same free queue, full queue, actor model, and buffers as all other actor
        # processes on the same device
        for i in range(flags.num_actors):
            # target=act refers to the dmc/utils act(...) function that is called whenever the process is running
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, curr_device, local_lock, lock=threading.Lock()):
        """Thread target for the learning process (takes batch -> forward pass -> loss -> backprop)"""
        nonlocal frames, stats
        while frames < flags.total_frames:  # train the model for 'flags.total_frames' many frames
            batch = get_batch(free_queue[curr_device], full_queue[curr_device],
                              buffers[curr_device],
                              flags, local_lock)
            _stats = learn(models, learner_model, batch,
                           optimizer, flags)
            with lock:  # critical section: update stats and log them for each learner thread
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B

    # to start: for each actor device, put each buffer index into that device's free queue
    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device].put(m)

    # for each actor device, create 'flags.num_threads' many threads and these conduct learner process
    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = threading.Lock()
    for device in device_iterator:
        for i in range(flags.num_threads):
            thread = threading.Thread(
                target=batch_and_learn, name='batch-and-learn-%d' % i,
                args=(i, device, locks[device]))
            thread.start()
            threads.append(thread)

    def checkpoint(num_frames):
        """
        Saves model and optimizer state dict periodically
        """
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        # save the state_dict of learner model
        torch.save({
            'model_state_dict': learner_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats': stats,
            'flags': vars(flags),
            'frames': num_frames,
        }, checkpointpath)

        # Save the weights of the learner model for evaluation purpose
        model_weights_dir = os.path.expandvars(os.path.expanduser('%s/%s/%s' % (
            flags.savedir, flags.xpid, '_weights_' + str(frames) + '.ckpt')))
        torch.save(learner_model.state_dict(), model_weights_dir)

    # handles timing and sleeps when flags.save_interval minutes pass, checkpoints the current state dict
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()

            end_time = timer()
            fps = (frames - start_frames) / (end_time - start_time)  # stores num frames computed in this time interval
            log.info('After %i frames: @ %.1f fps - Stats:\n%s',
                     frames,
                     fps,
                     pprint.pformat(stats))

    except KeyboardInterrupt:
        return  # try joining actors then quit
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
