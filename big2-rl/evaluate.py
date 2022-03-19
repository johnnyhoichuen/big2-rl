import os
import argparse

from douzero.evaluation.simulation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Big2-RL')
    # define which agents to place in which positions
    parser.add_argument('--south', type=str, default='baselines/random.ckpt')
    parser.add_argument('--east', type=str, default='baselines/random.ckpt')
    parser.add_argument('--north', type=str, default='baselines/random.ckpt')
    parser.add_argument('--west', type=str, default='baselines/random.ckpt')

    parser.add_argument('--eval_data', type=str, default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='0')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.eval_data,
             args.num_workers)
