#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/3 下午4:40
import pickle

# import click
import torch
from torch.utils.tensorboard import SummaryWriter


# TODO(yue)
import global_patch
import sys
sys.path.append("../../../..")
import roa_utils
import argparse
# TODO(yue)
def define_parser():
    parser = argparse.ArgumentParser()
    # @click.command()
    parser.add_argument("--env_id", type=str, default="BipedalWalker-v3", help="Environment Id")
    parser.add_argument("--render", type=bool, default=False, help="Render environment or not")
    parser.add_argument("--num_process", type=int, default=4, help="Number of process to run environment")
    parser.add_argument("--lr_p", type=float, default=3e-4, help="Learning rate for Policy Net")
    parser.add_argument("--lr_v", type=float, default=3e-4, help="Learning rate for Value Net")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.95, help="GAE factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clip rate for PPO")
    parser.add_argument("--batch_size", type=int, default=4000, help="Batch size")
    parser.add_argument("--ppo_mini_batch_size", type=int, default=500,
                  help="PPO mini-batch size (default 0 -> don't use mini-batch update)")
    parser.add_argument("--ppo_epochs", type=int, default=10, help="PPO step")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations to run")
    parser.add_argument("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations to save the model")
    parser.add_argument("--model_path", type=str, default="trained_models", help="Directory to store model")
    parser.add_argument("--log_path", type=str, default="../log/", help="Directory to save logs")

    parser.add_argument('--nn_sim', action='store_true', default=False)

    # TODO(yue)
    parser = global_patch.add_parser(parser)
    return parser

def main(rest_args=None):
    # TODO(yue)
    parser = define_parser()
    args, log_path, model_path, base_dir, writer = global_patch.setup_dir(parser, rest_args)

    from Algorithms.pytorch.PPO.ppo import PPO
    ppo = PPO(env_id=args.env_id,
              render=args.render,
              num_process=args.num_process,
              min_batch_size=args.batch_size,
              lr_p=args.lr_p,
              lr_v=args.lr_v,
              gamma=args.gamma,
              tau=args.tau,
              clip_epsilon=args.epsilon,
              ppo_epochs=args.ppo_epochs,
              ppo_mini_batch_size=args.ppo_mini_batch_size,
              seed=args.seed,
              args=args)  # TODO(yue)

    for i_iter in range(1, args.max_iter + 1):
        ppo.learn(writer, i_iter)
        if i_iter==1 or i_iter % 10 == 0:
            if hasattr(ppo.env, "print_stat"):
                ppo.env.print_stat()
        if i_iter % args.eval_iter == 0:
            ppo.eval(i_iter, render=args.render)

        if i_iter % args.save_iter == 0:
            ppo.save(model_path)

            pickle.dump(ppo,
                        open('{}/{}_ppo.p'.format(model_path, args.env_id), 'wb'))
            global_patch.save(ppo, model_path, i_iter)  # TODO(yue)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
