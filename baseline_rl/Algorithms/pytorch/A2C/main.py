#!/usr/bin/env python
# Created at 2020/3/31

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
    parser.add_argument("--num_process", type=int, default=1, help="Number of process to run environment")
    parser.add_argument("--lr_ac", type=float, default=3e-4, help="Learning rate for Actor-Critic Net")
    parser.add_argument("--value_net_coeff", type=float, default=0.5, help="Coefficient for value loss")
    parser.add_argument("--entropy_coeff", type=float, default=1e-2, help="Coefficient for entropy loss")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.95, help="GAE factor")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations to run")
    parser.add_argument("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations to save the model")
    parser.add_argument("--model_path", type=str, default="trained_models", help="Directory to store model")
    parser.add_argument("--log_path", type=str, default="../log/", help="Directory to save logs")

    # TODO(yue)
    parser = global_patch.add_parser(parser)
    return parser


def main(rest_args=None):
    # TODO(yue)
    parser = define_parser()
    args, log_path, model_path, base_dir, writer = global_patch.setup_dir(parser, rest_args)

    from Algorithms.pytorch.A2C.a2c import A2C
    a2c = A2C(env_id=args.env_id,
              render=args.render,
              num_process=args.num_process,
              min_batch_size=args.batch_size,
              lr_ac=args.lr_ac,
              value_net_coeff=args.value_net_coeff,
              entropy_coeff=args.entropy_coeff,
              gamma=args.gamma,
              tau=args.tau,
              seed=args.seed,
              args=args)  # TODO(yue)

    for i_iter in range(1, args.max_iter + 1):
        a2c.learn(writer, i_iter)

        if i_iter % args.eval_iter == 0:
            a2c.eval(i_iter, render=args.render)

        if i_iter % args.save_iter == 0:
            a2c.save(model_path)
            global_patch.save(a2c, model_path, i_iter)  # TODO(yue)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
