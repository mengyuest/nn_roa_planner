#!/usr/bin/env python
# Created at 2020/2/15
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
    parser.add_argument("--lr_p", type=float, default=3e-4, help="Learning rate for Policy Net")
    parser.add_argument("--lr_v", type=float, default=1e-3, help="Learning rate for Value Net")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.95, help="GAE factor")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--vpg_epochs", type=int, default=10, help="Vanilla PG step")
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

    from Algorithms.pytorch.VPG.vpg import VPG
    vpg = VPG(args.env_id, args.render, args.num_process,
              args.batch_size, args.lr_p, args.lr_v, args.gamma, args.tau,
              args.vpg_epochs, seed=args.seed, args=args)

    for i_iter in range(1, args.max_iter + 1):
        vpg.learn(writer, i_iter)

        if i_iter % args.eval_iter == 0:
            vpg.eval(i_iter, render=args.render)

        if i_iter % args.save_iter == 0 or i_iter==1:
            vpg.save(model_path)
            global_patch.save(vpg, model_path, i_iter)  # TODO(yue)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
