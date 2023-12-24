#!/usr/bin/env python
# Created at 2020/3/1
import click
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
    parser.add_argument("--env_id", type=str, default="BipedalWalker-v3", help="Environment Id")
    parser.add_argument("--render", type=bool, default=False, help="Render environment or not")
    parser.add_argument("--num_process", type=int, default=1, help="Number of process to run environment")
    parser.add_argument("--lr_p", type=float, default=1e-3, help="Learning rate for Policy Net")
    parser.add_argument("--lr_v", type=float, default=1e-3, help="Learning rate for Value Net")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--polyak", type=float, default=0.995,
                  help="Interpolation factor in polyak averaging for target networks")
    parser.add_argument("--target_action_noise_std", type=float, default=0.2, help="Std for noise of target action")
    parser.add_argument("--target_action_noise_clip", type=float, default=0.5, help="Clip ratio for target action noise")
    parser.add_argument("--explore_size", type=int, default=10000, help="Explore steps before execute deterministic policy")
    parser.add_argument("--memory_size", type=int, default=1000000, help="Size of replay memory")
    parser.add_argument("--step_per_iter", type=int, default=1000, help="Number of steps of interaction in each iteration")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--min_update_step", type=int, default=1000, help="Minimum interacts for updating")
    parser.add_argument("--update_step", type=int, default=50, help="Steps between updating policy and critic")
    parser.add_argument("--max_iter", type=int, default=500, help="Maximum iterations to run")
    parser.add_argument("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations to save the model")
    parser.add_argument("--action_noise", type=float, default=0.1, help="Noise for action")
    parser.add_argument("--policy_update_delay", type=int, default=2, help="Frequency for policy update")
    parser.add_argument("--model_path", type=str, default="trained_models", help="Directory to store model")
    parser.add_argument("--log_path", type=str, default="../log/", help="Directory to save logs")

    # TODO(yue)
    parser = global_patch.add_parser(parser)
    return parser


def main(rest_args=None):
    # TODO(yue)
    parser = define_parser()
    args, log_path, model_path, base_dir, writer = global_patch.setup_dir(parser, rest_args)

    from Algorithms.pytorch.TD3.td3 import TD3
    td3 = TD3(args.env_id,
              render=args.render,
              num_process=args.num_process,
              memory_size=args.memory_size,
              lr_p=args.lr_p,
              lr_v=args.lr_v,
              gamma=args.gamma,
              polyak=args.polyak,
              target_action_noise_std=args.target_action_noise_std,
              target_action_noise_clip=args.target_action_noise_clip,
              explore_size=args.explore_size,
              step_per_iter=args.step_per_iter,
              batch_size=args.batch_size,
              min_update_step=args.min_update_step,
              update_step=args.update_step,
              action_noise=args.action_noise,
              policy_update_delay=args.policy_update_delay,
              seed=args.seed,
              args=args)

    for i_iter in range(1, args.max_iter + 1):
        td3.learn(writer, i_iter)

        if i_iter % args.eval_iter == 0:
            td3.eval(i_iter, render=args.render)

        if i_iter % args.save_iter == 0:
            td3.save(model_path)
            global_patch.save(td3, model_path, i_iter)  # TODO(yue)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
