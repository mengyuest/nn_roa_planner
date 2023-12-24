#!/usr/bin/env python
# Created at 2020/2/9

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
parser = argparse.ArgumentParser()
# @click.command()
parser.add_argument("--env_id", type=str, default="MountainCar-v0", help="Environment Id")
parser.add_argument("--render", type=bool, default=False, help="Render environment or not")
parser.add_argument("--num_process", type=int, default=1, help="Number of process to run environment")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Policy Net")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--polyak", type=float, default=0.995,
              help="Interpolation factor in polyak averaging for target networks")
parser.add_argument("--epsilon", type=float, default=0.90, help="Probability controls greedy action")
parser.add_argument("--explore_size", type=int, default=5000, help="Explore steps before execute deterministic policy")
parser.add_argument("--memory_size", type=int, default=100000, help="Size of replay memory")
parser.add_argument("--step_per_iter", type=int, default=1000, help="Number of steps of interaction in each iteration")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--min_update_step", type=int, default=1000, help="Minimum interacts for updating")
parser.add_argument("--update_target_gap", type=int, default=50, help="Steps between updating target q net")
parser.add_argument("--max_iter", type=int, default=500, help="Maximum iterations to run")
parser.add_argument("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
parser.add_argument("--save_iter", type=int, default=50, help="Iterations to save the model")
parser.add_argument("--model_path", type=str, default="trained_models", help="Directory to store model")
parser.add_argument("--log_path", type=str, default="../log/", help="Directory to save logs")

# TODO(yue)
parser = global_patch.add_parser(parser)

def main():
    # TODO(yue)
    args, log_path, model_path, base_dir, writer = global_patch.setup_dir(parser)

    from Algorithms.pytorch.DoubleDQN.double_dqn import DoubleDQN
    ddqn = DoubleDQN(args.env_id,
                     render=args.render,
                     num_process=args.num_process,
                     memory_size=args.memory_size,
                     lr_q=args.lr,
                     gamma=args.gamma,
                     polyak=args.polyak,
                     epsilon=args.epsilon,
                     explore_size=args.explore_size,
                     step_per_iter=args.step_per_iter,
                     batch_size=args.batch_size,
                     min_update_step=args.min_update_step,
                     update_target_gap=args.update_target_gap,
                     seed=args.seed,args=args)

    for i_iter in range(1, args.max_iter + 1):
        ddqn.learn(writer, i_iter)

        if i_iter % args.eval_iter == 0:
            ddqn.eval(i_iter, render=args.render)

        if i_iter % args.save_iter == 0:
            ddqn.save(model_path)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
