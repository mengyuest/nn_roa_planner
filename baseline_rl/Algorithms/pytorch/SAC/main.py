#!/usr/bin/env python
# Created at 2020/3/25
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
    parser.add_argument("--lr_p", type=float, default=1e-3, help="Learning rate for Policy Net")
    parser.add_argument("--lr_v", type=float, default=1e-3, help="Learning rate for Value Net")
    parser.add_argument("--lr_q", type=float, default=1e-3, help="Learning rate for QValue Net")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--polyak", type=float, default=0.995,
                  help="Interpolation factor in polyak averaging for target networks")
    parser.add_argument("--explore_size", type=int, default=10000, help="Explore steps before execute deterministic policy")
    parser.add_argument("--memory_size", type=int, default=1000000, help="Size of replay memory")
    parser.add_argument("--step_per_iter", type=int, default=1000, help="Number of steps of interaction in each iteration")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--min_update_step", type=int, default=1000, help="Minimum interacts for updating")
    parser.add_argument("--update_step", type=int, default=50, help="Steps between updating policy and critic")
    parser.add_argument("--max_iter", type=int, default=500, help="Maximum iterations to run")
    parser.add_argument("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations to save the model")
    parser.add_argument("--target_update_delay", type=int, default=1, help="Frequency for target Value Net update")
    parser.add_argument("--model_path", type=str, default="trained_models", help="Directory to store model")
    parser.add_argument("--log_path", type=str, default="../log/", help="Directory to save logs")

    parser.add_argument('--nn_sim', action='store_true', default=False)
    # parser.add_argument('--pretrained_path', action='store_true', default=None)

    # TODO(yue)
    parser = global_patch.add_parser(parser)
    return parser


def main(rest_args=None):  # TODO(yue)
    # TODO(yue)
    parser = define_parser()
    args, log_path, model_path, base_dir, writer = global_patch.setup_dir(parser, rest_args)

    from Algorithms.pytorch.SAC.sac import SAC
    sac = SAC(args.env_id,
              render=args.render,
              num_process=args.num_process,
              memory_size=args.memory_size,
              lr_p=args.lr_p,
              lr_v=args.lr_v,
              lr_q=args.lr_q,
              gamma=args.gamma,
              polyak=args.polyak,
              explore_size=args.explore_size,
              step_per_iter=args.step_per_iter,
              batch_size=args.batch_size,
              min_update_step=args.min_update_step,
              update_step=args.update_step,
              target_update_delay=args.target_update_delay,
              seed=args.seed,
              args=args)  # TODO(yue)
    if args.pretrained_path is not None:
        offset = int(args.pretrained_path.split("/model_")[1].split(".p")[0])
        print("offset", offset)
        # offset = 0
    else:
        offset = 0
    for i_iter in range(1, args.max_iter + 1):
        sac.learn(writer, i_iter, offset=offset)

        if i_iter==1 or i_iter % 10 == 0:
            if hasattr(sac.env, "print_stat"):
                sac.env.print_stat()

        if i_iter % args.eval_iter == 0:
            sac.eval(i_iter, render=args.render)

        if i_iter % args.save_iter == 0:
            sac.save(model_path)
            global_patch.save(sac, model_path, i_iter + offset)  # TODO(yue)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
