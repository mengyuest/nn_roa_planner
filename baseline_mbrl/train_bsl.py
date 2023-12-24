import hydra
import numpy as np
import omegaconf
import torch
from datetime import datetime

import os
import sys
import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet
import mbrl.util.env
import gym
from omegaconf import OmegaConf

for sys_pr in ["./","../","../"]:
    sys.path.append(sys_pr)
    sys.path.append(sys_pr + "pogo")
    sys.path.append(sys_pr + "car")
    sys.path.append(sys_pr + "walker")

from car_env import car_term_fn, CarEnv
from beta_env import beta_term_fn, BetaEnv
from cgw_env import cgw_term_fn, CgwEnv

class MockArgs:
    pass

def proc_env(env, cfg):
    env = gym.wrappers.TimeLimit(
            env, max_episode_steps=cfg.overrides.trial_length
        )
    if cfg.seed is not None:
        env.seed(cfg.seed)
        env.observation_space.seed(cfg.seed + 1)
        env.action_space.seed(cfg.seed + 2)
    return env

class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        self.is_created=False
        self.buffer = []

    def create_log(self, log_path):
        self.log = open(log_path + "/log.txt", "a", 1)
        self.is_created = True
        for message in self.buffer:
            self.log.write(message)

    def write(self, message, only_file=False):
        if not only_file:
            self._terminal.write(message)
        if self.is_created:
            self.log.write(message)
        else:
            self.buffer.append(message)

    def flush(self):
        pass

def get_env(cfg):
    # setup environment
    args = MockArgs()
    args.print_stat=True
    if cfg.overrides.env == "car":
        env = CarEnv(args=args)
        test_env = CarEnv(args=args)
        term_fn = car_term_fn
    elif cfg.overrides.env == "beta":
        args.nn_sim = True
        args.gpus = cfg.device.split(":")[1]
        env = BetaEnv(args=args)
        test_env = BetaEnv(args=args)
        term_fn = beta_term_fn
    elif cfg.overrides.env == "bp":
        args.num_samples = 1
        args.dx_mins = [-0.03, -0.06, -0.25, -0.5]
        args.dx_maxs = [0.03, 0.06, 0.25, 0.5]
        args.dcfg_min = -0.05
        args.dcfg_max = 0.05
        args.num_sim_steps = 2
        args.dt = 0.01
        args.switch_bonus = 100
        args.invalid_cost = 100        
        env = CgwEnv(args=args)
        test_env = CgwEnv(args=args)
        term_fn = cgw_term_fn
    return env, test_env, term_fn

@hydra.main(config_path="mbrl/examples/conf", config_name="main")
def main(cfg: omegaconf.DictConfig):
    full_exp_dir = os.getcwd()
    print(full_exp_dir)
    # stdout logger
    logger = Logger()
    logger.create_log(full_exp_dir)
    sys.stdout = logger
    logger.write("python " + " ".join(sys.argv) + "\n", only_file=True)

    # random seed
    print("Set random seed to %d"%(cfg.seed))
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    log_dir_bak = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())

    env, test_env, term_fn = get_env(cfg)

    # set random seed
    env = proc_env(env, cfg)
    test_env = proc_env(test_env, cfg)
    reward_fn = None
    
    os.chdir(log_dir_bak)
    # PETS: Deep reinforcement learning in a handful of trials using probabilistic dynamics models
    # MBPO: Model-based policy optimization
    # PlaNet: Learning latent dynamics for planning from pixels
    if cfg.algorithm.name == "pets":
        pets.train(env, term_fn, reward_fn, cfg)
    elif cfg.algorithm.name == "mbpo":
        mbpo.train(env, test_env, term_fn, cfg)


if __name__ == "__main__":
    main()