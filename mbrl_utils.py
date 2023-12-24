import os
import sys
import mbrl.planning
from mbrl.third_party.pytorch_sac_pranz24.sac import SAC

class MockArgs:
    pass

for sys_pr in ["./", "../","../../"]:
    sys.path.append(sys_pr)
    sys.path.append(sys_pr + "pogo")
    sys.path.append(sys_pr + "car")
    sys.path.append(sys_pr + "walker")

from car_env import CarEnv
from beta_env import BetaEnv
from cgw_env import CgwEnv

def get_pets_models(cfg):
    return


def get_mbpo_models(mbpo_path, args):
    env_args = MockArgs()
    if "car" in mbpo_path:
        env = CarEnv(args=env_args)
    elif "beta" in mbpo_path:
        env_args.nn_sim = True
        env_args.gpus = args.gpus
        env = BetaEnv(args=env_args)
    elif "_bp_" in mbpo_path:
        env_args.num_samples = 1
        env_args.dx_mins = [-0.03, -0.06, -0.25, -0.5]
        env_args.dx_maxs = [0.03, 0.06, 0.25, 0.5]
        env_args.dcfg_min = -0.05
        env_args.dcfg_max = 0.05
        env_args.num_sim_steps = 2
        env_args.dt = 0.01
        env_args.switch_bonus = 100
        env_args.invalid_cost = 100        
        env = CgwEnv(args=env_args)


    num_inputs = env.observation_space.shape[0]
    action_space = env.action_space

    sac_args = MockArgs()
    sac_args.gamma = 0.99
    sac_args.tau = 0.005
    sac_args.alpha = 0.2
    sac_args.policy = "Gaussian"
    sac_args.target_update_interval = 4
    sac_args.automatic_entropy_tuning = True
    sac_args.target_entropy = -0.05
    sac_args.hidden_size = 512
    sac_args.lr = 0.0003
    sac_args.batch_size = 256
    sac_args.device="cuda:%s"%(args.gpus)

    agent = SAC(num_inputs, action_space, sac_args)
    found = False
    for poss_dir_level in [".", "..", "../.."]:
        poss_path = os.path.join(poss_dir_level, "exps_cyclf", mbpo_path)
        if os.path.exists(poss_path):
            agent.load_checkpoint(ckpt_path=poss_path+"/sac.pth", evaluate=True, map_location=sac_args.device)
            found = True
    if not found:
        raise NotImplementedError
    
    # usage
    '''
    action = agent.act(agent_obs, sample=sac_samples_action, batched=True)
    '''

    return agent