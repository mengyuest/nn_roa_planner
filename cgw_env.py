import gym
gym.logger.set_level(40)
import numpy as np
from cgw_utils import bipedal_step, get_around_init_gait_switch
import roa_utils as utils
import os
import torch

class Meter():
    def __init__(self):
        self.n = 0
        self.avg = 0
        self.val = 0

    def update(self, x):
        self.val = x
        self.n += 1
        self.avg = (self.avg * (self.n-1) + self.val)/self.n

def cgw_term_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    x_valid = torch.logical_and(
        torch.logical_and(next_obs[:, 0]<0.5, next_obs[:, 0]>-0.5),
        torch.logical_and(next_obs[:, 1]<1, next_obs[:, 1]>-1)
    )
    done = torch.logical_not(x_valid)
    done = done[:, None] 
    return done

class CgwEnv(gym.Env):
    def __init__(self, **kwargs):
        super(CgwEnv, self).__init__()
        self.args = kwargs["args"]
        # torque
        self.action_space = gym.spaces.Box(
            low=np.array([-4,]), high=np.array([4,]), dtype=np.float32)
        # self.action_space.np_random.seed(self.args.seed)
        # q1, q2, dq1, dq2, reference gait
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.pi, -np.pi, -5, -5, 0.04]), high=np.array([np.pi, np.pi, 5, 5, 0.20]), dtype=np.float32)
        self.args.fine_switch = True
        self.args.constant_g = False
        self.args.changed_dynamics = False
        self.args.qp_bound = 4
        self.args.hjb_u_bound = 4
        self.args.reset_q1_threshold = -0.03
        self.args.nt = 200 # TODO
        self.reset_id = 0
        self.stat = {"rmse": Meter(), "valid": Meter(), "swi": Meter()}
        for i in range(10):
            the_traj_path = "./%s/walker/traj.npz"%("../"*i)
            if os.path.exists(the_traj_path):
                loaded_data = np.load(the_traj_path, allow_pickle=True)
                break
        self.ths = loaded_data['ths'][::-1] * -1.0
        self.gait_data = loaded_data['data'][::-1]


    def action_space_sample(self):
        return np.random.uniform(low=self.action_space.low, high=self.action_space.high, size=self.action_space.low.shape)


    def step(self, u):

        u = u.reshape((1, 1))

        # Execute one time step within the environment
        xf = self.xf_cache
        x_th = self.state
        x = x_th[:, :-1]
        th = x_th[:, -1:]
        x_plus, xf_plus, x_mode = bipedal_step(x, u, xf, self.args)
        x_th_next = np.concatenate((x_plus, th), axis=-1)
        q1_next = x_th_next[0, 0]
        x_valid = np.logical_and(np.logical_and(x_plus[:, 0]<0.5, x_plus[:, 0]>-0.5),
                                 np.logical_and(x_plus[:, 1]<1, x_plus[:, 1]>-1))

        rmse_cost = self.get_closest_gait_rmse_err(x_th_next[0])
        invalid_cost = (1-float(x_valid)) * self.args.invalid_cost
        swi_reward = self.args.switch_bonus * x_mode[0, 0] * (1 - np.abs(q1_next-th[0, 0])/th[0, 0])
        # print(rmse_cost, invalid_cost, swi_reward)
        reward = -rmse_cost - invalid_cost + swi_reward
        done = (x_valid[0] == False or self.tidx >= self.args.nt)

        # print(self.tidx, x_th_next[0], reward, done, x_mode, x_valid, xf_plus)

        self.tidx += 1
        self.state = x_th_next
        self.xf_cache = xf_plus

        self.stat["rmse"].update(rmse_cost)
        self.stat["valid"].update(float(x_valid))
        self.stat["swi"].update(x_mode[0, 0] * (1 - np.abs(q1_next-th[0, 0])/th[0, 0]))

        # print(self.tidx, "u", u, "obs", x_th_next[0])
        # exit()
        return x_th_next[0], reward, done, \
               {"mode": x_mode[0,0], "valid": x_valid[0], "xf": xf_plus[0,0], 'rmse': rmse_cost}

    def print_stat(self):
        print("rmse:%.3f (%.3f) valid:%.3f (%.3f) swi:%.3f (%.3f)"%(
            self.stat["rmse"].val, self.stat["rmse"].avg,
            self.stat["valid"].val, self.stat["valid"].avg,
            self.stat["swi"].val, self.stat["swi"].avg,
        ))

    def reset(self, seed=None, return_info=False, options=None):
        # Reset the state of the environment to an initial state
        # super().reset(seed=seed)

        # initialization
        x, cfg = get_around_init_gait_switch(self.args, perturbed=True)
        self.state = np.concatenate((utils.to_np(x), utils.to_np(cfg)), axis=-1)
        self.xf_cache = np.zeros((1, 1))
        self.tidx=0
        self.reset_id += 1
        if hasattr(self.args, "print_stat") and self.args.print_stat==True:
            if self.reset_id % 10 == 0:
                self.print_stat()
        if not return_info:
            return self.state[0]
        else:
            return self.state[0], {}

    def render(self, mode='human', close=False):
        do_nothing = True

    def get_closest_gait_rmse_err(self, x_th_next):
        # TODO need to check this validity

        x = x_th_next[:-1]
        q1 = x_th_next[0]
        th = x_th_next[-1]


        ref_th_idx = np.argmin(np.abs(th - self.ths))
        ref_th = self.ths[ref_th_idx]
        gait = self.gait_data[ref_th_idx]

        idx = np.argmin(np.abs(q1 - gait[:, 0]))
        ref_init = gait[idx]
        ref_x = ref_init

        return np.linalg.norm(ref_x-x)


def interp(x1, x2, a, b):
    return (x1 * b + x2 * a) / (a+b)
