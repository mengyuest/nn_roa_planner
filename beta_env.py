import gym
import os
import sys
from os.path import join as ospj

from pathlib import Path
path = Path(__file__).parent.absolute()

gym.logger.set_level(40)
import numpy as np
import torch

for sys_pr in ["../","../../","../../../","../../../../"]:
    sys.path.append(sys_pr)
    sys.path.append(sys_pr + "pogo")
    sys.path.append(sys_pr + "car")
    sys.path.append(sys_pr + "walker")


import beta_pogo_sim
import roa_utils as utils
from beta_fit import Dyna, preproc_data


def beta_term_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    h0 = next_obs[:, 9]  # the first/current segment height-base_floor 
    below_floor = next_obs[:, 2] < 0
    above_ceil = next_obs[:, 2] > h0
    done = torch.logical_or(below_floor, above_ceil)
    done = done[:, None] 
    return done

class Meter():
    def __init__(self):
        self.n = 0
        self.avg = 0
        self.val = 0

    def update(self, x):
        self.val = x
        self.n += 1
        self.avg = (self.avg * (self.n-1) + self.val)/self.n


class BetaEnv(gym.Env):
    def __init__(self, **kwargs):
        super(BetaEnv, self).__init__()
        self.args = kwargs["args"]

        # theta, torque
        self.action_space = gym.spaces.Box(
            low=np.array([-np.pi/2, -1]), high=np.array([np.pi/2, 1,]), dtype=np.float32)

        # x, xd, y, yd, xr, yr, mode, dx0, dh0, v0, dx1, dh1, v1
        # x (-max(seg) ~ max(seg))
        # xd (-50, 50)
        # y  (-50, 50)
        # yd (-50, 50)
        # xr (-50, 50)
        # yr (-50, 50)
        # mode (0, 1)
        # dx0 (-50, 50)
        # df0 (-50, 50)
        # dh0 (-50, 50)
        # v0 (-50, 50)
        # dx1 (-50, 50)
        # df1 (-50, 50)
        # dh1 (-50, 50)
        # v1 (-50, 50)
        self.observation_space = gym.spaces.Box(
            low=np.array([-50, -50, -50, -50, -50, -50, 0, -50, -50, -50, -50, -50, -50, -50, -50]),
            high=np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]), dtype=np.float32)
        self.args.nt = 50  # TODO
        self.args.nsteps = 1000  # TODO
        self.reset_id = 0
        # num_steps, dt, dt2, l0, g, k, M = self.args.nt, self.args.dt1, self.args.dt2, self.args.l0, self.args.g, self.args.k, self.args.M,
        if self.args.nn_sim:
            # load data
            data_pre_path = "g0208-214752_Pda_10M"
            dyna_pretrained_path = "g0208-222845_Pfit"

            data_path = utils.smart_path(data_pre_path) + "/data.npz"
            train_x_u, test_x_u, x_u_mean, x_u_std, train_y, test_y, y_mean, y_std = preproc_data(data_path, split_ratio=0.95)

            # load network
            smart_path = utils.smart_path(dyna_pretrained_path)
            if "models" in smart_path:
                nn_dir = smart_path.split("models")[0]
            else:
                nn_dir = smart_path
            net_args = np.load(ospj(nn_dir, "args.npz"), allow_pickle=True)['args'].item()
            net_args.in_means = x_u_mean
            net_args.in_stds = x_u_std
            net_args.out_means = y_mean
            net_args.out_stds = y_std
            net = Dyna(net_args)
            net = utils.safe_load_nn(net, dyna_pretrained_path, load_last=True, key="model_")
            net = utils.cuda(net, self.args)
            net.update_device()
            self.dyna = net

            # load dynamics network
            num_steps = 5000
            dt = 0.001
            dt2 = 0.0001
            l0 = 1.0
            g = 10.0
            k = 32000
            M = 80
        else:
            num_steps = 5000
            dt = 0.001
            dt2 = 0.0001
            l0 = 1.0
            g = 10.0
            k = 32000
            M = 80
        self.global_vars = num_steps, dt, dt2, l0, g, k, M
        self.stat={"length":Meter(), "hit":Meter(), "v_loss":Meter()}

    def get_env_3(self):
        de = 1  # 0.75
        h = 1.8
        ref_v = 0.75
        return [[5.0, 0.0, h, ref_v],
                [5.0, 0.0, h + 1.5 * de, ref_v],
                # [5.0, 0.0, h + 1 * de, ref_v],
                [3.0, 2 * de, h + 1.5 * de, ref_v * 1.5],
                # [5.0, 0.0, h + 1 * de, ref_v],
                [5.0, 0.0, h + 1.5 * de, ref_v],
                [5.0, 0.0, h, ref_v]]





    def action_space_sample(self):
        return np.random.uniform(low=self.action_space.low, high=self.action_space.high, size=self.action_space.low.shape)

    def step(self, u):
        seg_i, seg_j = self.find_seg(self.state[0, 0])

        x_obs = self.get_obs(self.state)

        u = u.reshape((1, 2))
        modes = to_torch(self.state[:, -1:])
        x_card = to_torch(self.state[:, :-1])
        u = to_torch(u)
        u[:, 0] = torch.clamp(u[:, 0], -0.5, 0.5)
        # u[:, 1] = torch.clamp(u[:, 1] * 5000, -5000, 5000)
        u[:, 1] = torch.clamp(u[:, 1], -5000, 5000)

        x_card[:, 3] -= self.env[seg_i][1]
        x_card[:, 5] -= self.env[seg_i][1]
        curr_offset = np.sum(self.env[:seg_i+1, 0])
        curr_floor = self.env[seg_i, 1]
        curr_ceil = self.env[seg_i, 2]

        xc_list = []
        low_list = []
        high_list = []
        mc_list = []

        if self.args.nn_sim:
            num_steps, dt, dt2, l0, g, k, M = self.global_vars
            nn_input = torch.cat([x_card[:, 1:3], u[:, :2]], dim=-1).cuda()
            nn_out = self.dyna(nn_input).detach().cpu()  # (1, 3) delta_x, new_dx, new_y

            next_x_card = x_card.detach().clone()
            next_x_card[:, 0] = next_x_card[:, 0] + nn_out[:, 0]  # delta_x
            next_x_card[:, 1] = nn_out[:, 1]  # new_dx
            next_x_card[:, 2] = nn_out[:, 2]  # new_y
            next_x_card[:, 3] = 0
            next_x_card[:, 4] = next_x_card[:, 0] + l0 * torch.sin(u[:, 0])
            next_x_card[:, 5] = next_x_card[:, 2] - l0 * torch.cos(u[:, 0])
            next_modes = modes
            if x_card[0, 4] < curr_offset and next_x_card[0, 4] >= curr_offset:
                next_x_card[0, 2] = next_x_card[0, 2] + self.env[seg_i][1] - self.env[seg_j][1]
                next_x_card[0, 5] = next_x_card[0, 2] + self.env[seg_i][1] - self.env[seg_j][1]
                curr_floor = self.env[seg_j, 1]
                curr_ceil = self.env[seg_j, 2]
            xc_list.append(card_to_world(next_x_card, 0, curr_floor))
            mc_list.append(next_modes)

            self.state = to_np(torch.cat((xc_list[-1], mc_list[-1]), dim=-1))

        else:
            for i in range(self.args.nt):
                next_x_card, next_modes = beta_pogo_sim.pytorch_sim_single(
                    x_card, modes, u, self.global_vars, check_invalid=True)
                # TODO handle jump switch case
                if x_card[0, 4] < curr_offset and next_x_card[0, 4] >= curr_offset:
                    next_x_card[0, 2] = next_x_card[0, 2] + self.env[seg_i][1] - self.env[seg_j][1]
                    next_x_card[0, 5] = next_x_card[0, 2] + self.env[seg_i][1] - self.env[seg_j][1]
                    curr_floor = self.env[seg_j, 1]
                    curr_ceil = self.env[seg_j, 2]
                xc_list.append(card_to_world(next_x_card, 0, curr_floor))
                mc_list.append(next_modes)
                low_list.append(curr_floor)
                high_list.append(curr_ceil)
                x_card, modes = next_x_card, next_modes
                if modes[0] == 7 or modes[0] == -1:  # comes to the apex point
                    break

            self.state = to_np(torch.cat((xc_list[-1], mc_list[-1]), dim=-1))
            if self.state[:, 6] == 7:
                self.state[:, 6] = 0

        obs = self.get_obs(self.state)
        new_seg_i, new_seg_j = self.find_seg(self.state[0, 0])
        above_ceil = obs[0, 2] > self.env[new_seg_i][2] - self.env[new_seg_i][1]
        below_floor = obs[0, 2] < 0

        if self.args.nn_sim:
            done = (self.tidx >= self.args.nsteps
                    or self.state[0, 0] >= self.total_len or self.state[0, 0] < 0
                    or above_ceil or below_floor)
        else:
            done = (next_modes[0, 0] == -1 or self.tidx >= self.args.nsteps
                    or self.state[0, 0] >= self.total_len or self.state[0, 0] < 0
                    or above_ceil or below_floor)

        len_reward = (self.state[0, 0]-self.x_init)
        vref = self.env[new_seg_i, 3]
        v_loss = 1 * np.abs(self.state[0, 1] - vref)
        hit_penalty = 10 * (above_ceil or below_floor)

        reward = len_reward - v_loss - hit_penalty
        self.tidx += 1

        if done and hasattr(self.args, "print_stat") and self.args.print_stat==True:
            if self.reset_id % 10 == 0:
                self.print_stat()

        self.stat["length"].update(len_reward)
        self.stat["v_loss"].update(v_loss)
        self.stat["hit"].update(hit_penalty)

        return obs[0], reward, done, \
               {"length": len_reward, "v_loss": v_loss, "hit": hit_penalty}

    def print_stat(self):
        print("length:%.3f (%.3f) v_loss:%.3f (%.3f) hit:%.3f (%.3f)"%(
            self.stat["length"].val, self.stat["length"].avg,
            self.stat["v_loss"].val, self.stat["v_loss"].avg,
            self.stat["hit"].val, self.stat["hit"].avg,
        ))

    def find_seg(self, x):
        offset = 0
        for seg_i, seg in enumerate(self.env):
            offset += seg[0]
            if x <= offset:
                break
        if seg_i == self.env.shape[0]-1:
            return seg_i, seg_i
        else:
            return seg_i, seg_i + 1


    def init_state(self):
        # x, xd, y, yd, xr, yr, mode
        x = np.random.uniform(0, self.total_len/4)
        xd = np.random.uniform(0, 3)
        seg_i, seg_j = self.find_seg(x)
        y = np.random.uniform(self.env[seg_i][1]+1, self.env[seg_i][2])
        yd = 0
        xr = x
        yr = y - 1
        mode = 0
        self.x_init = x
        return np.array([[x, xd, y, yd, xr, yr, mode]])


    def get_obs(self, state):
        # TODO interpretation
        # (relative state) + (relative configs in neighbor segs)
        # x, xr: to the seg_i left edge
        # y, yr: to the seg_i floor
        # f0, h0, f1, h1: to the seg_i floor
        seg_i, seg_j = self.find_seg(state[0, 0])
        base_offset = np.sum(self.env[:seg_i, 0])
        obs = get_obs_meta(state, base_offset, self.env[seg_i], self.env[seg_j])
        return obs


    def reset(self, seed=None, return_info=False, options=None):
        # Reset the state of the environment to an initial state
        self.env = get_rand_env()
        self.total_len = np.sum(self.env[:, 0])
        self.state = self.init_state()
        self.obs = self.get_obs(self.state)

        # initialization
        self.tidx=0
        self.reset_id += 1
        if not return_info:
            return self.obs[0]
        else:
            return self.obs[0], {}

    def render(self, mode='human', close=False):
        do_nothing = True


def card_to_world(x_card, offset, floor):
    return torch.stack([
        x_card[:, 0] + offset,
        x_card[:, 1],
        x_card[:, 2] + floor,
        x_card[:, 3],
        x_card[:, 4] + offset,
        x_card[:, 5] + floor,
    ], dim=-1)


def to_torch(arr):
    return torch.from_numpy(arr).float()


def to_np(tensor):
    return tensor.detach().numpy()


def get_obs_meta(state, offset, seg0, seg1, use_torch=False):
    # TODO interpretation
    # (relative state) + (relative configs in neighbor segs)
    # x, xr: to the seg_i left edge
    # y, yr: to the seg_i floor
    # f0, h0, f1, h1: to the seg_i floor
    base_offset = offset
    base_floor = seg0[1]

    # x, xd, y, yd, xr, yr, mode, dx0, dh0, v0, dx1, dh1, v1
    x = state[0, 0] - base_offset
    xd = state[0, 1]
    y = state[0, 2] - base_floor
    yd = state[0, 3]
    xr = state[0, 4] - base_offset
    yr = state[0, 5] - base_floor
    mode = state[0, 6]

    # (should be in seg_i frame)
    x0 = seg0[0]
    f0 = seg0[1] - base_floor
    h0 = seg0[2] - base_floor
    v0 = seg0[3]
    x1 = seg1[0]
    f1 = seg1[1] - base_floor
    h1 = seg1[2] - base_floor
    v1 = seg1[3]

    obs = np.array([[x, xd, y, yd, xr, yr, mode, x0, f0, h0, v0, x1, f1, h1, v1]])

    if use_torch:
        obs = torch.from_numpy(obs).float().type_as(state)

    return obs


def get_rand_env_insider():
    de_min = 0.5
    de_max = 1.0
    ref_v_min = 0.5
    ref_v_max = 1.5
    h_min = 1.5
    h_max = 2.0

    de = np.random.rand()
    de = de * (de_max - de_min) + de_min

    ref_v = np.random.rand()
    ref_v = ref_v * (ref_v_max - ref_v_min) + ref_v_min

    h = np.random.rand()
    h = h * (h_max - h_min) + h_min

    map_id = np.random.randint(0, 4)
    if map_id == 0:
        env = [[5.0, 0.0, h, ref_v],
               [5.0, 0.0, h, ref_v],
               [5.0, 0.0, h, ref_v],
               [5.0, 0.0, h, ref_v]]
    elif map_id == 1:
        env = [[5.0, 0.0, h, ref_v],
               [5.0, 0.0, h + 1.5 * de, ref_v],
               [5.0, de, h + 1.5 * de, ref_v],
               [5.0, de, h + 1.5 * de, ref_v]]
    elif map_id == 2:
        env = [[5.0, 0.0, h, ref_v],
               [5.0, -de, h, ref_v],
               [5.0, -de, h - 0.5 * de, ref_v],
               [5.0, -de, h - 0.5 * de, ref_v]]
    elif map_id == 3:
        env = [[5.0, 0.0, h, ref_v],
            [5.0, 0.0, h + 1.5 * de, ref_v],
            [3.0, 2 * de, h + 1.5 * de, ref_v * 1.5],
            [5.0, 0.0, h + 1.5 * de, ref_v],
            [5.0, 0.0, h, ref_v]]

    for i in range(len(env)):
        env[i][0] *= np.random.uniform(0.8, 1.2)
    return np.array(env)

def get_rand_env():
    seg_list = get_rand_env_insider()
    num_try=0
    while check_env(seg_list)==False:
        # print(num_try, "regenerate valid environment!")
        num_try+=1
        seg_list = get_rand_env_insider()
    return seg_list

def check_env(seg_list):
    bloat_factor = 0.1
    for seg_i in range(seg_list.shape[0]):
        floor0, ceil0 = seg_list[seg_i][1], seg_list[seg_i][2]
        if seg_i == seg_list.shape[0] - 1:
            floor1, ceil1 = seg_list[seg_i][1], seg_list[seg_i][2]
        else:
            floor1, ceil1 = seg_list[seg_i+1][1], seg_list[seg_i+1][2]
        mid_y_abs_min = max(floor0, floor1) + 1.0 + bloat_factor
        mid_y_abs_max = min(ceil0, ceil1) - bloat_factor
        if mid_y_abs_min >= mid_y_abs_max:
            return False
    return True