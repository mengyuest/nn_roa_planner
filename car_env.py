import gym
import sys

from pathlib import Path
path = Path(__file__).parent.absolute()

gym.logger.set_level(40)
import numpy as np
import torch
import matplotlib.pyplot as plt

for sys_pr in ["../","../../","../../../","../../../../"]:
    sys.path.append(sys_pr)
    sys.path.append(sys_pr + "pogo")
    sys.path.append(sys_pr + "car")
    sys.path.append(sys_pr + "walker")

import car_utils as sscar_utils
import car_clf_train as za



class Meter():
    def __init__(self):
        self.n = 0
        self.avg = 0
        self.val = 0

    def update(self, x):
        self.val = x
        self.n += 1
        self.avg = (self.avg * (self.n-1) + self.val)/self.n


'''TODO
python roa_planner.py --exp_name PLAN_nnSam_ring_REDO --gpus 0 --dt 0.01 --nt 100 --cutoff_norm 0.5 
--actor_nn_res_ratio 0.3 --mu_choices 1.0 0.1 --ref_mins 3.0 -0.0 --ref_maxs 30.0 0.0 --num_samples 10000 
--clf_hiddens 256 256 --joint_pretrained_path g0128-111623_JOI_ROA_U12_SA_grow --actor_hiddens 256 256 
--u_limit 12 12 --net_hiddens 256 256 --net_pretrained_path g0128-125643_EST_grow --plan_setups 0.01 500 10 2000 2000 
--allow_factor 5.0 --complex --cap_seg 16 --num_workers 1 --skip_viz 10 --simple_plan --accumulate_traj --use_sample 
--no_lqr --lqr_nt 140 --clf_nt 120 --use_middle
'''

def car_term_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    # out_of_lane = np.abs(self.state[0, 1]) > 3
    # rmse = 1 * np.linalg.norm(self.state[:, :7]).item()
    # valid = 1 * (not out_of_lane)
    # reward = -rmse + valid
    out_of_lane = torch.abs(next_obs[:, 1]) > 3
    huge_rmse = torch.norm(next_obs[:, :7], dim=1) > 1000
    done = torch.logical_or(out_of_lane, huge_rmse)
    done = done[:, None]
    return done


class CarEnv(gym.Env):
    def __init__(self, **kwargs):
        super(CarEnv, self).__init__()
        self.args = kwargs["args"]

        # theta, accel
        self.action_space = gym.spaces.Box(
            low=np.array([-12, -12]), high=np.array([12, 12]), dtype=np.float32)

        #
        # x_err + ref + mode + angle1 + dist + angle2 + next_ref + next_mode
        self.err_mins=[-1.5] * 7
        self.err_maxs=[1.5] * 7
        self.ref_mins = [3, 0]
        self.ref_maxs = [30, 0]
        self.m_mins = [0]
        self.m_maxs = [1]
        self.angle1_mins = [-np.pi]
        self.angle1_maxs = [np.pi]
        self.dist_mins = [5]
        self.dist_maxs = [20]
        self.angle2_mins = [-np.pi]
        self.angle2_maxs = [np.pi]
        self.next_ref_mins = [3, 0]
        self.next_ref_maxs = [30, 0]
        self.next_m_mins = [0]
        self.next_m_maxs = [1]

        self.observation_space = gym.spaces.Box(
            low=np.array(self.err_mins + self.ref_mins + self.m_mins + self.angle1_mins +
                         self.dist_mins + self.angle2_mins + self.next_ref_mins + self.next_m_mins),
            high=np.array(self.err_maxs + self.ref_maxs + self.m_maxs + self.angle1_maxs +
                          self.dist_maxs + self.angle2_maxs + self.next_ref_maxs + self.next_m_maxs),
            dtype=np.float32)
        # TODO(DEBUG)
        self.args.nt = 500  # TODO 500
        self.args.dt = 0.01
        self.stat = {"rmse": Meter(), "valid": Meter()}
        self.car_params = sscar_utils.VehicleParameters()
        self.params = None  # (mu, vref, wref)
        self.args.clip_angle = True
        self.global_id = 0
        self.reset_id = 0


    def action_space_sample(self):
        return np.random.uniform(low=self.action_space.low, high=self.action_space.high, size=self.action_space.low.shape)

    def step(self, u):
        x = to_torch(self.state[:, :7])
        u = to_torch(u.reshape((1, 2)))

        val_f = sscar_utils._f(x, self.car_params, self.params)
        val_g = sscar_utils._g(x, self.car_params, self.params)
        x_dot = val_f + torch.bmm(val_g, u.unsqueeze(-1))
        x_dot = x_dot.squeeze(-1)
        new_x = x + x_dot * self.args.dt

        ref = to_torch(self.ref_state)
        new_ref = torch.stack([
            ref[:, 0] + ref[:, 3] * torch.cos(ref[:, 2]) * self.args.dt,
            ref[:, 1] + ref[:, 3] * torch.sin(ref[:, 2]) * self.args.dt,
            ref[:, 2],
            ref[:, 3]
        ], dim=-1)
        # print(x)

        if self.args.clip_angle:
            new_x = torch.cat([
                new_x[:, 0:2],
                to_pi(new_x[:, 2:3]),
                new_x[:, 3:4],
                to_pi(new_x[:, 4:5]),
                new_x[:, 5:6],
                to_pi(new_x[:, 6:7]),
            ], dim=-1)

        # TODO(yue)
        new_x_world = err_to_world(new_x, new_ref)
        self.new_x_world = to_np(new_x_world)

        if self.seg_i == 0:
            to_dist = dist_to_seg(to_np(new_x_world[0, :2]), self.knot1[0], self.knot2[0], self.knot3[0])
        elif self.seg_i == 1:
            to_dist = dist_to_seg(to_np(new_x_world[0, :2]), self.knot2[0], self.knot3[0], self.knot4[0])


        # x_err + ref + mode + angle1 + dist + angle2 + next_ref + next_mode
        self.ref_state = to_np(new_ref)
        if to_dist < 0:  # TODO change from one seg to another seg
            # print("switch from %d to %d at t=%3d"%(self.seg_i, self.seg_i+1, self.tidx))
            if self.seg_i == 0:
                self.ref_state = self.ref2
                self.params = to_torch(np.array([[self.state[0, 15], self.state[0, 13], self.state[0, 14], ]]))
            elif self.seg_i == 1:
                self.ref_state = self.ref3
                self.params = to_torch(np.array([[self.state[0, 20], self.state[0, 18], self.state[0, 19], ]]))

            new_x_err = world_to_err(new_x_world, to_torch(self.ref_state))
            self.state[0, :7] = to_np(new_x_err)
            self.seg_i += 1
        else:
            self.state[0, :7] = to_np(new_x)

        obs = self.get_obs(self.state, to_dist)
        out_of_lane = np.abs(self.state[0, 1]) > 3
        rmse = 1 * np.linalg.norm(self.state[:, :7]).item()
        valid = 1 * (not out_of_lane)
        reward = -rmse + valid

        # # # TODO (DEBUG)
        # if self.reset_id % 1 == 0:
        #     self.render()
        done = (self.tidx >= self.args.nt or self.seg_i == 2 or out_of_lane or rmse > 1000)
        if done and hasattr(self.args, "print_stat") and self.args.print_stat==True:
            if self.reset_id % 10 == 0:
                self.print_stat()
        self.tidx += 1
        self.global_id += 1
        self.stat["rmse"].update(rmse)
        self.stat["valid"].update(valid)
        return obs[0], reward, done, \
               {"rmse": rmse, "valid": valid}


    def print_stat(self):
        print("rmse:%.3f (%.3f) valid:%.3f (%.3f) "%(
            self.stat["rmse"].val, self.stat["rmse"].avg,
            self.stat["valid"].val, self.stat["valid"].avg,
        ))


    def init_state(self):
        x_err = np.random.uniform(self.err_mins, self.err_maxs)
        ref = np.random.uniform(self.ref_mins, self.ref_maxs)
        mode = np.random.choice([0.1, 1.0], (1,))
        angle1 = np.random.uniform(self.angle1_mins, self.angle1_maxs)
        dist = np.random.uniform(self.dist_mins, self.dist_maxs)
        angle2 = to_pi_pi(angle1 + np.random.uniform(self.angle2_mins, self.angle2_maxs)/4)
        next_ref = np.random.uniform(self.next_ref_mins, self.next_ref_maxs)
        next_mode = np.random.choice([0.1, 1.0], (1,))

        dist3 = np.random.uniform(self.dist_mins, self.dist_maxs)
        angle3 = to_pi_pi(angle2 + np.random.uniform(self.angle2_mins, self.angle2_maxs)/4)

        next_ref3 = np.random.uniform(self.next_ref_mins, self.next_ref_maxs)
        next_mode3 = np.random.choice([0.1, 1.0], (1,))

        state = np.concatenate((x_err, ref, mode, angle1,
                                dist, angle2, next_ref, next_mode,
                                dist3, angle3, next_ref3, next_mode3), axis=-1)

        self.state = state.reshape((1, -1))
        self.knot1 = np.array([[0, 0]])
        self.knot2 = np.array([[dist.item() * np.cos(angle1.item()), dist.item() * np.sin(angle1.item())]])
        self.knot3 = np.array([[self.knot2[0, 0] + dist3.item() * np.cos(angle2.item()), self.knot2[0, 1] + dist3.item() * np.sin(angle2.item())]])
        self.knot4 = np.array([[self.knot3[0, 0] + np.cos(angle3.item()), self.knot3[0, 1] + np.sin(angle3.item())]])

        self.ref1 = np.array([[self.knot1[0, 0], self.knot1[0, 1], angle1.item(), ref[0]]])
        self.ref2 = np.array([[self.knot2[0, 0], self.knot2[0, 1], angle2.item(), next_ref[0]]])
        self.ref3 = np.array([[self.knot3[0, 0], self.knot3[0, 1], angle3.item(), next_ref3[0]]])
        self.ref4 = np.array([[self.knot4[0, 0], self.knot4[0, 1], angle3.item(), next_ref3[0]]])

        self.seg_i = 0

    def get_obs(self, state, to_dist):
        # x_err + ref + mode + angle1 + dist + angle2 + next_ref + next_mode
        x_err = self.state[0, :7]
        dist = np.array([to_dist])
        if self.seg_i == 0:
            ref = self.state[0, 7:9]
            mode = self.state[0, 9:10]
            angle1 = self.state[0, 10:11]
            angle2 = self.state[0, 12:13]
            next_ref = self.state[0, 13:15]
            next_mode = self.state[0, 15:16]
        elif self.seg_i == 1:
            ref = self.state[0, 13:15]
            mode = self.state[0, 15:16]
            angle1 = self.state[0, 12:13]
            angle2 = self.state[0, 17:18]
            next_ref = self.state[0, 18:20]
            next_mode = self.state[0, 20:21]
        elif self.seg_i == 2:
            ref = self.state[0, 17:18]
            mode = self.state[0, 18:20]
            angle1 = self.state[0, 20:21]
            angle2 = self.state[0, 17:18]
            next_ref = self.state[0, 18:20]
            next_mode = self.state[0, 20:21]
        obs = np.concatenate((x_err, ref, mode, angle1, dist, angle2, next_ref, next_mode), axis=-1)
        obs = obs.reshape((1, -1))
        return obs


    def reset(self, seed=None, return_info=False, options=None):
        # Reset the state of the environment to an initial state
        self.init_state()
        self.ref_state = self.ref1
        self.params = to_torch(np.array([[self.state[0, 9], self.state[0, 7], self.state[0, 8], ]]))
        to_dist = dist_to_seg(self.knot1[0], self.knot1[0], self.knot2[0], self.knot3[0])
        self.obs = self.get_obs(self.state, to_dist)

        # initialization
        self.tidx=0
        self.reset_id += 1
        if not return_info:
            return self.obs[0]
        else:
            return self.obs[0], {}

    def render(self, mode='human', close=False):

        plt.plot([self.knot1[0, 0], self.knot2[0, 0]],
                 [self.knot1[0, 1], self.knot2[0, 1]], label="seg-0", linewidth=4, color="darkgray")
        plt.plot([self.knot2[0, 0], self.knot3[0, 0]],
                 [self.knot2[0, 1], self.knot3[0, 1]], label="seg-1", linewidth=4, color="lightgray")
        plt.plot([self.knot3[0, 0], self.knot4[0, 0]],
                 [self.knot3[0, 1], self.knot4[0, 1]], label="seg-2", linewidth=4, color="gray")
        plt.scatter(self.new_x_world[0, 0], self.new_x_world[0, 1], s=32, color="red", label="ego")
        plt.scatter(self.ref_state[0, 0], self.ref_state[0, 1], s=24, color="blue", label="ref")
        plt.legend()
        plt.savefig("../viz/car/%05d_t%05d.png"%(self.reset_id, self.tidx), bbox_inches='tight', pad_inches=0.1)
        plt.close()

        do_nothing = True


def to_torch(arr):
    return torch.from_numpy(arr).float()


def to_np(tensor):
    return tensor.detach().numpy()


def to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def err_to_world(s_err, s_ref):
    x_e, y_e, delta_e, v_e, psi_e, psi_dot_e, beta_e = torch.split(s_err, 1, dim=-1)
    x_ref, y_ref, psi_ref, v_ref = torch.split(s_ref, 1, dim=-1)

    x = x_ref + x_e * torch.cos(psi_ref) - y_e * torch.sin(psi_ref)
    y = y_ref + x_e * torch.sin(psi_ref) + y_e * torch.cos(psi_ref)
    delta = 0.0 + delta_e
    v = v_ref + v_e
    psi = psi_ref + psi_e
    psi_dot = 0.0 + psi_dot_e
    beta = 0.0 + beta_e

    s_world = torch.cat([x, y, delta, v, psi, psi_dot, beta], dim=-1)
    return s_world


def world_to_err(s, s_ref):
    # x (x, y, delta, v, psi, psi_dot, beta)
    # ref (x_ref, y_ref, psi_ref, v_ref, w_ref)
    x, y, delta, v, psi, psi_dot, beta = torch.split(s, 1, dim=-1)
    x_ref, y_ref, psi_ref, v_ref = torch.split(s_ref, 1, dim=-1)
    x_e = (x - x_ref) * torch.cos(psi_ref) + (y - y_ref) * torch.sin(psi_ref)
    y_e = -(x - x_ref) * torch.sin(psi_ref) + (y - y_ref) * torch.cos(psi_ref)
    delta_e = delta - 0
    v_e = v - v_ref
    psi_e = (psi - psi_ref + np.pi) % (np.pi * 2) - np.pi
    psi_dot_e = psi_dot - 0
    beta_e = beta - 0

    s_err = torch.cat([x_e, y_e, delta_e, v_e, psi_e, psi_dot_e, beta_e], dim=-1)

    return s_err


def dist_to_seg(point, knot1, knot2, knot3, shall_print=False, use_torch=False):
    x, y = point
    x1, y1 = knot1
    x2, y2 = knot2
    x3, y3 = knot3

    if use_torch:
        arctan2 = torch.atan2
        sqrt = torch.sqrt
        sin = torch.sin
        cos = torch.cos
        sign = torch.sign
    else:
        arctan2 = np.arctan2
        sqrt = np.sqrt
        sin = np.sin
        cos = np.cos
        sign = np.sign

    xe, ye = x - x2, y - y2
    xa, ya = x1 - x2, y1 - y2
    xc, yc = x3 - x2, y3 - y2

    angle_a = arctan2(ya, xa)
    angle_c = arctan2(yc, xc)
    angle_eq = (angle_a + angle_c) / 2
    if shall_print:
        print(angle_a, angle_c, angle_eq)


    x_eq = cos(angle_eq)
    y_eq = sin(angle_eq)

    dist_a = (xa * y_eq - ya * x_eq)/sqrt(y_eq**2+x_eq**2)
    dist_c = (xc * y_eq - yc * x_eq)/sqrt(y_eq**2+x_eq**2)
    dist_x = (xe * y_eq - ye * x_eq)/sqrt(y_eq**2+x_eq**2)

    return dist_x * sign(dist_a)


def main():
    import matplotlib.pyplot as plt
    for i in range(50):
        print(i)
        knot1 = np.random.uniform([-4, -4], [4, 4])
        knot2 = np.random.uniform([-4, -4], [4, 4])
        knot3 = np.random.uniform([-4, -4], [4, 4])

        xs = np.random.uniform([-4, -4], [4, 4], (5000, 2))
        ds = []
        for j in range(xs.shape[0]):
            dist = dist_to_seg(xs[j], knot1, knot2, knot3, j==0)
            ds.append(dist)
        ds=np.array(ds)
        plt.scatter(xs[np.where(ds > 0), 0], xs[np.where(ds > 0), 1], c="blue", label="a", s=2, alpha=0.3)
        plt.scatter(xs[np.where(ds < 0), 0], xs[np.where(ds < 0), 1], c="red", label="c", s=2, alpha=0.3)
        plt.plot([knot1[0], knot2[0]], [knot1[1], knot2[1]], label="ab", color="blue", linewidth=6)
        plt.plot([knot2[0], knot3[0]], [knot2[1], knot3[1]], label="bc", color="red", linewidth=6)
        plt.legend()
        plt.savefig("viz/img_%02d.png"%(i))
        plt.close()


def to_pi_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    main()