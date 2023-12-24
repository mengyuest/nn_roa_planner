import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon, Rectangle, Ellipse
import pickle

import car_clf_train as za
import car_utils as sscar_utils
import car_mpc
from os.path import join as ospj

sys.path.append("../")
import roa_nn
import roa_utils as utils
import car_env

import mbrl_utils

class MockArgs(object):
    pass


def handle(x):
    y = x.detach().cuda()
    y.requires_grad = True
    return y


def world_to_err(s, s_ref):
    # x (x, y, delta, v, psi, psi_dot, beta)
    # ref (x_ref, y_ref, psi_ref, v_ref, w_ref)
    x, y, delta, v, psi, psi_dot, beta = torch.split(s, 1, dim=-1)
    x_ref, y_ref, psi_ref, v_ref, w_ref = torch.split(s_ref, 1, dim=-1)
    x_e = (x - x_ref) * torch.cos(psi_ref) + (y - y_ref) * torch.sin(psi_ref)
    y_e = -(x - x_ref) * torch.sin(psi_ref) + (y - y_ref) * torch.cos(psi_ref)
    delta_e = delta - 0
    v_e = v - v_ref
    psi_e = (psi - psi_ref + np.pi) % (np.pi * 2) - np.pi
    psi_dot_e = psi_dot - w_ref
    beta_e = beta - 0

    s_err = torch.cat([x_e, y_e, delta_e, v_e, psi_e, psi_dot_e, beta_e], dim=-1)

    return s_err


def err_to_world(s_err, s_ref):
    x_e, y_e, delta_e, v_e, psi_e, psi_dot_e, beta_e = torch.split(s_err, 1, dim=-1)
    x_ref, y_ref, psi_ref, v_ref, w_ref = torch.split(s_ref, 1, dim=-1)

    x = x_ref + x_e * torch.cos(psi_ref) - y_e * torch.sin(psi_ref)
    y = y_ref + x_e * torch.sin(psi_ref) + y_e * torch.cos(psi_ref)
    delta = 0.0 + delta_e
    v = v_ref + v_e
    psi = psi_ref + psi_e
    psi_dot = w_ref + psi_dot_e
    beta = 0.0 + beta_e

    s_world = torch.cat([x, y, delta, v, psi, psi_dot, beta], dim=-1)
    return s_world


def get_ds_ref(s_ref):
    x_ref, y_ref, psi_ref, v_ref, w_ref = torch.split(s_ref, 1, dim=-1)
    return torch.cat([
        v_ref * torch.cos(psi_ref),
        v_ref * torch.sin(psi_ref),
        w_ref,
        0 * v_ref,
        0 * w_ref,
    ], dim=-1)


def get_ref_list(s_init_ref, nt, dt):
    ref_list = [s_init_ref]
    for ti in range(nt):
        s_ref = ref_list[-1].clone()
        ds_ref = get_ds_ref(s_ref)
        new_s_ref = s_ref + ds_ref * dt
        ref_list.append(new_s_ref)
    return torch.stack(ref_list, dim=0)


def err_to_world_list(err_list, ref_list):
    tlen = err_list.shape[0]
    world_list = []
    for ti in range(tlen):
        s_world = err_to_world(err_list[ti], ref_list[ti])
        world_list.append(s_world)
    return torch.stack(world_list, dim=0)


def to_np(x):
    return x.detach().cpu().numpy()


def to_item(x):
    return x.detach().cpu().item()


def to_cuda(x, args):
    if args.gpus is not None:
        return x.cuda()

def to_tensor(x):
    return torch.from_numpy(x).float()


def plan_a_b_sample(curr_s, ref_a, theta_a, mu_a, ref_b, theta_b, mu_b, actor, clf, net,
                    args, is_mid=False, mid_ref=None, mid_mu=None, mid_theta=None, only_first_half=False, diff_planner=False):
    dbg_t1 = time.time()
    n_points = 20
    n_speeds = 10
    n_iters = 5
    lr = 0.05

    min_time = (args.nt - (args.clf_nt-args.nt)) * args.dt
    max_time = args.clf_nt * args.dt if args.clf_nt is not None else 2 * args.nt * args.dt
    dbg_t2 = time.time()
    if args.use_middle:
        half_road_length_a = (((mid_ref[1] - ref_a[1]) ** 2 + (mid_ref[0] - ref_a[0]) ** 2) ** 0.5)
        half_road_length_b = (((ref_b[1] - mid_ref[1]) ** 2 + (ref_b[0] - mid_ref[0]) ** 2) ** 0.5)
        min_v = max(half_road_length_a.detach().item() / max_time, 5)
        max_v = min(half_road_length_b.detach().item() / min_time, 15)
    else:
        half_road_length = (((ref_b[1] - ref_a[1]) ** 2 + (ref_b[0] - ref_a[0]) ** 2) ** 0.5) / 2
        half_road_length = half_road_length.detach().item()
        min_v = max(half_road_length / max_time, 5)
        max_v = min(half_road_length / min_time, 15)
    dbg_t3 = time.time()

    if args.use_middle:
        mid_point_x = mid_ref[0]
        mid_point_y = mid_ref[1]
    else:
        mid_point_x = (ref_a[0] + ref_b[0]) / 2
        mid_point_y = (ref_a[1] + ref_b[1]) / 2
    norm_vec = torch.atan2(ref_b[1] - ref_a[1], ref_b[0] - ref_a[0]) + np.pi / 2
    norm_vec = norm_vec.detach().item()

    dbg_ttt0 = time.time()
    # TODO Start differentiable things
    if diff_planner:
        v1_ref_cands = torch.linspace(min_v, max_v, n_speeds).type_as(mid_ref).cuda().requires_grad_()
        v2_ref_cands = torch.linspace(min_v, max_v, n_speeds).type_as(mid_ref).cuda().requires_grad_()
        norm_len_cands = torch.linspace(-1.0, 1.0, n_points).type_as(mid_ref).cuda().requires_grad_()

        optimizer = torch.optim.RMSprop([v1_ref_cands, v2_ref_cands, norm_len_cands], lr=lr)
        for sgd_i in range(n_iters):
            len_v1_v2 = torch.meshgrid(norm_len_cands, v1_ref_cands, v2_ref_cands)
            len_v1_v2 = torch.stack(len_v1_v2, dim=-1).reshape((-1, 3))
            norm_len, v1_ref, v2_ref = len_v1_v2[:, 0], len_v1_v2[:, 1], len_v1_v2[:, 2]
            ones = torch.ones_like(v1_ref)
            zeros = torch.zeros_like(v1_ref)
            mid_point_x = mid_point_x.cuda()
            mid_point_y = mid_point_y.cuda()
            mid_cfgs = torch.stack([mid_point_x + norm_len * np.cos(norm_vec), mid_point_y + norm_len * np.sin(norm_vec)],
                                   dim=-1)
            theta1 = torch.atan2(mid_cfgs[:, 1] - ref_a[1], mid_cfgs[:, 0] - ref_a[0])
            theta2 = torch.atan2(ref_b[1] - mid_cfgs[:, 1], ref_b[0] - mid_cfgs[:, 0])
            x1_init = torch.stack([ref_a[0] * ones, ref_a[1] * ones, theta1, v1_ref, zeros], dim=-1)
            x1_term = torch.stack([mid_cfgs[:, 0], mid_cfgs[:, 1], theta1, v1_ref, zeros], dim=-1)
            x2_init = torch.stack([mid_cfgs[:, 0], mid_cfgs[:, 1], theta2, v2_ref, zeros], dim=-1)
            x2_term = torch.stack([ref_b[0] * ones, ref_b[1] * ones, theta2, v2_ref, zeros], dim=-1)
            s_world1 = curr_s.type_as(v1_ref).unsqueeze(0).repeat([x1_init.shape[0], 1])
            s_err1 = world_to_err(s_world1, x1_init)
            s_world2 = torch.stack(
                [x1_term[:, 0], x1_term[:, 1], zeros, x1_term[:, 3], x1_term[:, 2], x1_term[:, 4], zeros], dim=-1)
            s_err2 = world_to_err(s_world2, x2_init)
            if args.use_middle and is_mid:
                p_1 = torch.stack([mid_mu * ones, v1_ref, zeros, zeros], dim=-1)
                p_2 = torch.stack([mid_mu * ones, v2_ref, zeros, zeros], dim=-1)
            else:
                p_1 = torch.stack([mu_a * ones, v1_ref, zeros, zeros], dim=-1)
                p_2 = torch.stack([mu_b * ones, v2_ref, zeros, zeros], dim=-1)
            xp1_tensor = torch.cat([s_err1, p_1], dim=-1)
            xp2_tensor = torch.cat([s_err2, p_2], dim=-1)
            v1 = clf(xp1_tensor)
            v2 = clf(xp2_tensor)
            c1 = net(xp1_tensor[:, args.X_DIM:args.X_DIM + args.N_REF])
            c2 = net(xp2_tensor[:, args.X_DIM:args.X_DIM + args.N_REF])
            loss_roa = torch.relu(v1 - c1 / args.allow_factor) + torch.relu(v2 - c2 / args.allow_factor)
            loss = torch.min(loss_roa)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        sort_v, indices = torch.sort(loss_roa, axis=0)

        xp1_tensor_list = xp1_tensor
        xp2_tensor_list = xp2_tensor
        x1_init_list = x1_init
        x2_init_list = x2_init
        x1_term_list = x1_term
        x2_term_list = x2_term
        v1_ref_list = v1_ref
        v2_ref_list = v2_ref
        mid_cfgs_list = mid_cfgs
        p_1_list = p_1
        p_2_list = p_2

    else:
        v1_ref_cands = torch.linspace(min_v, max_v, n_speeds).type_as(mid_ref)
        v2_ref_cands = torch.linspace(min_v, max_v, n_speeds).type_as(mid_ref)
        norm_len_cands = torch.linspace(-1.0, 1.0, n_points).type_as(mid_ref)

        len_v1_v2 = torch.meshgrid(norm_len_cands, v1_ref_cands, v2_ref_cands)
        len_v1_v2 = torch.stack(len_v1_v2, dim=-1).reshape((-1, 3))
        norm_len, v1_ref, v2_ref = len_v1_v2[:, 0], len_v1_v2[:, 1], len_v1_v2[:, 2]
        ones = torch.ones_like(v1_ref)
        zeros = torch.zeros_like(v1_ref)

        mid_cfgs = torch.stack([mid_point_x + norm_len * np.cos(norm_vec), mid_point_y + norm_len * np.sin(norm_vec)], dim=-1)
        theta1 = torch.atan2(mid_cfgs[:, 1] - ref_a[1], mid_cfgs[:, 0] - ref_a[0])
        theta2 = torch.atan2(ref_b[1] - mid_cfgs[:, 1], ref_b[0] - mid_cfgs[:, 0])
        x1_init = torch.stack([ref_a[0] * ones, ref_a[1] * ones, theta1, v1_ref, zeros], dim=-1)
        x1_term = torch.stack([mid_cfgs[:, 0], mid_cfgs[:, 1], theta1, v1_ref, zeros], dim=-1)
        x2_init = torch.stack([mid_cfgs[:, 0], mid_cfgs[:, 1], theta2, v2_ref, zeros], dim=-1)
        x2_term = torch.stack([ref_b[0] * ones, ref_b[1] * ones, theta2, v2_ref, zeros], dim=-1)

        s_world1 = curr_s.type_as(v1_ref).unsqueeze(0).repeat([x1_init.shape[0], 1])
        s_err1 = world_to_err(s_world1, x1_init)
        s_world2 = torch.stack([x1_term[:, 0], x1_term[:, 1], zeros, x1_term[:, 3], x1_term[:, 2], x1_term[:, 4], zeros], dim=-1)
        s_err2 = world_to_err(s_world2, x2_init)
        if args.use_middle and is_mid:
            p_1 = torch.stack([mid_mu * ones, v1_ref, zeros, zeros], dim=-1)
            p_2 = torch.stack([mid_mu * ones, v2_ref, zeros, zeros], dim=-1)
        else:
            p_1 = torch.stack([mu_a * ones, v1_ref, zeros, zeros], dim=-1)
            p_2 = torch.stack([mu_b * ones, v2_ref, zeros, zeros], dim=-1)
        xp1_tensor = torch.cat([s_err1, p_1], dim=-1)
        xp2_tensor = torch.cat([s_err2, p_2], dim=-1)

        if args.gpus is not None:
            xp1_tensor_list = xp1_tensor.cuda()
            xp2_tensor_list = xp2_tensor.cuda()
            x1_init_list = x1_init.cuda()
            x2_init_list = x2_init.cuda()
            x1_term_list = x1_term.cuda()
            x2_term_list = x2_term.cuda()
            v1_ref_list = v1_ref.cuda()
            v2_ref_list = v2_ref.cuda()
            mid_cfgs_list = mid_cfgs.cuda()
            p_1_list = p_1.cuda()
            p_2_list = p_2.cuda()
        dbg_t6 = time.time()
        v1 = clf(xp1_tensor_list)
        v2 = clf(xp2_tensor_list)
        c1 = net(xp1_tensor_list[:, args.X_DIM:args.X_DIM + args.N_REF])
        c2 = net(xp2_tensor_list[:, args.X_DIM:args.X_DIM + args.N_REF])
        loss_roa = torch.relu(v1 - c1 / args.allow_factor) + torch.relu(v2 - c2 / args.allow_factor)
        sort_v, indices = torch.sort(loss_roa, axis=0)
    dbg_ttt1 = time.time()

    # TODO End differentiable things
    idx = indices[0].item()
    x1_init = x1_init_list[idx]
    x1_term = x1_term_list[idx]
    x2_init = x2_init_list[idx]
    x2_term = x2_term_list[idx]

    dbg_t7 = time.time()

    if args.gpus is not None:
        x1_init = x1_init.cuda()
        x1_term = x1_term.cuda()
        x2_init = x2_init.cuda()
        x2_term = x2_term.cuda()

    v1_ref = v1_ref_list[idx]
    v2_ref = v2_ref_list[idx]

    mid_cfgs = mid_cfgs_list[idx]
    half_road_length_1 = ((mid_cfgs[1] - ref_a[1]) ** 2 + (mid_cfgs[0] - ref_a[0]) ** 2) ** 0.5
    half_road_length_2 = ((mid_cfgs[1] - ref_b[1]) ** 2 + (mid_cfgs[0] - ref_b[0]) ** 2) ** 0.5
    nt_1 = int(half_road_length_1/v1_ref/args.dt)
    nt_2 = int(half_road_length_2/v2_ref/args.dt)
    xp1_tensor = xp1_tensor_list[idx].unsqueeze(0)
    xp2_tensor = xp2_tensor_list[idx].unsqueeze(0)
    p_1 = p_1_list[idx]
    p_2 = p_2_list[idx]
    if args.gpus is not None:
        p_1 = p_1.cuda()
        p_2 = p_2.cuda()
    x1_list = get_ref_list(x1_init_list[idx], nt_1, args.dt)
    x2_list = get_ref_list(x2_init_list[idx], nt_2, args.dt)

    dbg_t8 = time.time()

    trs_1, _ = za.get_trajectories(xp1_tensor, actor, car_params, params=xp1_tensor[:, args.X_DIM:], args=args, manual_t=nt_1)
    s_err2_real = world_to_err(err_to_world(trs_1[0, -1, :args.X_DIM], x1_term), x2_init)
    x1_real_list = err_to_world_list(trs_1[0, :, :args.X_DIM], x1_list)

    if only_first_half:
        xp2_tensor_real = None
        trs_2 = None
        x2_real_list = None
    else:
        xp2_tensor_real = torch.cat([s_err2_real, p_2], dim=0).unsqueeze(0)
        trs_2, _ = za.get_trajectories(xp2_tensor_real, actor, car_params, params=xp2_tensor[:, args.X_DIM:], args=args, manual_t=nt_2)
        x2_real_list = err_to_world_list(trs_2[0, :, :args.X_DIM], x2_list)

    dbg_t9 = time.time()

    return v1_ref, 0, 0, 0, v2_ref, 0, 0, 0, 0, 0, xp1_tensor, xp2_tensor, x1_list, x2_list, trs_1, trs_2, x1_real_list, x2_real_list


def plan_a_b_lqr2(curr_s, ref_a, theta_a, mu_a, ref_b, theta_b, mu_b, seg_i, args):
    if args.lqr_nt is not None:
        num_t = args.lqr_nt
    else:
        num_t = args.nt
    total_time = num_t * args.dt * 2
    v_ref1_lqr = torch.tensor((((ref_a[0] - ref_b[0])**2 + (ref_a[1] - ref_b[1])**2)**0.5 / total_time).item())
    w_ref1_lqr = torch.tensor(0.0)
    v_ref2_lqr = v_ref1_lqr.clone()
    w_ref2_lqr = torch.tensor(0.0)

    if args.gpus is not None:
        mu_a = mu_a.cuda()
        mu_b = mu_b.cuda()
        v_ref1_lqr = v_ref1_lqr.cuda()
        w_ref1_lqr = w_ref1_lqr.cuda()
        v_ref2_lqr = v_ref2_lqr.cuda()
        w_ref2_lqr = w_ref2_lqr.cuda()

    zo = torch.zeros_like(v_ref1_lqr)
    x1_lqr_init = torch.stack([ref_a[0], ref_a[1], torch.atan2(ref_b[1]-ref_a[1], ref_b[0]-ref_a[0]), v_ref1_lqr, w_ref1_lqr])
    x1_lqr_list = get_ref_list(x1_lqr_init, num_t, args.dt)
    x1_lqr_term = x1_lqr_list[-1]
    x2_lqr_init = torch.stack([x1_lqr_term[0], x1_lqr_term[1], x1_lqr_term[2], v_ref2_lqr, w_ref2_lqr], dim=-1)
    x2_lqr_list = get_ref_list(x2_lqr_init, num_t, args.dt)
    x2_lqr_term = x2_lqr_list[-1]

    p_lqr_1 = torch.stack([mu_a, v_ref1_lqr, w_ref1_lqr, zo])
    p_lqr_2 = torch.stack([mu_b, v_ref2_lqr, w_ref2_lqr, zo])

    s_lqr_err1 = world_to_err(curr_s, x1_lqr_init)
    xp1_lqr_tensor = torch.cat([s_lqr_err1, p_lqr_1], dim=0).unsqueeze(0)

    _, lqr_K1 = za.solve_for_P_K(p_lqr_1.unsqueeze(0).detach().cpu(), car_params, args)
    _, lqr_K2 = za.solve_for_P_K(p_lqr_2.unsqueeze(0).detach().cpu(), car_params, args)
    if args.gpus is not None:
        lqr_K1_cuda = lqr_K1.cuda()
        lqr_K2_cuda = lqr_K2.cuda()


    trs_lqr2_1, u1 = za.get_trajectories(xp1_lqr_tensor, None, car_params, params=p_lqr_1.unsqueeze(0),
                                        is_lqr=True, lqr_K_cuda=lqr_K1_cuda, args=args, manual_t=num_t)
    s_lqr_err2 = world_to_err(err_to_world(trs_lqr2_1[0, -1, :args.X_DIM], x1_lqr_term), x2_lqr_init)
    xp2_lqr_tensor = torch.cat([s_lqr_err2, p_lqr_2], dim=0).unsqueeze(0)
    trs_lqr2_2, u2 = za.get_trajectories(xp2_lqr_tensor, None, car_params, params=p_lqr_2.unsqueeze(0),
                                    is_lqr=True, lqr_K_cuda=lqr_K2_cuda, args=args, manual_t=num_t)

    x1_real_lqr2_list = err_to_world_list(trs_lqr2_1[0, :, :args.X_DIM], x1_lqr_list)
    x2_real_lqr2_list = err_to_world_list(trs_lqr2_2[0, :, :args.X_DIM], x2_lqr_list)


    return x1_lqr_list, x2_lqr_list, x1_real_lqr2_list, x2_real_lqr2_list


def plot_traj(x_list, color, label, linewidth, alpha=1, no_legend=False):
    if no_legend:
        plt.plot(x_list[:, 0], x_list[:, 1], color=color, linewidth=linewidth, alpha=alpha)
    else:
        plt.plot(x_list[:,0], x_list[:,1], color=color, label=label, linewidth=linewidth, alpha=alpha)


def plot_map(map_cache, args):
    legend_fontsize = 15
    if args.num_trials == 10:
        aligns = (2, 5)  # rows, columns
    elif args.num_trials == 9:
        aligns = (3, 3)
    elif args.num_trials == 8:
        aligns = (2, 4)
    elif args.num_trials == 12:
        aligns = (3, 4)
    elif args.num_trials == 24:
        aligns = (4, 6)
    elif args.num_trials == 16:
        aligns = (4, 4)
    elif args.num_trials == 21:
        aligns = (3, 7)
    elif args.num_trials == 25:
        aligns = (5, 5)
    else:
        raise NotImplementedError
    fig = plt.figure(figsize=(8, 4))
    for trial_i in range(args.num_trials):
        roads, thetas, mus, new_roads, new_thetas, new_mus = \
            map_cache["roads"][trial_i], map_cache["thetas"][trial_i], map_cache["mus"][trial_i],\
            map_cache["new_roads"][trial_i], map_cache["new_thetas"][trial_i], map_cache["new_mus"][trial_i]

        # align
        axes = plt.subplot(aligns[0], aligns[1], trial_i+1)
        plt.subplots_adjust(wspace=0, hspace=0)

        # plot
        plot_seg(new_roads, "steelblue", "Roads", 2.0, alpha=0.8, mus=new_mus, boundary=True, use_label=trial_i==0,
                 fig=fig, axes=axes)
        plt.tight_layout(pad=0)
    # savefig
    ax=plt.gca()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    handles, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(handles, labels, loc='upper center', fontsize=legend_fontsize, ncol=2)
    plt.tight_layout(pad=0)
    plt.savefig("%s/car_env_demo.png"%(args.exp_dir_full), bbox_inches='tight', pad_inches=0.01)
    plt.close()


def plot_seg(x_list, color, label, linewidth, alpha=1, mus=None, use_label=True, boundary=False, fig=None, axes=None):
    angle_list=[]
    width = linewidth

    seen_icy=False
    seen_common=False

    x_min=1000000
    x_max=-x_min
    y_min=1000000
    y_max=-y_min

    for i in range(x_list.shape[0]-1):
        angle_mid = np.arctan2(x_list[i + 1, 1] - x_list[i, 1], x_list[i + 1, 0] - x_list[i, 0])
        if i != 0:
            angle_left = np.arctan2(x_list[i, 1] - x_list[i - 1, 1], x_list[i, 0] - x_list[i -1, 0])
            angle0 = (angle_left + angle_mid)/2
        else:
            angle0 = angle_mid
        angle0 += np.pi/2

        if i!=x_list.shape[0]-2:
            angle_right = np.arctan2(x_list[i+2, 1] - x_list[i + 1, 1], x_list[i+2, 0] - x_list[i + 1, 0])
            angle1 = (angle_mid + angle_right)/2
        else:
            angle1 = angle_mid
        angle1 += np.pi/2

        x0 = x_list[i, 0]
        y0 = x_list[i, 1]
        x1 = x_list[i+1, 0]
        y1 = x_list[i+1, 1]
        x00 = x0 + width / np.sin(angle0-angle_mid) * np.cos(angle0)
        y00 = y0 + width / np.sin(angle0-angle_mid) * np.sin(angle0)
        x01 = x0 - width / np.sin(angle0-angle_mid) * np.cos(angle0)
        y01 = y0 - width / np.sin(angle0-angle_mid) * np.sin(angle0)
        x10 = x1 + width / np.sin(angle1-angle_mid) * np.cos(angle1)
        y10 = y1 + width / np.sin(angle1-angle_mid) * np.sin(angle1)
        x11 = x1 - width / np.sin(angle1-angle_mid) * np.cos(angle1)
        y11 = y1 - width / np.sin(angle1-angle_mid) * np.sin(angle1)
        points = np.array([[x00, y00], [x01, y01], [x11, y11], [x10, y10]])
        hull = ConvexHull(points=points)
        ax = plt.gca()

        curr_icy = True
        if mus is not None:
            if mus[i] == 0:
                if mus[i+1] == 1:
                    the_alpha = alpha
                    curr_icy=False
                else:
                    the_alpha = alpha * 0.2
            else:
                if mus[i] == 1:
                    the_alpha = alpha
                    curr_icy = False
                else:
                    the_alpha = alpha * 0.2
        else:
            the_alpha = alpha
            curr_icy = False

        the_label = None
        if use_label:
            if seen_icy == False and curr_icy:
                the_label = "Icy road"
                seen_icy = True
            if seen_common == False and not curr_icy:
                the_label = "Normal road"
                seen_common = True
        patch = Polygon(points[hull.vertices], edgecolor=None, facecolor=color, antialiased=True, label=the_label, alpha=the_alpha)
        ax.add_patch(patch)
        ax.autoscale_view()

        x_min = min(x_min, np.min(points[:, 0]))
        x_max = max(x_max,np.max(points[:, 0]))
        y_min = min(y_min,np.min(points[:, 1]))
        y_max = max(y_max,np.max(points[:, 1]))

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    if boundary:
        plt.axis('off')
        ax=plt.gca()
        axis = ax.axis()
        rec = plt.Rectangle((axis[0] - 0.0, axis[2] - 0.0), (axis[1] - axis[0]) + 0, (axis[3] - axis[2]) + 0.,
                            fill=False, lw=1, linestyle="-")
        ax.add_patch(rec)
        ax.autoscale_view()



def plot_point(point, color, marker, markersize, label):
    plt.scatter(to_item(point[0]), to_item(point[1]), color=color, marker=marker, s=markersize, label=label)


def plot_point_np(point, color, marker, markersize, label):
    plt.scatter(point[0], point[1], color=color, marker=marker, s=markersize, label=label)

def plot_arrow_np(point, color, marker, markersize, label, no_label=False):
    arr_l = 2.4  # 1.5
    arr_w = 0.8  # 0.5
    plt.arrow(point[0]-arr_l*np.cos(point[4]), point[1]-arr_l*np.sin(point[4]),
              arr_l*np.cos(point[4]), arr_l*np.sin(point[4]), width=arr_w, color=color,
              label=label if no_label==False else None, zorder=99999)


def get_thetas(roads):
    theta_list=[]
    for i in range(roads.shape[0]):
        if i==0:
            theta = np.arctan2(roads[i+1, 1] - roads[i, 1], roads[i+1, 0] - roads[i, 0])
        elif i==roads.shape[0]-1:
            theta = np.arctan2(roads[i, 1] - roads[i-1, 1], roads[i, 0] - roads[i-1, 0])
        else:
            theta1 = np.arctan2(roads[i+1, 1] - roads[i, 1], roads[i+1, 0] - roads[i, 0])
            theta2 = np.arctan2(roads[i, 1] - roads[i-1, 1], roads[i, 0] - roads[i-1, 0])
            theta = (theta1 + theta2) / 2
        theta_list.append(theta)
    return np.array(theta_list)



def generate_roads(args):
    # TODO challenging case
    roads = np.array([[0, 0], [19.0, 13.0], [17, 24], [36, 29], [34, 14], [19, 0]])
    thetas = np.array([0.0, 0.7, 0.7, 0.7, 0.7, 0.7])
    mus = np.array([1.0, 0.01, 0.01, 1.0, 0.01, 1.0])

    # TODO (rect)
    roads = np.array([[0, 0], [25, 0], [25, 10], [0, 10]] * 2)
    thetas = np.array([0.0, 0.7, 0.7, 0.7] * 2)
    mus = np.array([1.0, 0.02, 0.02, 1.0] * 2)

    # TODO (ring)
    num_segs = args.new_ring_seg  # 5
    roads = [np.array([0, 0])]
    d_theta = 2 * np.pi / num_segs
    theta = d_theta
    length = 15
    num_pieces = num_segs

    if args.animation:
        lens = [40.0, 9.0, 30.0]
        mus = np.array([1.0, 0.1, 1.0, 0.1])
        thetas = np.array([0, np.pi / 4, np.pi / 2, np.pi / 2])

        for i in range(num_pieces):
            road_seg = np.array([roads[-1][0] + lens[i] * np.cos(thetas[i]),
                                roads[-1][1] + lens[i] * np.sin(thetas[i])])
            roads.append(road_seg)
        roads = np.stack(roads, axis=0)
    else:
        if args.race:
            if args.wild:
                d_theta = args.race_angle
            else:
                d_theta = args.race_angle / 180 * np.pi
            mus = np.random.choice([1.0, 0.1], num_pieces+1)
            for i in range(num_pieces):
                ratio = np.random.choice([2.5, 1, 0.5])
                d_angle = np.random.choice([1.5*d_theta, d_theta,0.5*d_theta,-0.5*d_theta,-d_theta,-1.5*d_theta])
                if args.wild:
                    theta = theta + d_angle
                else:
                    theta = (theta + d_angle + np.pi * 2) % (np.pi *2)
                # print(theta)
                if i == 0:
                    theta = np.random.choice([-1.0, -0.5, -0.25, 0.25, 0.5, 1.0])
                roads.append(
                    np.array([roads[-1][0] + ratio*length * np.cos(theta), roads[-1][1] + ratio*length * np.sin(theta)]))
            roads = np.stack(roads, axis=0)
            thetas = np.array([0.0] * (num_pieces + 1))
        else:
            for i in range(num_pieces):
                if i in [1, 2, 3]:
                    roads.append(
                        np.array([roads[-1][0] + length * 0.5 * np.cos(theta), roads[-1][1] + length * 0.5 * np.sin(theta)]))
                elif i == 0:
                    roads.append(np.array(
                        [roads[-1][0] + length * 2.5 * np.cos(theta), roads[-1][1] + length * 2.5 * np.sin(theta)]))
                else:
                    roads.append(np.array([roads[-1][0] + length * np.cos(theta), roads[-1][1] + length * np.sin(theta)]))
                if i >= num_pieces // 2 or args.always_same:
                    theta += d_theta
                else:
                    theta -= d_theta
            roads = np.stack(roads, axis=0)
            thetas = np.array([0.0] * (num_pieces + 1))
            mus = np.array([1.0, 0.1, 0.1, 0.1, 1.0, 0.1, 1.0] * (num_pieces + 1))

    if args.all_ice:
        mus[3:] = 0.1

    if args.use_middle:
        mid_roads = []
        mid_mus = []
        new_roads = []
        new_mus = []
        for i in range(roads.shape[0] - 1):
            mid_roads.append((roads[i] + roads[i + 1]) / 2)
            mid_mus.append(0.0)
            new_roads.append(roads[i])
            new_roads.append(mid_roads[i])
            new_mus.append(mus[i])
            new_mus.append(mid_mus[i])
        new_roads.append(roads[-1])
        new_mus.append(mus[-1])
        new_roads = np.stack(new_roads)
        new_thetas = np.array([0.0] * (num_pieces + 1) * 2)
        new_mus = np.stack(new_mus)
    else:
        new_roads = None
        new_thetas = None
        new_mus = None
    return roads, thetas, mus, new_roads, new_thetas, new_mus


def plt_lim_scaled(trajs=None, new_roads=None):
    xmin, xmax, ymin, ymax = plt.axis()
    side_len = max(xmax - xmin, ymax - ymin) / 2
    side_len *= 1.3
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    if args.animation:
        plt.xlim(0, 55)
        plt.ylim(-5, 35)
    elif args.multi_ani:
        set_xmin = torch.min(trajs[:, 0])
        set_xmax = torch.max(trajs[:, 0])
        set_ymin = torch.min(trajs[:, 1])
        set_ymax = torch.max(trajs[:, 1])

        xmin1 = np.min(new_roads[:, 0])
        xmax1 = np.max(new_roads[:, 0])
        ymin1 = np.min(new_roads[:, 1])
        ymax1 = np.max(new_roads[:, 1])

        gap = 10
        the_xmin = min(set_xmin, xmin1) - gap
        the_xmax = max(set_xmax, xmax1) + gap
        the_ymin = min(set_ymin, ymin1) - gap
        the_ymax = max(set_ymax, ymax1) + gap

        plt.xlim(the_xmin, the_xmax)
        plt.ylim(the_ymin, the_ymax)
    else:
        plt.axis("scaled")
        plt.xlim(cx - side_len, cx + side_len)
        plt.ylim(cy - side_len, cy + side_len)

def get_angle(node1, node2):
    return torch.atan2(node2[1] - node1[1], node2[0] - node1[0])


def main():
    global args
    if args.load_from:
        old_args = np.load(ospj(args.load_from, "args.npz"), allow_pickle=True)['args'].item()
        old_args.load_from = args.load_from
        old_args.from_file = args.from_file
        old_args.max_keep_viz_trials = args.max_keep_viz_trials
        old_args.use_d = args.use_d
        old_args.plot_map_only = args.plot_map_only
        old_args.multi_ani = args.multi_ani
        old_args.select_me = args.select_me
        old_args.exp_dir_full = utils.get_exp_dir()+old_args.exp_dir_full.split("/")[-1]
        old_args.viz_dir = utils.get_exp_dir() + "/"+old_args.viz_dir.split("/")[-2] + "/"+old_args.viz_dir.split("/")[-1]
        args = old_args
    else:
        utils.set_seed_and_exp(args, offset=1)
    V_INIT = 5.0
    CAP_SEG = args.cap_seg
    assert args.use_middle
    if args.load_from is None:
        map_cache = {"roads":[], "thetas":[], "mus":[],
                     "new_roads":[], "new_thetas":[], "new_mus":[]}
        cache = {
            me_i: {
                "trajs": [],  # a list (trials) of data
                "segs": [], # a list (trials) of list (segs) data
                "refs": [],   # a list (trials) of data
            } for me_i, method in enumerate(args.methods)
        }

        for me_i, method in enumerate(args.methods):
            # RMSE towards reference
            cache[me_i]["rmse"] = []
            cache[me_i]["masked_rmse"] = []
            # deviation towards the lane
            cache[me_i]["dev"] = []
            cache[me_i]["masked_dev"] = []
            cache[me_i]["valid"] = []
            cache[me_i]["dist_goal"] = []
            cache[me_i]["dist_goal_rel"] = []
            cache[me_i]["dist_goal_t"] = []
            cache[me_i]["time"] = []
            cache[me_i]["valid_time"] = []
            cache[me_i]["runtime"] = []
            cache[me_i]["runtime_step"] = []
            cache[me_i]["reward"] = []

        # TODO Initialize & load models
        rlp_list = utils.get_loaded_rl_list(args.rlp_paths, args.methods, args.auto_rl)

        clf_list = []
        actor_list = []
        net_list = []
        mbpo_list = []
        pets_list = []
        counter={"mbpo":0, "pets":0}
        for me_i, method in enumerate(args.methods):
            clf_tmp, actor_tmp, net_tmp, mbpo_tmp, pets_tmp = None, None, None, None, None
            if method == "ours" or method == "ours-d":
                print("our method", args.methods)
                clf_tmp, _ = load_nn("clf", args.ours_clf_paths[me_i], args)
                actor_tmp, _ = load_nn("actor", args.ours_actor_paths[me_i], args)
                net_tmp, _ = load_nn("net", args.ours_roa_paths[me_i], args)
            elif method == "mbpo":
                mbpo_tmp = mbrl_utils.get_mbpo_models(args.mbpo_paths[counter["mbpo"]], args)
                counter["mbpo"]+=1
            elif method == "pets":
                pets_tmp = mbrl_utils.get_pets_models(args.pets_paths[counter["pets"]], args)
                counter["pets"]+=1

            clf_list.append(clf_tmp)
            actor_list.append(actor_tmp)
            net_list.append(net_tmp)
            mbpo_list.append(mbpo_tmp)
            pets_list.append(pets_tmp)


        if "clf" in args.methods:
            mono_actor, _ = load_nn("clf_actor", args.mono_actor_pretrained_path, args)

    if args.load_from is None:
        for trial_i in range(args.num_trials):
            # TODO generate the map
            roads, thetas, mus, new_roads, new_thetas, new_mus = generate_roads(args)
            map_cache["roads"].append(roads)
            map_cache["thetas"].append(thetas)
            map_cache["mus"].append(mus)
            map_cache["new_roads"].append(new_roads)
            map_cache["new_thetas"].append(new_thetas)
            map_cache["new_mus"].append(new_mus)

            if args.plot_map_only:
                continue

            init_test_x = torch.tensor([0.0, 0.0, 0.0, V_INIT, 0.0, 0.0, 0.0])
            roads = to_cuda(to_tensor(roads), args)
            thetas = to_cuda(to_tensor(thetas), args)
            mus = to_cuda(to_tensor(mus), args)
            if args.use_middle:
                new_roads = to_cuda(to_tensor(new_roads), args)
                new_thetas = to_cuda(to_tensor(new_thetas), args)
                new_mus = to_cuda(to_tensor(new_mus), args)
            curr_env = roads, thetas, mus, new_roads, new_thetas, new_mus

            plt.figure(figsize=(8, 4))
            plot_seg(to_np(new_roads), "steelblue", "Roads", 2.0, alpha=0.4, mus=new_mus)
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.tight_layout()
            plt.legend()
            plt.savefig("%s/viz_seg_init.png" % (args.viz_dir), bbox_inches='tight', pad_inches=0.1)
            plt.close()

            for me_i, method in enumerate(args.methods):
                if "mbpo" in method:
                    cache = solve_in_mbpo(trial_i, me_i, init_test_x, cache, curr_env, mbpo_list[me_i], CAP_SEG, args)

                if "pets" in method:
                    cache = solve_in_pets(trial_i, me_i, init_test_x, cache, curr_env, pets_list[me_i], CAP_SEG, args)

                if "rl" in method:
                    cache = solve_in_rl(trial_i, me_i, init_test_x, cache, curr_env, rlp_list[me_i], CAP_SEG, args)

                if "clf" in method:
                    cache = solve_in_clf(trial_i, me_i, init_test_x, cache, curr_env, mono_actor, CAP_SEG, args)

                if "lqr" in method:
                    cache = solve_in_lqr(trial_i, me_i, init_test_x, cache, curr_env, CAP_SEG, args)

                if "mpc" in method:
                    cache = solve_in_mpc(trial_i, me_i, init_test_x, cache, curr_env, CAP_SEG, args)

                if "ours" in method:
                    if "d" in method:
                        cache = solve_in_ours(trial_i, me_i, init_test_x, cache, curr_env, actor_list[me_i], clf_list[me_i], net_list[me_i], CAP_SEG,
                                              diff_planner=True, args=args)
                    else:
                        cache = solve_in_ours(trial_i, me_i, init_test_x, cache, curr_env, actor_list[me_i], clf_list[me_i], net_list[me_i], CAP_SEG,
                                              diff_planner=False, args=args)

        np.savez("%s/cache.npz"%(args.exp_dir_full), data=cache)
        np.savez("%s/map_cache.npz" % (args.exp_dir_full), data=map_cache)
    else:
        cache = np.load(ospj(args.load_from, "cache.npz"), allow_pickle=True)['data'].item()
        if os.path.exists(ospj(args.load_from, "map_cache.npz")):
            map_cache = np.load(ospj(args.load_from, "map_cache.npz"), allow_pickle=True)['data'].item()
        else:
            map_cache = None
    if args.plot_map_only:
        plot_map(map_cache, args)
        return

    if args.from_file:
        utils.gen_plot_rl_std_curve(
            args.ds_list, args.methods, cache, "%s/std_car_curve.png" % (args.exp_dir_full),
        )#loc="upper left")
        return

    if args.animation==False and args.multi_ani==False:
        cs_d = utils.get_cs_d()
        for metric in ["rmse", "masked_rmse", "dev", "masked_dev", "valid", "dist_goal", "dist_goal_rel", "dist_goal_t", "time", "valid_time", "runtime", "runtime_step"]:
            utils.metric_bar_plot(cache, metric, args, header="_car",
                                  index=True, cs_d=cs_d, mode="car", rl_merged=True)
            utils.metric_bar_plot(cache, metric, args, header="_car",
                                  index=True, cs_d=cs_d, mode="car", rl_merged=False)

    if args.multi_ani or args.load_from is None:
        print("Finished the metric, now plotting...")
        # TODO (visualization)
        markersize = 360
        if args.multi_ani:
            linewidth = 2.5
            fontsize = 15
            legend_fontsize = 12
        else:
            linewidth = 4
            fontsize = 20
            legend_fontsize = 15
        color_maps = {"ours": "purple", "ours-d": "purple", "rl": "red", "rl-ddpg":"red", "rl-sac":"red","rl-ppo":"red", 
                      "lqr": "black", "mpc": "green", "clf": "gray", "mbpo":"pink", "pets":"brown"}
        label_maps = {"ours": "Ours", "ours-d": "Ours-d", "rl": "RL", "rl-ddpg":"DDPG", "rl-sac":"SAC","rl-ppo":"PPO", 
                      "lqr": "LQR", "mpc": "MPC", "clf":"CLF", "mbpo":"MBPO", "pets":"PETS"}


        cs_d = utils.get_cs_d()
        color_maps = cs_d

        if args.multi_ani and args.select_me is not None:
            args.methods = [args.methods[me_i] for me_i in args.select_me]
            cache = {ii: cache[me_i] for ii, me_i in enumerate(args.select_me)}
            assert args.methods[-1] == "ours-d"
            args.methods[-1] = "ours"

        for trial_i in range(min(args.num_trials, args.max_keep_viz_trials)):

            max_tlen = np.max([cache[me_i]["trajs"][trial_i].shape[0] for me_i, method in enumerate(args.methods)])
            min_tlen = np.min([cache[me_i]["trajs"][trial_i].shape[0] for me_i, method in enumerate(args.methods)])
            print(trial_i,min_tlen, max_tlen, args.num_trials, args.max_keep_viz_trials)
            for ti in range(max_tlen):
                if args.multi_ani:
                    if ti != min_tlen - 1:
                        continue
                else:
                    if ti % args.skip_viz !=0:
                        continue
                # if ti % args.skip_viz == 0:
                plt.figure(figsize=(8, 4))
                plot_seg(map_cache["new_roads"][trial_i], "steelblue", "Roads", 2.0, alpha=0.8, mus=map_cache["new_mus"][trial_i])
                for me_i, method in enumerate(args.methods):
                    trajs = to_np(cache[me_i]["trajs"][trial_i])
                    plot_traj(trajs, color_maps[method], label_maps[method] + "",
                              linewidth, no_legend=True, alpha=0)
                for me_i, method in enumerate(args.methods):
                    trajs = to_np(cache[me_i]["trajs"][trial_i])
                    plot_traj(trajs[:ti], color_maps[method], label_maps[method] + "",
                              linewidth, alpha=0.85)
                for me_i, method in enumerate(args.methods):
                    trajs = to_np(cache[me_i]["trajs"][trial_i])
                    if ti < trajs.shape[0]:
                        plot_arrow_np(trajs[ti], color_maps[method], ".", markersize, label_maps[method] + " State", no_label=True)
                plt.xlabel("x (m)", fontsize=fontsize)
                plt.ylabel("y (m)", fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                if args.multi_ani:
                    for me_i, method in enumerate(args.methods):
                        if method == "ours-d":
                            plt_lim_scaled(cache[me_i]["trajs"][trial_i], map_cache["new_roads"][trial_i])
                else:
                    plt_lim_scaled()
                ax = plt.gca()
                ax_handles, ax_labels = ax.get_legend_handles_labels()
                ax_handles = ax_handles[-2:] + ax_handles[:-2]
                ax_labels = ax_labels[-2:] + ax_labels[:-2]
                plt.legend(ax_handles, ax_labels, fontsize=legend_fontsize)
                if args.multi_ani:
                    plt.savefig("%s/trial%04d.png" % (args.viz_dir, trial_i), bbox_inches='tight', pad_inches=0.1)
                else:
                    plt.savefig("%s/trial%04d_viz_t%04d.png" % (args.viz_dir, trial_i, ti), bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()


def solve_in_mpc(trial_i, me_i, init_test_x, cache, curr_env, CAP_SEG, args):
    mpc_t1 = time.time()
    mpc_ulen = 10
    mpc_horizon = 20
    curr_s = to_cuda(init_test_x.detach().reshape((1, -1)), args)
    half_time = args.nt * args.dt
    cache[me_i]["trajs"].append([])
    cache[me_i]["refs"].append([])
    cache[me_i]["segs"].append([])
    roads, thetas, mus, new_roads, new_thetas, new_mus = curr_env
    # print("start mpc")
    for seg_i in range(min(CAP_SEG * 2, new_roads.shape[0] - 1)):
        cache[me_i]["segs"][-1].append([])
        ref_a, ref_b, ref_c, mu_a, mu_b, v_ref1_mpc, v_ref2_mpc, w_ref1_mpc, w_ref2_mpc = \
            get_ref_from_roads(new_roads, new_mus, seg_i, half_time)

        params_a = torch.stack([mu_a.cpu(), v_ref1_mpc, w_ref1_mpc, torch.zeros_like(w_ref1_mpc)])
        params_b = torch.stack([mu_b.cpu(), v_ref2_mpc, w_ref2_mpc, torch.zeros_like(w_ref2_mpc)])
        if args.gpus is not None:
            v_ref1_mpc = v_ref1_mpc.cuda()
            w_ref1_mpc = w_ref1_mpc.cuda()
            v_ref2_mpc = v_ref2_mpc.cuda()
            w_ref2_mpc = w_ref2_mpc.cuda()
            params_a = params_a.cuda()
            params_b = params_b.cuda()
        angle1 = get_angle(ref_a, ref_b)
        angle2 = get_angle(ref_b, ref_c)
        x1_mpc_init = torch.stack([ref_a[0], ref_a[1], angle1, v_ref1_mpc, w_ref1_mpc])
        x1_mpc_list = get_ref_list(x1_mpc_init, args.nt, args.dt)
        x1_mpc_term = x1_mpc_list[-1]
        x2_mpc_init = torch.stack([x1_mpc_term[0], x1_mpc_term[1], angle2, v_ref2_mpc, w_ref2_mpc])
        x2_mpc_list = get_ref_list(x2_mpc_init, args.nt, args.dt)
        for ti in range(0, args.nt, mpc_ulen):
            s_rl_err1 = world_to_err(curr_s, x1_mpc_list[ti:ti + 1])
            xs_try_mpc, us_mpc = car_mpc.plan_mpc(s_rl_err1, x1_mpc_list[ti:], x2_mpc_list,
                                                  params_a, params_b, mpc_horizon, args.dt, args)
            for ttti in range(mpc_ulen):
                s_rl_err1, _ = za.next_step(s_rl_err1, us_mpc[ttti:ttti + 1], car_params, params_a,
                                            False, None, args, detached=False, directed_u=True)
                cache[me_i]["trajs"][-1].append(err_to_world(s_rl_err1, x1_mpc_list[ti + ttti + 1:ti + ttti + 2]))
                cache[me_i]["refs"][-1].append(x1_mpc_list[ti + ttti + 1:ti + ttti + 2])
                cache[me_i]["segs"][-1][-1].append(cache[me_i]["trajs"][-1][-1])
            curr_s = cache[me_i]["trajs"][-1][-1]

    # TODO cache
    cache[me_i]["trajs"][-1] = torch.cat(cache[me_i]["trajs"][-1], dim=0)
    cache[me_i]["refs"][-1] = torch.cat(cache[me_i]["refs"][-1], dim=0)
    for seg_i in range(len(cache[me_i]["segs"][-1])):
        cache[me_i]["segs"][-1][seg_i] = torch.cat(cache[me_i]["segs"][-1][seg_i], dim=0)

    mpc_t2 = time.time()
    cache = compute_metric(cache, curr_env, me_i, args.methods[me_i], trial_i, mpc_t2 - mpc_t1, cache[me_i]["trajs"][-1].shape[0])
    return cache


def solve_in_mbpo(trial_i, me_i, init_test_x, cache, curr_env, mbpo_net, CAP_SEG, args):
    rl_t1 = time.time()
    curr_s = to_cuda(init_test_x.detach().reshape((1, -1)), args)
    half_time = args.nt * args.dt
    cache[me_i]["trajs"].append([])
    cache[me_i]["refs"].append([])
    cache[me_i]["segs"].append([])
    roads, thetas, mus, new_roads, new_thetas, new_mus = curr_env
    for seg_i in range(min(CAP_SEG * 2, new_roads.shape[0] - 1)):
        cache[me_i]["segs"][-1].append([])
        ref_a, ref_b, ref_c, mu_a, mu_b, v_ref1_rl, v_ref2_rl, w_ref1_rl, w_ref2_rl = \
            get_ref_from_roads(new_roads, new_mus, seg_i, half_time)

        params = torch.stack([mu_a.cpu(), v_ref1_rl, w_ref1_rl, torch.zeros_like(w_ref1_rl)])
        if args.gpus is not None:
            mu_a = mu_a.cuda()
            mu_b = mu_b.cuda()
            v_ref1_rl = v_ref1_rl.cuda()
            w_ref1_rl = w_ref1_rl.cuda()
            v_ref2_rl = v_ref2_rl.cuda()
            w_ref2_rl = w_ref2_rl.cuda()
            params = params.cuda()

        angle1 = get_angle(ref_a, ref_b)
        angle2 = get_angle(ref_b, ref_c)

        x1_rl_init = torch.stack([ref_a[0], ref_a[1], angle1, v_ref1_rl, w_ref1_rl])
        x1_rl_list = get_ref_list(x1_rl_init, args.nt, args.dt)

        for ti in range(args.nt):
            s_rl_err1 = world_to_err(curr_s, x1_rl_list[ti:ti + 1])
            dist = car_env.dist_to_seg(curr_s[0, :2], ref_a, ref_b, ref_c, use_torch=True)
            obs = torch.zeros((1, 16)).cuda()
            obs[:, 0:7] = s_rl_err1
            obs[0, 7:] = torch.stack([v_ref1_rl, w_ref1_rl, mu_a, angle1, dist,
                                      angle2, v_ref2_rl, w_ref2_rl, mu_b], dim=-1)
            u = utils.rl_u(obs, None, mbpo_net, mbpo=True)
            new_s_rl_err, _ = za.next_step(s_rl_err1, u, car_params, params,
                                           False, None, args, detached=False, directed_u=True)
            curr_s = err_to_world(new_s_rl_err, x1_rl_list[ti + 1:ti + 2]).detach()
            cache[me_i]["trajs"][-1].append(curr_s)
            cache[me_i]["refs"][-1].append(x1_rl_list[ti + 1:ti + 2])
            cache[me_i]["segs"][-1][-1].append(curr_s)
    # TODO visualize
    cache[me_i]["trajs"][-1] = torch.cat(cache[me_i]["trajs"][-1], dim=0)
    cache[me_i]["refs"][-1] = torch.cat(cache[me_i]["refs"][-1], dim=0)
    for seg_i in range(len(cache[me_i]["segs"][-1])):
        cache[me_i]["segs"][-1][seg_i] = torch.cat(cache[me_i]["segs"][-1][seg_i], dim=0)

    rl_t2 = time.time()
    cache = compute_metric(cache, curr_env, me_i, args.methods[me_i], trial_i, rl_t2 - rl_t1, cache[me_i]["trajs"][-1].shape[0])
    return cache


def solve_in_rl(trial_i, me_i, init_test_x, cache, curr_env, rlp_pair, CAP_SEG, args):
    rl_t1 = time.time()
    curr_s = to_cuda(init_test_x.detach().reshape((1, -1)), args)
    half_time = args.nt * args.dt
    cache[me_i]["trajs"].append([])
    cache[me_i]["refs"].append([])
    cache[me_i]["segs"].append([])
    roads, thetas, mus, new_roads, new_thetas, new_mus = curr_env
    rlp_policy, rlp_running = rlp_pair
    for seg_i in range(min(CAP_SEG * 2, new_roads.shape[0] - 1)):
        cache[me_i]["segs"][-1].append([])
        ref_a, ref_b, ref_c, mu_a, mu_b, v_ref1_rl, v_ref2_rl, w_ref1_rl, w_ref2_rl = \
            get_ref_from_roads(new_roads, new_mus, seg_i, half_time)

        params = torch.stack([mu_a.cpu(), v_ref1_rl, w_ref1_rl, torch.zeros_like(w_ref1_rl)])
        if args.gpus is not None:
            mu_a = mu_a.cuda()
            mu_b = mu_b.cuda()
            v_ref1_rl = v_ref1_rl.cuda()
            w_ref1_rl = w_ref1_rl.cuda()
            v_ref2_rl = v_ref2_rl.cuda()
            w_ref2_rl = w_ref2_rl.cuda()
            params = params.cuda()

        angle1 = get_angle(ref_a, ref_b)
        angle2 = get_angle(ref_b, ref_c)

        x1_rl_init = torch.stack([ref_a[0], ref_a[1], angle1, v_ref1_rl, w_ref1_rl])
        x1_rl_list = get_ref_list(x1_rl_init, args.nt, args.dt)

        for ti in range(args.nt):
            s_rl_err1 = world_to_err(curr_s, x1_rl_list[ti:ti + 1])
            dist = car_env.dist_to_seg(curr_s[0, :2], ref_a, ref_b, ref_c, use_torch=True)
            obs = torch.zeros((1, 16)).cuda()
            obs[:, 0:7] = s_rl_err1
            obs[0, 7:] = torch.stack([v_ref1_rl, w_ref1_rl, mu_a, angle1, dist,
                                      angle2, v_ref2_rl, w_ref2_rl, mu_b], dim=-1)
            u = utils.rl_u(obs, rlp_running, rlp_policy, umin=-args.u_limit[0], umax=args.u_limit[1])
            new_s_rl_err, _ = za.next_step(s_rl_err1, u, car_params, params,
                                           False, None, args, detached=False, directed_u=True)
            curr_s = err_to_world(new_s_rl_err, x1_rl_list[ti + 1:ti + 2]).detach()
            cache[me_i]["trajs"][-1].append(curr_s)
            cache[me_i]["refs"][-1].append(x1_rl_list[ti + 1:ti + 2])
            cache[me_i]["segs"][-1][-1].append(curr_s)
    # TODO visualize
    cache[me_i]["trajs"][-1] = torch.cat(cache[me_i]["trajs"][-1], dim=0)
    cache[me_i]["refs"][-1] = torch.cat(cache[me_i]["refs"][-1], dim=0)
    for seg_i in range(len(cache[me_i]["segs"][-1])):
        cache[me_i]["segs"][-1][seg_i] = torch.cat(cache[me_i]["segs"][-1][seg_i], dim=0)

    rl_t2 = time.time()
    cache = compute_metric(cache, curr_env, me_i, args.methods[me_i], trial_i, rl_t2 - rl_t1, cache[me_i]["trajs"][-1].shape[0])
    return cache


def solve_in_clf(trial_i, me_i, init_test_x, cache, curr_env, mono_actor, CAP_SEG, args):
    clf_t1 = time.time()
    curr_s = to_cuda(init_test_x.detach().reshape((1, -1)), args)
    half_time = args.nt * args.dt
    cache[me_i]["trajs"].append([])
    cache[me_i]["refs"].append([])
    cache[me_i]["segs"].append([])
    roads, thetas, mus, new_roads, new_thetas, new_mus = curr_env
    for seg_i in range(min(CAP_SEG * 2, new_roads.shape[0] - 1)):
        cache[me_i]["segs"][-1].append([])
        ref_a, ref_b, ref_c, mu_a, mu_b, v_ref1_clf, v_ref2_clf, w_ref1_clf, w_ref2_clf = \
            get_ref_from_roads(new_roads, new_mus, seg_i, half_time)
        params = torch.stack([mu_a.cpu(), v_ref1_clf, w_ref1_clf, torch.zeros_like(w_ref1_clf)])
        if args.gpus is not None:
            mu_a = mu_a.cuda()
            v_ref1_clf = v_ref1_clf.cuda()
            w_ref1_clf = w_ref1_clf.cuda()
            params = params.cuda()

        angle1 = get_angle(ref_a, ref_b)

        x1_clf_init = torch.stack([ref_a[0], ref_a[1], angle1, v_ref1_clf, w_ref1_clf])
        x1_clf_list = get_ref_list(x1_clf_init, args.nt, args.dt)

        for ti in range(args.nt):
            s_clf_err1 = world_to_err(curr_s, x1_clf_list[ti:ti + 1])
            obs = torch.zeros((1, 10)).cuda()
            obs[:, 0:7] = s_clf_err1
            obs[0, 7:] = torch.stack([mu_a, v_ref1_clf, w_ref1_clf], dim=-1)
            u = mono_actor(obs)
            new_s_clf_err, _ = za.next_step(s_clf_err1, u, car_params, params,
                                           False, None, args, detached=False, directed_u=True)
            curr_s = err_to_world(new_s_clf_err, x1_clf_list[ti + 1:ti + 2]).detach()
            cache[me_i]["trajs"][-1].append(curr_s)
            cache[me_i]["refs"][-1].append(x1_clf_list[ti + 1:ti + 2])
            cache[me_i]["segs"][-1][-1].append(curr_s)

    # TODO visualize
    cache[me_i]["trajs"][-1] = torch.cat(cache[me_i]["trajs"][-1], dim=0)
    cache[me_i]["refs"][-1] = torch.cat(cache[me_i]["refs"][-1], dim=0)

    clf_t2 = time.time()
    cache = compute_metric(cache, curr_env, me_i, args.methods[me_i], trial_i, clf_t2 - clf_t1, cache[me_i]["trajs"][-1].shape[0])
    return cache


def solve_in_lqr(trial_i, me_i, init_test_x, cache, curr_env, CAP_SEG, args):
    ###### LQR2 ######
    lqr_t1 = time.time()
    curr_s = init_test_x.detach()
    if args.gpus is not None:
        curr_s = curr_s.cuda()

    cache[me_i]["trajs"].append([])
    cache[me_i]["refs"].append([])
    cache[me_i]["segs"].append([])

    lqr2_curr_s_stack = []
    lqr2_x1_stack = []
    lqr2_x2_stack = []
    lqr2_x1_real_stack = []
    lqr2_x2_real_stack = []
    roads, thetas, mus, new_roads, new_thetas, new_mus = curr_env
    for seg_i in range(min(CAP_SEG, roads.shape[0] - 1)):
        cache[me_i]["segs"][-1].append([])
        lqr2_curr_s_stack.append(curr_s)

        ref_a = roads[seg_i]
        ref_b = roads[seg_i + 1]
        theta_a = thetas[seg_i]
        theta_b = thetas[seg_i + 1]
        mu_a = mus[seg_i]
        mu_b = mus[seg_i + 1]

        x1_list, x2_list, x1_real_lqr_list, x2_real_lqr_list = \
            plan_a_b_lqr2(curr_s, ref_a, theta_a, mu_a, ref_b, theta_b, mu_b, seg_i, args)

        curr_s = x2_real_lqr_list[-1].detach()
        lqr2_x1_stack.append(x1_list)
        lqr2_x2_stack.append(x2_list)
        lqr2_x1_real_stack.append(x1_real_lqr_list)
        lqr2_x2_real_stack.append(x2_real_lqr_list)
        cache[me_i]["trajs"][-1].append(x1_real_lqr_list)
        cache[me_i]["trajs"][-1].append(x2_real_lqr_list)
        cache[me_i]["segs"][-1].append(x1_real_lqr_list)
        cache[me_i]["segs"][-1].append(x2_real_lqr_list)
        cache[me_i]["refs"][-1].append(x1_list)
        cache[me_i]["refs"][-1].append(x2_list)

    cache[me_i]["trajs"][-1] = torch.cat(cache[me_i]["trajs"][-1], dim=0)
    cache[me_i]["refs"][-1] = torch.cat(cache[me_i]["refs"][-1], dim=0)

    lqr_t2 = time.time()
    cache = compute_metric(cache, curr_env, me_i, args.methods[me_i], trial_i, lqr_t2 - lqr_t1, cache[me_i]["trajs"][-1].shape[0])
    return cache


def solve_in_ours(trial_i, me_i, init_test_x, cache, curr_env, actor, clf, net, CAP_SEG, diff_planner=False, args=None):
    ours_t1 = time.time()
    curr_s = init_test_x.detach()

    clf_curr_s_stack = []
    clf_ref_stack = []
    clf_x1_stack = []
    clf_x2_stack = []
    clf_x1_real_stack = []
    clf_x2_real_stack = []

    cache[me_i]["trajs"].append([])
    cache[me_i]["refs"].append([])
    cache[me_i]["segs"].append([])
    roads, thetas, mus, new_roads, new_thetas, new_mus = curr_env
    ###### NN ######
    for seg_i in range(min(CAP_SEG, roads.shape[0] - 1)):
        # print(seg_i, curr_s)
        dbg_t1 = time.time()
        clf_curr_s_stack.append(curr_s)

        ref_a = roads[seg_i]
        ref_b = roads[seg_i + 1]
        theta_a = thetas[seg_i]
        theta_b = thetas[seg_i + 1]
        mu_a = mus[seg_i]
        mu_b = mus[seg_i + 1]
        dbg_t2 = time.time()
        only_first_half = args.use_middle and seg_i != min(CAP_SEG, roads.shape[0] - 1) - 1
        v_ref1, w_ref1, d_x, d_y, v_ref2, w_ref2, d_th, d_x0, d_y0, d_th0, xp1_tensor, xp2_tensor, x1_list, x2_list, trs_1, trs_2, x1_real_list, x2_real_list = \
            plan_a_b_sample(curr_s, ref_a, theta_a, mu_a, ref_b, theta_b, mu_b, actor, clf, net,
                            args, is_mid=False, mid_ref=new_roads[seg_i * 2 + 1], mid_mu=new_mus[seg_i * 2 + 1],
                            mid_theta=new_thetas[seg_i * 2 + 1], only_first_half=only_first_half, diff_planner=diff_planner)
        dbg_t3 = time.time()
        x1_real_list = x1_real_list.detach()
        x1_list = x1_list.detach()
        cache[me_i]["segs"][-1].append(x1_real_list)
        cache[me_i]["trajs"][-1].append(x1_real_list)
        cache[me_i]["refs"][-1].append(x1_list)
        dbg_t4 = time.time()
        if only_first_half:
            ref_a = new_roads[seg_i * 2 + 1]
            ref_b = new_roads[seg_i * 2 + 3]
            theta_a = new_thetas[seg_i * 2 + 1]
            theta_b = new_thetas[seg_i * 2 + 3]
            mu_a = new_mus[seg_i * 2 + 1]
            mu_b = new_mus[seg_i * 2 + 3]
            mid_ref = roads[seg_i + 1]
            mid_mu = mus[seg_i + 1]
            mid_theta = thetas[seg_i + 1]
            curr_s = x1_real_list[-1].detach()
            clf_ref_stack.append(
                [v_ref1, w_ref1, d_x, d_y, v_ref2, w_ref2, d_th, d_x0, d_y0, d_th0, xp1_tensor, xp2_tensor])
            clf_x1_stack.append(x1_list)
            clf_x1_real_stack.append(x1_real_list)
            v_ref1, w_ref1, d_x, d_y, v_ref2, w_ref2, d_th, d_x0, d_y0, d_th0, xp1_tensor, xp2_tensor, x1_list, x2_list, trs_1, trs_2, x1_real_list, x2_real_list = \
                plan_a_b_sample(curr_s, ref_a, theta_a, mu_a, ref_b, theta_b, mu_b, actor, clf, net,
                                args, is_mid=True, mid_ref=mid_ref, mid_mu=mid_mu, mid_theta=mid_theta,
                                only_first_half=only_first_half, diff_planner=diff_planner)

            clf_ref_stack.append(
                [v_ref1, w_ref1, d_x, d_y, v_ref2, w_ref2, d_th, d_x0, d_y0, d_th0, xp1_tensor, xp2_tensor])
            clf_x2_stack.append(x1_list)
            clf_x2_real_stack.append(x1_real_list)
            curr_s = x1_real_list[-1].detach()

            x1_real_list = x1_real_list.detach()
            x1_list = x1_list.detach()

            cache[me_i]["segs"][-1].append(x1_real_list)
            cache[me_i]["trajs"][-1].append(x1_real_list)
            cache[me_i]["refs"][-1].append(x1_list)

        else:
            curr_s = x2_real_list[-1].detach()
            clf_ref_stack.append(
                [v_ref1, w_ref1, d_x, d_y, v_ref2, w_ref2, d_th, d_x0, d_y0, d_th0, xp1_tensor, xp2_tensor])
            clf_x1_stack.append(x1_list)
            clf_x2_stack.append(x2_list)
            clf_x1_real_stack.append(x1_real_list)
            clf_x2_real_stack.append(x2_real_list)

            x2_real_list = x2_real_list.detach()
            x2_list = x2_list.detach()
            cache[me_i]["segs"][-1].append(x2_real_list)
            cache[me_i]["trajs"][-1].append(x2_real_list)
            cache[me_i]["refs"][-1].append(x2_list)

        dbg_t5 = time.time()

    cache[me_i]["trajs"][-1] = torch.cat(cache[me_i]["trajs"][-1], dim=0)
    cache[me_i]["refs"][-1] = torch.cat(cache[me_i]["refs"][-1], dim=0)

    ours_t2 = time.time()
    cache = compute_metric(cache, curr_env, me_i, args.methods[me_i], trial_i, ours_t2 - ours_t1, cache[me_i]["trajs"][-1].shape[0])
    return cache


def find_before_nan(x):
    idx = torch.where(torch.any(torch.isnan(x), dim=-1))[0]
    if idx.shape[0] != 0:
        return x[:idx[0]]
    else:
        return x


def get_road_len(roads):
    s = 0
    for seg_i, seg in enumerate(roads[:-1]):
        s += torch.norm(roads[seg_i] - roads[seg_i+1])
    return s


def point_to_line_dist(a, b, c):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    area = (1/2) * (x1*y2 - x2*y2 + x1*y1 - x2*y1 - x3*y1 - x1*y1 + x3*y3 - x1*y3 - x3*y2 + x2*y2 - x3*y3 + x2*y3)
    line = ((x2-x3)**2+(y2-y3)**2)**2
    dist = abs(area) * 2 / line
    return dist

def compute_rest_road_length(state, curr_env):
    roads, thetas, mus, new_roads, new_thetas, new_mus = curr_env
    total_s = get_road_len(roads)
    dist_list=[]
    for seg_i, seg in enumerate(roads[:-1]):
        to_road_dist = point_to_line_dist(state[:2], roads[seg_i], roads[seg_i+1])
        dist_list.append(to_road_dist.item())
    idx = np.argmin(np.array(dist_list))
    rest_s = get_road_len(roads[idx+1:].cpu()) + torch.norm(state[:2] - roads[idx:idx+2].cpu())
    return rest_s, rest_s / total_s


def compute_metric(cache, curr_env, me_i, method, trial_i, runtime_t, steps):
    # # RMSE towards reference
    # consider the case when it's already too far from the road
    cache[me_i]["trajs"][trial_i] = cache[me_i]["trajs"][trial_i].detach().cpu()
    cache[me_i]["refs"][trial_i] = cache[me_i]["refs"][trial_i].detach().cpu()
    trajs = cache[me_i]["trajs"][trial_i]
    refs = cache[me_i]["refs"][trial_i]
    trajs = find_before_nan(trajs)
    refs = refs[:trajs.shape[0]]  # TODO cap to non-nan first length
    errs = world_to_err(trajs, refs)

    valids = torch.abs(errs[:, 1]) < args.deviation_thres
    indices = torch.where(valids != True)[0]
    if len(indices) != 0:
        first_idx = indices[0]
        valids = torch.ones_like(valids)
        valids[first_idx:] = 0
        dist_goal, dist_goal_rel = compute_rest_road_length(trajs[first_idx], curr_env)
    else:
        dist_goal, dist_goal_rel = torch.zeros(1), torch.zeros(1)
    valids = valids.float()

    dist_goal_t = 1 - torch.mean(valids)

    masked_rmse = torch.sqrt(utils.mask_mean(torch.norm(errs, dim=-1) ** 2, valids))
    rmse = torch.sqrt(torch.mean(torch.norm(errs, dim=-1) ** 2))
    rmse = torch.clamp(rmse, max=1e4)
    cache[me_i]["rmse"].append(rmse.item())
    cache[me_i]["masked_rmse"].append(masked_rmse.item())

    masked_dev = utils.mask_mean(torch.abs(errs[:, 1]), valids)
    masked_dev = torch.clamp(masked_dev, max=1e4)
    dev = torch.mean(torch.abs(errs[:, 1]))

    cache[me_i]["dev"].append(dev.item())
    cache[me_i]["masked_dev"].append(masked_dev.item())
    cache[me_i]["valid"].append(torch.mean(valids).item())
    cache[me_i]["dist_goal"].append(dist_goal.item())
    cache[me_i]["dist_goal_rel"].append(dist_goal_rel.item())
    cache[me_i]["dist_goal_t"].append(dist_goal_t.item())
    cache[me_i]["time"].append(errs.shape[0])
    cache[me_i]["valid_time"].append(torch.sum(valids).item())
    cache[me_i]["runtime"].append(runtime_t)
    cache[me_i]["runtime_step"].append(runtime_t / steps)

    reward_rmse = utils.mask_mean(torch.norm(errs, dim=-1), valids)
    reward_valid = torch.mean(valids)
    reward = -reward_rmse + reward_valid
    cache[me_i]["reward"].append(reward.item())

    tmp = cache[me_i]
    print("%02d %s rmse:%.3f  m_rmse:%.3f  dev:%.3f  m_dev:%.3f  d:%.3f  d_rel:%.3f  d_t:%.3f  val:%.3f  t:%.3f  run:%.3f  step:%.5f r:%.3f"%(
        trial_i, method, tmp["rmse"][-1], tmp["masked_rmse"][-1],
        tmp["dev"][-1], tmp["masked_dev"][-1],  tmp["dist_goal"][-1],  tmp["dist_goal_rel"][-1],
        tmp["dist_goal_t"][-1],  tmp["valid"][-1],
        tmp["time"][-1], tmp["runtime"][-1], tmp["runtime_step"][-1], tmp["reward"][-1]
    ))
    print(
        "%02d %s rmse:%.3f  m_rmse:%.3f  dev:%.3f  m_dev:%.3f  d:%.3f  d_rel:%.3f  d_t:%.3f  val:%.3f  t:%.3f  run:%.3f  step:%.5f r:%.3f" % (
            trial_i, method, _avg(tmp["rmse"]), _avg(tmp["masked_rmse"]),
            _avg(tmp["dev"]), _avg(tmp["masked_dev"]), _avg(tmp["dist_goal"]), _avg(tmp["dist_goal_rel"]),
            _avg(tmp["dist_goal_t"]), _avg(tmp["valid"]),
            _avg(tmp["time"]), _avg(tmp["runtime"]), _avg(tmp["runtime_step"]), _avg(tmp["reward"][-1])
        ))

    if trial_i > args.max_keep_viz_trials:
        del cache[me_i]["trajs"][trial_i]
        del cache[me_i]["refs"][trial_i]
        cache[me_i]["trajs"].append(None)
        cache[me_i]["refs"].append(None)
    return cache


def _avg(li):
    return np.mean(np.array(li))



def get_ref_from_roads(new_roads, new_mus, seg_i, half_time):
    ref_a = new_roads[seg_i]
    ref_b = new_roads[seg_i + 1]
    if seg_i == new_roads.shape[0] - 2:
        ref_c = ref_b + (ref_b - ref_a)  # TODO extend one side
    else:
        ref_c = new_roads[seg_i + 2]

    if new_mus[seg_i] == 0:
        mu_a = new_mus[seg_i + 1]
        mu_b = new_mus[seg_i + 1]
    else:
        mu_a = new_mus[seg_i]
        if seg_i == new_roads.shape[0] - 2:
            mu_b = new_mus[seg_i]
        else:
            mu_b = new_mus[seg_i + 2]

    v_ref1 = torch.tensor((torch.norm(ref_b - ref_a) / half_time).item())
    w_ref1 = torch.tensor(0.0)
    v_ref2 = torch.tensor((torch.norm(ref_c - ref_b) / half_time).item())
    w_ref2 = torch.tensor(0.0)
    return ref_a, ref_b, ref_c, mu_a, mu_b, v_ref1, v_ref2, w_ref1, w_ref2


def load_nn(key, path, args):
    smart_path = utils.smart_path(path)
    if "models" in smart_path:
        nn_dir = smart_path.split("models")[0]
    else:
        nn_dir = smart_path
    net_args = np.load(ospj(nn_dir, "args.npz"), allow_pickle=True)['args'].item()
    if hasattr(net_args, "normalize")==False:
        net_args.normalize = False
    if hasattr(net_args, "clf_ell_eye")==False:
        net_args.clf_ell_eye = False
    net_func = {"clf_actor": roa_nn.Actor,
                "actor": roa_nn.Actor,
                "clf": roa_nn.CLF,
                "net": roa_nn.Net}
    last_key = {"clf_actor": "actor_",
                "actor": "actor_",
                "clf": "clf_",
                "net": "net_",
                }
    if key == "net":
        net = net_func[key](net_args)
    else:
        net = net_func[key](None, net_args)
    utils.safe_load_nn(net, path, load_last=args.load_last, key=last_key[key])
    if args.gpus is not None:
        net = net.cuda()
    return net, net_args


if __name__ == "__main__":
    args = za.hyperparameters()
    
    # TODO fixed setup args
    if args.u_limit is not None:
        args.tanh_w_gain = args.u_limit[0]
        args.tanh_a_gain = args.u_limit[1]
    if args.joint_pretrained_path is not None:
        args.actor_pretrained_path = args.joint_pretrained_path
        args.clf_pretrained_path = args.joint_pretrained_path
    if args.joint_pretrained_path0 is not None:
        args.actor_pretrained_path0 = args.joint_pretrained_path0
        args.clf_pretrained_path0 = args.joint_pretrained_path0
    if args.joint_pretrained_path1 is not None:
        args.actor_pretrained_path1 = args.joint_pretrained_path1
        args.clf_pretrained_path1 = args.joint_pretrained_path1
    args.actor_clip_u = True
    args.multi = True
    args.load_last = True
    args.clip_angle = True
    args.X_DIM = 7
    args.N_REF = 3  # (mu, vref, wref)
    args.U_DIM = 2  # (omega, accel)
    args.N_CFGS = args.num_samples
    args.controller_dt = args.dt

    if args.from_file:
        data_from_file = utils.parse_from_file(args.from_file)
        args.rlp_paths = data_from_file["rlp_paths"]
        args.ours_actor_paths = data_from_file["actor_paths"]
        args.ours_clf_paths = data_from_file["clf_paths"]
        args.ours_roa_paths = data_from_file["roa_paths"]
        args.methods = data_from_file["methods"]
        args.ds_list = data_from_file["ds_list"]
    else:
        if args.auto_rl:
            args.rlp_paths = utils.gen_rlp_path_list("car", args.methods)
        args.ours_clf_paths = []
        args.ours_actor_paths = []
        args.ours_roa_paths = []
        for me_i, method in enumerate(args.methods):
            if "ours" in method:
                args.ours_clf_paths.append(args.joint_pretrained_path)
                args.ours_actor_paths.append(args.joint_pretrained_path)
                args.ours_roa_paths.append(args.net_pretrained_path)
            else:
                args.ours_clf_paths.append(None)
                args.ours_actor_paths.append(None)
                args.ours_roa_paths.append(None)

    t1=time.time()
    car_params = sscar_utils.VehicleParameters()
    main()
    t2 = time.time()
    print("ROA training finished in %.4fs" % (t2 - t1))
