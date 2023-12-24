import os, sys, time
from os.path import join as ospj
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse

import beta_pogo_sim
from beta_fit import Dyna, preproc_data
from beta_train import next_x_full
from beta_mpc import plan_waypoint_mpc
import beta_utils

sys.path.append("../")
import roa_utils as utils
from roa_utils import to_np, to_item
from roa_nn import CLF, Actor, Net
from beta_env import get_rand_env

import mbrl_utils

def plot_seg_figure(seg_list, args):
    # plot the background
    margin = 0.5
    hmin = np.min(seg_list[:, 1]) - margin
    hmax = np.max(seg_list[:, 2]) + margin
    ax = plt.gca()
    offset = 0
    for seg_ii,seg in enumerate(seg_list):
        seglen = seg[0]
        floor = seg[1]
        ceil = seg[2]

        floor1 = floor
        ceil1 = ceil
        floor2 = floor if seg_ii == len(seg_list) - 1 else seg_list[seg_ii + 1][1]
        ceil2 = ceil if seg_ii == len(seg_list) - 1 else seg_list[seg_ii + 1][2]

        y1_abs = beta_utils.ref_h(max(floor1, floor2), min(ceil1, ceil2))
        # print(floor1, floor2, ceil1, ceil2, y1_abs)
        floor_pts = np.array([[offset, hmin], [offset+seglen, hmin], [offset+seglen, floor], [offset, floor]])
        ceil_pts = np.array([[offset, hmax], [offset + seglen, hmax], [offset + seglen, ceil], [offset, ceil]])
        patch = Polygon(floor_pts, facecolor="saddlebrown", edgecolor=None, alpha=0.5)
        ax.add_patch(patch)
        patch = Polygon(ceil_pts, facecolor="saddlebrown", edgecolor=None, alpha=0.5)
        ax.add_patch(patch)
        offset += seglen
    # exit()
    plt.xlim(0, offset)
    plt.ylim(hmin, hmax)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.savefig("%s/background.png" % (args.viz_dir), bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_simulation(seg_list, xs, whole_xs, ttotal, seg_i, args):
    for ti in range(xs.shape[0]):
        if ti % args.viz_freq == 0:
            plot_frame(seg_list, xs, whole_xs, ttotal, seg_i, args, ti)

def plot_seg(seg_list, first=True):
    ref_size = 48
    fontsize = 18
    legend_fontsize = 12
    margin = 0.5
    hmin = np.min(seg_list[:, 1]) - margin
    hmax = np.max(seg_list[:, 2]) + margin * 2
    ax = plt.gca()
    offset = 0
    for seg_ii, seg in enumerate(seg_list):
        seglen = seg[0]
        floor = seg[1]
        ceil = seg[2]
        floor_pts = np.array([[offset, hmin], [offset + seglen, hmin], [offset + seglen, floor], [offset, floor]])
        ceil_pts = np.array([[offset, hmax], [offset + seglen, hmax], [offset + seglen, ceil], [offset, ceil]])
        patch = Polygon(ceil_pts, facecolor="saddlebrown", edgecolor=None, alpha=0.4, label=None if not first or seg_ii > 0 else "Ceiling")
        ax.add_patch(patch)
        patch = Polygon(floor_pts, facecolor="saddlebrown", edgecolor=None, alpha=0.6, label=None if not first or seg_ii>0 else "Floor")
        ax.add_patch(patch)

        floor1 = floor
        ceil1 = ceil
        floor2 = floor if seg_ii==len(seg_list)-1 else seg_list[seg_ii+1][1]
        ceil2 = ceil if seg_ii==len(seg_list)-1 else seg_list[seg_ii+1][2]

        y1_abs = beta_utils.ref_h(max(floor1, floor2), min(ceil1,ceil2))
        plt.scatter(seglen + offset, y1_abs, color="red", s=ref_size, alpha=0.5, label=None if not first or seg_ii>0 else "Ref. apex state")
        offset += seglen

    plt.xlim(0, offset)
    plt.ylim(hmin, hmax)
    plt.axis("off")
    ax=plt.gca()
    axis = ax.axis()
    rec = plt.Rectangle((axis[0] - 0.0, axis[2] - 0.0), (axis[1] - axis[0]) + 0, (axis[3] - axis[2]) + 0.,
                        fill=False, lw=1, linestyle="-")
    ax.add_patch(rec)
    ax.autoscale_view()



def plot_frame(seg_list, xs, whole_xs, ttotal, seg_i, args, ti, header="", simple=False, method_name=""):
    # plot the background
    pogo_linewidth = 3.0
    traj_linewidth = 2.0
    pogo_head_size = 32
    ref_size = 48
    fontsize = 18
    legend_fontsize = 12

    plt.figure(figsize=(8, 4))

    margin = 0.5
    hmin = np.min(seg_list[:, 1]) - margin
    hmax = np.max(seg_list[:, 2]) + margin * 2
    ax = plt.gca()
    offset = 0
    for seg_ii, seg in enumerate(seg_list):
        seglen = seg[0]
        floor = seg[1]
        ceil = seg[2]
        floor_pts = np.array([[offset, hmin], [offset + seglen, hmin], [offset + seglen, floor], [offset, floor]])
        ceil_pts = np.array([[offset, hmax], [offset + seglen, hmax], [offset + seglen, ceil], [offset, ceil]])
        patch = Polygon(ceil_pts, facecolor="saddlebrown", edgecolor=None, alpha=0.4, label=None if seg_ii > 0 else "Ceiling")
        ax.add_patch(patch)
        patch = Polygon(floor_pts, facecolor="saddlebrown", edgecolor=None, alpha=0.6, label=None if seg_ii>0 else "Floor")
        ax.add_patch(patch)

        floor1 = floor
        ceil1 = ceil
        floor2 = floor if seg_ii==len(seg_list)-1 else seg_list[seg_ii+1][1]
        ceil2 = ceil if seg_ii==len(seg_list)-1 else seg_list[seg_ii+1][2]

        y1_abs = beta_utils.ref_h(max(floor1, floor2), min(ceil1,ceil2))
        # ref_circ = Ellipse(np.array([offset, y1_abs]), width=, height=, facecolor="red", edgecolor=None, alpha=0.5)
        # ax.add_patch(ref_circ)
        plt.scatter(seglen + offset, y1_abs, color="red", s=ref_size, alpha=0.5, label=None if seg_ii>0 else "Ref. apex state")

        offset += seglen

    # plot the bot
    whole_xs_np = to_np(whole_xs)
    xs_np = to_np(xs)
    state = xs_np[ti]  # (x, xdot, y, ydot, xf, yf)

    if simple:
        plt.plot(whole_xs_np[:ti, 0], whole_xs_np[:ti, 2], color="indigo", alpha=0.5, linewidth=traj_linewidth,
                 linestyle="--", label="Trajectory %s"%(method_name))
    else:
        tprev = ttotal - xs_np.shape[0]
        plt.plot(whole_xs_np[:tprev + ti, 0], whole_xs_np[:tprev + ti, 2], color="indigo", alpha=0.5, linewidth=1.0,
             linestyle="--", label="Trajectory")
    plt.plot([state[0], state[4]], [state[2], state[5]], color="brown", linewidth=pogo_linewidth) #, label="Pogobot")
    plt.scatter(state[0], state[2], color="blue", s=pogo_head_size, zorder=99999)
    plt.scatter(state[4], state[5], color="black", s=pogo_head_size, zorder=99999)

    plt.xlim(0, offset)
    plt.ylim(hmin, hmax)
    plt.legend(fontsize=legend_fontsize, loc='upper left')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("x (m)", fontsize=fontsize)
    plt.ylabel("y (m)", fontsize=fontsize)
    plt.title("Simulation t=%.3f s" % (ti*0.001), fontsize=fontsize)
    if args.multi_ani:
        plt.savefig("%s/pogo_%s.png" % (args.viz_dir, header), bbox_inches='tight', pad_inches=0.1)
    else:
        plt.savefig("%s/pogo_sim_%s_%05d.png" % (args.viz_dir, header, ti), bbox_inches='tight', pad_inches=0.1)
    plt.close()


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
        seg_list = map_cache["segs"][trial_i]

        # align
        plt.subplot(aligns[0], aligns[1], trial_i + 1)

        # plot
        plot_seg(seg_list, first=trial_i==0)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    handles, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(handles, labels, loc='upper center', fontsize=legend_fontsize, ncol=3)
    plt.tight_layout(pad=0)
    plt.savefig("%s/pogo_env_demo.png" % (args.exp_dir_full), bbox_inches='tight', pad_inches=0.01)
    plt.close()


def get_env_1():
    return [[5.0, 0.0, 2.5, 1.0],
            [5.0, 0.0, 2.5, 1.0],
            [5.0, 0.0, 4.0, 1.0],
            [5.0, 2.0, 4.0, 1.0],
            [5.0, 1.0, 4.0, 1.0],
            [5.0, 0.0, 2.5, 1.0],
            [5.0, -1.0, 2.5, 0.0]] # (length, floor, height, xdot)


def get_env_2():
    de = 0.75
    return [[5.0, 0.0, 2.0, 0.5],
            [5.0, 0.0 + de, 2.0 + de, 0.5],
            [5.0, 0.0 + de * 2, 2.0 + de * 2, 0.5],
            [5.0, 0.0 + de * 3, 2.0 + de * 3, 0.5],
            [5.0, 0.0 + de * 4, 2.0 + de * 4, 0.5],
            [5.0, 0.0 + de * 5, 2.0 + de * 5, 0.5],
            [5.0, 0.0 + de * 4, 2.0 + de * 4, 0.5],
            [5.0, 0.0 + de * 3, 2.0 + de * 3, 0.5],
            [5.0, 0.0 + de * 2, 2.0 + de * 2, 0.5],
            [5.0, 0.0 + de * 1, 2.0 + de * 1, 0.5],
            [5.0, -0.5, 1.5, 0.0]]  # (length, floor, height, xdot)


def get_env_3():
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

def get_env_1_var(x, de, h, ref_v):
    return [[x, 0.0, h, ref_v],
            [x, 0.0, h, ref_v],
            [x, 0.0, h, ref_v]]

def get_env_2_var(x, de, h, ref_v):
    return [[x, 0.0, h, ref_v],
            [x, 0.0, h + 1.5 * de, ref_v],
            [x, 2 * de, h + 1.5 * de, ref_v * 1.5]]

def get_env_3_var(x, de, h, ref_v):
    return [[x, 0.0, h, ref_v],
            [x, 0.0, h + 1.5 * de, ref_v],
            [0.6*x, 2 * de, h + 1.5 * de, ref_v * 1.5],
            [x, 0.0, h + 1.5 * de, ref_v],
            [x, 0.0, h, ref_v]]

def get_random_env():
    # seg_list = np.array(get_env_1())
    # seg_list = np.array(get_env_2())
    # seg_list = np.array(get_env_3())
    if args.animation:
        seg_list = np.array(get_env_3())
    else:
        seg_list = get_rand_env()
        num_try=0
        while check_env(seg_list)==False:
            print(num_try, "regenerate valid environment!")
            num_try+=1
            seg_list = get_rand_env()
    return seg_list

def check_env(seg_list):
    for seg_i in range(seg_list.shape[0]):
        floor0, ceil0 = seg_list[seg_i][1], seg_list[seg_i][2]
        if seg_i == seg_list.shape[0] - 1:
            floor1, ceil1 = seg_list[seg_i][1], seg_list[seg_i][2]
        else:
            floor1, ceil1 = seg_list[seg_i+1][1], seg_list[seg_i+1][2]
        mid_y_abs_min = max(floor0, floor1) + 1.0 + args.bloat_factor
        mid_y_abs_max = min(ceil0, ceil1) - args.bloat_factor
        if mid_y_abs_min >= mid_y_abs_max:
            return False
    return True


def update_cache(cache, me_i, method):
    cache[me_i]["trajs"].append([])
    cache[me_i]["modes"].append([])
    cache[me_i]["refs"].append([])
    cache[me_i]["xc_list"].append([])
    cache[me_i]["mc_list"].append([])
    return cache


def solve_in_rl(trial_i, me_i, x_sim_ori, cache, map_cache, rlp_pair, global_vars, mbpo=False):
    rl_t1 = time.time()
    total_runtime = 0
    x_sim = x_sim_ori.detach()
    base_t = 0
    xc_list_list = []
    mc_list_list = []
    becomes_invalid = False

    seg_list = map_cache["segs"][trial_i]
    if mbpo:
        mbpo_net = rlp_pair
    else:
        rlp_policy, rlp_running = rlp_pair

    for hop_i in range(args.num_sim_hops):
        next_offset, seg_i, seg0, seg1 = get_offset_seg0_seg1(x_sim, seg_list)
        prev_offset = next_offset - seg0[0]
        base_floor = seg0[1]
        curr_floor = seg0[1]
        curr_ceil = seg0[2]
        curr_speed = seg0[3]
        modes = torch.zeros(1, 1)

        x_obs = torch.tensor([[
            x_sim[0, 0] - prev_offset,
            x_sim[0, 1], x_sim[0, 2] - base_floor, x_sim[0, 3],
            x_sim[0, 4] - prev_offset, x_sim[0, 5] - base_floor,
            modes[0, 0],
            seg0[0], seg0[1] - base_floor, seg0[2] - base_floor, seg0[3],
            seg1[0], seg1[1] - base_floor, seg1[2] - base_floor, seg1[3]
        ]]).cuda()

        if x_obs[0, 2] < 0 or x_obs[0, 2] > seg0[2] - seg0[1]:
            # print("x_obs", x_obs, x_sim, seg0, seg1)
            becomes_invalid = True
            break

        xc_list = []
        mc_list = []
        if mbpo:
            u = utils.rl_u(x_obs, None, mbpo_net, mbpo=True)
        else:
            u = utils.rl_u(x_obs, rlp_running, rlp_policy)

        u[:, 0] = torch.clamp(u[:, 0], -0.5, 0.5)
        # u[:, 1] = torch.clamp(u[:, 1] * 5000, -5000, 5000)
        u[:, 1] = torch.clamp(u[:, 1], -5000, 5000)
        # print("RL: Action_i=%d | %.3f %.3f" % (hop_i, u[0, 0], u[0, 1]))
        x_card = world_to_card(x_sim, offset=0, floor=seg0[1]).detach()
        for ti in range(args.nt):
            next_x_card, next_modes = beta_pogo_sim.pytorch_sim_single(
                x_card, modes, u, global_vars, check_invalid=True)

            # TODO handle jump switch case
            if x_card[0, 4] < next_offset and next_x_card[0, 4] >= next_offset:
                next_x_card[0, 2] = next_x_card[0, 2] + seg0[1] - seg1[1]
                next_x_card[0, 5] = next_x_card[0, 2] + seg0[1] - seg1[1]
                curr_floor = seg1[1]
                curr_ceil = seg1[2]
                curr_speed = seg1[3]
            xc_list.append(card_to_world(next_x_card, 0, curr_floor).detach())
            mc_list.append(next_modes.detach())

            x_card, modes = next_x_card.detach(), next_modes.detach()
            if modes[0, 0] == 7:
                break
            if modes[0, 0] == -1 or ti == args.nt - 1:  # comes to invalid
                becomes_invalid = True
                break

        x_sim = xc_list[-1].detach()
        modes = mc_list[-1].detach()

        xc_list = torch.cat(xc_list, dim=0)
        mc_list = torch.cat(mc_list, dim=0)
        xc_list_list.append(xc_list)
        mc_list_list.append(mc_list)

        cache[me_i]["xc_list"][trial_i].append(xc_list.cpu())
        cache[me_i]["mc_list"][trial_i].append(mc_list.cpu())

        if becomes_invalid:
            break

    cache[me_i]["trajs"][trial_i] = cat_all(xc_list_list).cpu()
    cache[me_i]["modes"][trial_i] = cat_all(mc_list_list).cpu()
    rl_t2 = time.time()
    cache[me_i]["runtime"].append(rl_t2 - rl_t1)
    cache[me_i]["runtime_step"].append((rl_t2 - rl_t1) / cache[me_i]["trajs"][trial_i].shape[0])

    return cache


def solve_in_mpc(trial_i, me_i, x_sim_ori, cache, map_cache, total_env_len, NO_CUDA, global_vars):
    total_runtime = 0
    x_sim = x_sim_ori.detach()
    base_t = 0
    xc_list_list = []
    mc_list_list = []
    becomes_invalid = False
    mpc_t1 = time.time()
    seg_list = map_cache["segs"][trial_i]
    for hop_i in range(args.num_sim_hops):
        # TODO find seg_i
        mpc_offset, seg_i, seg0, seg1 = get_offset_seg0_seg1(x_sim, seg_list)

        # TODO plan u
        dbg_t0 = time.time()
        x_sim_mpc = get_x_relative_add_mode(x_sim, seg0)
        extra_vars = mpc_offset, seg0[1], seg0[2], seg0[3], seg1[1], seg1[2], seg1[3]
        u_sim = plan_waypoint_mpc(x_sim_mpc, None, None, None, global_vars, extra_vars, args)
        dbg_t1 = time.time()
        # print("MPC solved in %.3f seconds" % (dbg_t1 - dbg_t0))
        total_runtime += dbg_t1 - dbg_t0

        # TODO convert x_sim to x_card, modes
        curr_floor = seg0[1]
        curr_ceil = seg0[2]
        curr_speed = seg0[3]
        x_card = world_to_card(x_sim, 0, curr_floor)
        modes = torch.zeros(1, 1)

        if NO_CUDA:
            x_card = x_card.cpu()
            modes = modes.cpu()
            u_sim = u_sim.cpu()

        xc_list = []
        mc_list = []

        # TODO simulation
        for ti in range(args.nt):
            next_x_card, next_modes = beta_pogo_sim.pytorch_sim_single(
                x_card, modes, u_sim, global_vars, check_invalid=True)

            # TODO handle jump switch case
            if x_card[0, 4] < mpc_offset and next_x_card[0, 4] >= mpc_offset:
                next_x_card[0, 2] = next_x_card[0, 2] + seg0[1] - seg1[1]
                next_x_card[0, 5] = next_x_card[0, 2] + seg0[1] - seg1[1]
                curr_floor = seg1[1]
                curr_ceil = seg1[2]
                curr_speed = seg1[3]

            xc_list.append(card_to_world(next_x_card, 0, curr_floor))
            mc_list.append(next_modes)

            x_card, modes = next_x_card, next_modes
            if modes[0] == 7:  # comes to the apex point
                break
            if modes[0, 0] == -1 or ti == args.nt - 1:  # comes to invalid
                becomes_invalid = True
                break

        xc_list = torch.cat(xc_list, dim=0)
        mc_list = torch.cat(mc_list, dim=0)
        xc_list_list.append(xc_list)
        mc_list_list.append(mc_list)

        cache[me_i]["xc_list"][trial_i].append(xc_list.cpu())
        cache[me_i]["mc_list"][trial_i].append(mc_list.cpu())

        # TODO update the x_sim
        x_sim = xc_list[-1:]
        if next_x_card[0, 0] >= total_env_len:
            break

    cache[me_i]["trajs"][trial_i] = cat_all(xc_list_list).cpu()
    cache[me_i]["modes"][trial_i] = cat_all(mc_list_list).cpu()

    mpc_t2 = time.time()
    cache[me_i]["runtime"].append(mpc_t2 - mpc_t1)
    cache[me_i]["runtime_step"].append((mpc_t2 - mpc_t1) / cache[me_i]["trajs"][trial_i].shape[0])
    return cache


def solve_in_ours(trial_i, me_i, x_sim_ori, cache, map_cache, net, clf, actor, roa_est, method, NO_CUDA, global_vars):
    ours_t1 = time.time()
    x_sim = x_sim_ori.detach()
    offset = 0
    total_runtime = 0
    gt_x_segs = []
    gt_m_segs = []
    seg_list = map_cache["segs"][trial_i]

    if "-d" in method:
        diff_planner = True
    else:
        diff_planner = False

    for seg_i in range(len(seg_list)):
        dbg_tt0 = time.time()
        # plan for waypoint
        wpt_xd, wpt_y, wpt_xd_abs, wpt_y_abs = \
            plan_waypoint(x_sim, offset, seg_list, net, clf, actor, roa_est, seg_i, diff_planner, args)
        dbg_tt1 = time.time()
        # print("Planner took %.4f seconds" % (dbg_tt1 - dbg_tt0))
        total_runtime += dbg_tt1 - dbg_tt0

        # save and execute (via real sim)
        xc_list_list = []
        mc_list_list = []
        gt_x_segs.append(xc_list_list)
        gt_m_segs.append(mc_list_list)

        floor_i = seg_list[seg_i][1]

        x_card = world_to_card(x_sim, offset=offset, floor=floor_i)

        # multiple hops
        dbg_t0 = time.time()
        for hop_i in range(args.num_sim_hops):
            xc_list = []
            mc_list = []
            modes = torch.zeros(1, 1).cuda()
            x_ref = card_to_ref(x_card, xd_ref=wpt_xd, y_ref=wpt_y).detach()
            if NO_CUDA:
                x_ref = x_ref.cuda()
            u_sim = actor(x_ref.detach()).detach()

            if NO_CUDA:
                x_card = x_card.cpu()
                modes = modes.cpu()
                u_sim = u_sim.cpu()

            for ti in range(args.nt):
                # next_x_card, next_modes = beta_pogo_sim.pytorch_sim(
                #     x_card, modes, u_sim, global_vars, check_invalid=True)
                next_x_card, next_modes = beta_pogo_sim.pytorch_sim_single(
                    x_card, modes, u_sim, global_vars, check_invalid=True)
                xc_list.append(next_x_card)
                mc_list.append(next_modes)
                x_card, modes = next_x_card, next_modes
                if modes[0] == 7:  # comes to the apex point
                    break

                if modes[0] == -1:
                    break
            # convert xc_list -> world frame
            xc_list = torch.cat(xc_list, dim=0)
            mc_list = torch.cat(mc_list, dim=0)
            xc_list = card_to_world(xc_list, offset=offset, floor=floor_i)
            xc_list_list.append(xc_list)
            mc_list_list.append(mc_list)
            cache[me_i]["xc_list"][trial_i].append(xc_list.cpu())
            cache[me_i]["mc_list"][trial_i].append(mc_list.cpu())

            if modes[0] == -1:
                break

            # if next apex state across the switching surface
            if x_card[0, 0] > seg_list[seg_i][0]:
                # if next down state across the switching surface
                down_across = False
                for tmp_i in range(xc_list.shape[0]):
                    if mc_list[tmp_i] == 5 and xc_list[tmp_i, 0] > offset + seg_list[seg_i][0]:
                        # print("down across x=%.4f and switching=%.4f" % (xc_list[tmp_i, 0], offset + seg_list[seg_i][0]))
                        down_across = True
                if down_across:
                    del xc_list_list[-1]
                    del mc_list_list[-1]
                    del cache[me_i]["xc_list"][trial_i][-1]
                    del cache[me_i]["mc_list"][trial_i][-1]
                break  # break from hop

        # assert hop_i < args.num_sim_hops - 1
        dbg_t1 = time.time()
        x_sim = xc_list_list[-1][-1:].detach()
        # vis?
        offset = offset + seg_list[seg_i][0]

    cache[me_i]["trajs"][trial_i] = cat_all_all(gt_x_segs).cpu()
    cache[me_i]["modes"][trial_i] = cat_all_all(gt_m_segs).cpu()

    ours_t2 = time.time()
    cache[me_i]["runtime"].append(ours_t2 - ours_t1)
    cache[me_i]["runtime_step"].append((ours_t2 - ours_t1) / cache[me_i]["trajs"][trial_i].shape[0])

    return cache


def main():
    global args
    if args.load_from is not None:
        old_args = np.load(ospj(args.load_from, "args.npz"), allow_pickle=True)['args'].item()
        old_args.load_from = args.load_from
        old_args.use_d = args.use_d
        old_args.animation = args.animation
        old_args.from_file = args.from_file
        old_args.multi_ani = args.multi_ani
        old_args.select_me = args.select_me
        old_args.plot_map_only = args.plot_map_only
        old_args.exp_dir_full = utils.get_exp_dir() + old_args.exp_dir_full.split("/")[-1]
        old_args.viz_dir = utils.get_exp_dir() + "/" + old_args.viz_dir.split("/")[-2] + "/" + old_args.viz_dir.split("/")[-1]
        args = old_args
    else:
        utils.set_seed_and_exp(args, offset=1)

        # TODO setup simulation configs
        num_steps, dt, dt2, l0, g, k, M = args.nt, args.dt1, args.dt2, args.l0, args.g, args.k, args.M,
        global_vars = num_steps, dt, dt2, l0, g, k, M

        map_cache = {"segs": []}
        cache = {
            me_i: {
                "trajs": [],
                "modes": [],
                "refs": [],
                "xc_list": [],
                "mc_list": [],
                "v_err": [],
                "x_len": [],
                "goal_len": [],
                "x_ratio": [],
                "goal_ratio": [],
                "hit": [],
                "succ": [],
                "runtime": [],
                "runtime_m": [],
                "runtime_step": [],
                "reward": [],
            } for me_i, method in enumerate(args.methods)
        }

        # TODO load the data stats & networks
        if ("ours" in args.methods or "ours-d" in args.methods) and args.data_path is not None:
            data_path = utils.smart_path(args.data_path) + "/data.npz"
            train_x_u, test_x_u, x_u_mean, x_u_std, train_y, test_y, y_mean, y_std = preproc_data(data_path, split_ratio=args.split_ratio)
            np.savez("%s/data_mean_std.npz" % args.model_dir, x_u_mean=x_u_mean, x_u_std=x_u_std, y_mean=y_mean,
                     y_std=y_std)
            args.in_means = x_u_mean
            args.in_stds = x_u_std
            args.out_means = y_mean
            args.out_stds = y_std
            net, _ = build_net("dyna", args.dyna_pretrained_path, args, normalize=True)

        actor_list = []
        clf_list = []
        roa_est_list = []
        mbpo_list = []
        pets_list = []
        for me_i, method in enumerate(args.methods):
            actor_tmp, clf_tmp, roa_est_tmp, mbpo_tmp, pets_tmp = None, None, None, None, None
            if "ours" in method:
                actor_tmp, _ = build_net("actor", args.ours_actor_paths[me_i], args, normalize=True)
                clf_tmp, _ = build_net("clf", args.ours_clf_paths[me_i], args, normalize=True)
                roa_est_tmp, _ = build_net("net", args.ours_roa_paths[me_i], args, normalize=True)
            elif "mbpo" in method:
                mbpo_tmp = mbrl_utils.get_mbpo_models(args.mbpo_paths[me_i], args)
                
            actor_list.append(actor_tmp)
            clf_list.append(clf_tmp)
            roa_est_list.append(roa_est_tmp)
            mbpo_list.append(mbpo_tmp)
            pets_list.append(pets_tmp)

        rlp_list = utils.get_loaded_rl_list(args.rlp_paths, args.methods, args.auto_rl)

    if args.load_from is None:
        for trial_i in range(args.num_trials):
            # TODO define the environment [(length, floor, height, xdot)]
            print("TRIAL %03d" % (trial_i))
            seg_list = get_random_env()

            for me_i, method in enumerate(args.methods):
                cache = update_cache(cache, me_i, method)

            map_cache["segs"].append(seg_list)

            if args.plot_map_only:
                continue

            total_env_len = np.sum(seg_list[:, 0])
            # plot the seg
            plot_seg_figure(seg_list, args)

            ####################################################
            # TODO experiments starting here
            # TODO plan for reference apex velocity, and height (xdot, y)
            NO_CUDA = True
            x_sim_ori = torch.zeros(1, 6).cuda()  # (x, xdot, y, ydot, xfoot, yfoot)
            x_sim_ori[:, 1] = seg_list[0][-1] + 0.5  # faster than first seg refV
            x_sim_ori[:, 2] = beta_utils.ref_h(seg_list[0][1], seg_list[0][2])
            x_sim_ori[:, 5] = x_sim_ori[:, 2] - 1.0

            for me_i, method in enumerate(args.methods):
                if "rl" in method:
                    cache = solve_in_rl(trial_i, me_i, x_sim_ori, cache, map_cache, rlp_list[me_i], global_vars)
                elif "mbpo" in method:
                    cache = solve_in_rl(trial_i, me_i, x_sim_ori, cache, map_cache, mbpo_list[me_i], global_vars, mbpo=True)
                elif "mpc" in method:
                    cache = solve_in_mpc(trial_i, me_i, x_sim_ori, cache, map_cache, total_env_len, NO_CUDA, global_vars)
                elif "ours" in method:
                    cache = solve_in_ours(trial_i, me_i, x_sim_ori, cache, map_cache, net,
                                          clf_list[me_i], actor_list[me_i], roa_est_list[me_i], method, NO_CUDA, global_vars)
                else:
                    raise NotImplementedError
                cache = compute_metric(trial_i, me_i, method, cache, seg_list, total_env_len, args)

        np.savez("%s/map_cache.npz"%(args.exp_dir_full), data=map_cache)
        np.savez("%s/cache.npz"%(args.exp_dir_full), data=cache)
    else:
        cache = np.load(ospj(args.load_from, "cache.npz"), allow_pickle=True)['data'].item()
        map_cache = np.load(ospj(args.load_from, "map_cache.npz"), allow_pickle=True)['data'].item()

    if args.plot_map_only:
        plot_map(map_cache, args)
        return

    if args.from_file:
        utils.gen_plot_rl_std_curve(
            args.ds_list, args.methods, cache, "%s/std_pogo_curve.png" % (args.exp_dir_full), loc="upper left")
        return

    # TODO bar plots
    if args.animation==False and args.multi_ani==False:
        cs_d = utils.get_cs_d()
        for metric in ["v_err", "x_len", "x_ratio", "goal_len", "goal_ratio", "hit", "succ", "runtime", "runtime_m", "runtime_step"]:
            utils.metric_bar_plot(cache, metric, args, header="_pogo", index=True, cs_d=cs_d, mode="pogo", rl_merged=True)
            utils.metric_bar_plot(cache, metric, args, header="_pogo", index=True, cs_d=cs_d, mode="pogo", rl_merged=False)


    # TODO visualize
    if args.multi_ani or args.load_from is None:
        if args.multi_ani and args.select_me is not None:
            args.methods = [args.methods[me_i] for me_i in args.select_me]
            cache = {ii: cache[me_i] for ii, me_i in enumerate(args.select_me)}
            print(args.methods)
            assert args.methods[-1] == "ours-d"
            args.methods[-1] = "ours"

        for me_i, method in enumerate(args.methods):
            if args.multi_ani:
                for trial_i in range(len(map_cache["segs"])):
                    the_trial_i = trial_i
                    trajs = cache[me_i]["trajs"][the_trial_i]
                    seg_list = map_cache["segs"][the_trial_i]
                    seg_i = 0
                    val_mask = np.logical_and(trajs[:, 0] > 0, trajs[:, 0] < np.sum(seg_list[:,0]))
                    val_idx = np.where(val_mask)
                    tmax = val_idx[0][-1]
                    ti = tmax

                    print(the_trial_i)
                    plot_frame(seg_list, trajs, trajs, trajs.shape[0], seg_i, args, ti,
                           header=method + str(me_i) + "_" + str(the_trial_i), simple=True, method_name=method)
            else:
                the_trial_i = 0
                trajs = cache[me_i]["trajs"][the_trial_i]
                seg_list = map_cache["segs"][the_trial_i]
                seg_i = 0
                for ti in range(trajs.shape[0]):
                    if ti % args.viz_freq == 0:
                        plot_frame(seg_list, trajs, trajs, trajs.shape[0], seg_i, args, ti,
                                   header=method+str(me_i)+"_"+str(the_trial_i), simple=True, method_name=method)



def compute_metric(trial_i, me_i, method, cache, seg_list, total_env_len, args):
    cac = cache[me_i]
    # find the ref
    trajs = cac["trajs"][trial_i]
    modes = cac["modes"][trial_i]
    refs = find_ref(trajs, seg_list)
    cac["refs"][trial_i] = refs

    # evaluate (check speed only in flight mode, mode=0,1,)
    v_err = utils.mask_mean(torch.abs(trajs[:, 1] - refs[:, 2]), is_flight_mode(modes)[:, 0]).item()

    all_hits = torch.logical_or(trajs[:, 2] > refs[:, 1], trajs[:, 2] < refs[:, 0])
    indices = torch.where(all_hits)[0]
    if len(indices) != 0:
        first_idx = indices[0]
        if first_idx == 0:
            first_idx +=1
    else:
        first_idx = all_hits.shape[0] - 1

    x_len = torch.max(trajs[:first_idx, 0]).item()
    goal_len = max(total_env_len - x_len, 0)

    hit = torch.sum(all_hits.float()).item()
    hit = hit / trajs.shape[0]
    succ = float(x_len > np.sum(seg_list[:-1, 0]) and hit < 0.001)

    x_ratio = min(1, x_len / (total_env_len))
    goal_ratio = max(0, goal_len / (total_env_len))

    cac["v_err"].append(v_err)
    cac["x_len"].append(x_len)
    cac["x_ratio"].append(x_ratio)
    cac["goal_len"].append(goal_len)
    cac["goal_ratio"].append(goal_ratio)
    cac["hit"].append(hit)
    cac["succ"].append(succ)
    cac["runtime_m"].append(cac["runtime"][-1] / max(x_ratio, 0.01))

    reward = x_len - v_err - 10 * hit
    cac["reward"].append(reward)

    print(
        "### %5s v_e:%.3f(%.3f) x:%.3f(%.3f) xr:%.3f(%.3f) g:%.3f(%.3f) gr:%.3f(%.3f) hit:%.5f(%.5f) suc:%.3f(%.3f) "
        "t:%.3f(%.3f) tme:%.3f(%.3f) step:%.5f(%.5f) r:%.2f(%.2f)" % (
            method, cac["v_err"][-1], np.mean(cac["v_err"]),
            cac["x_len"][-1], np.mean(cac["x_len"]),
            cac["x_ratio"][-1], np.mean(cac["x_ratio"]),
            cac["goal_len"][-1], np.mean(cac["goal_len"]),
            cac["goal_ratio"][-1], np.mean(cac["goal_ratio"]),
            cac["hit"][-1], np.mean(cac["hit"]),
            cac["succ"][-1], np.mean(cac["succ"]),
            cac["runtime"][-1], np.mean(cac["runtime"]),
            cac["runtime_m"][-1], np.mean(cac["runtime_m"]),
            cac["runtime_step"][-1], np.mean(cac["runtime_step"]),
            cac["reward"][-1], np.mean(cac["reward"]),
        ))
    return cache


def is_flight_mode(modes):
    return torch.logical_or(
        torch.logical_or(
            modes == 0, modes == 1
        ),
        torch.logical_or(
            modes == 6, modes == 7
        ),
    )


def is_stance_mode(modes):
    return torch.logical_or(
        torch.logical_or(
            modes == 2, modes == 3
        ),
        torch.logical_or(
            modes == 4, modes == 5
        ),
    )


def find_ref(trajs, segs):
    ref_list=[]
    for i in range(trajs.shape[0]):
        offset, seg_i, seg0, seg1 = get_offset_seg0_seg1(trajs[i:i+1], segs)
        curr_floor = seg0[1]
        curr_ceil = seg0[2]
        curr_speed = seg0[3]
        ref_list.append(torch.tensor([[curr_floor, curr_ceil, curr_speed]]).float())
    return cat_all(ref_list)


def get_x_relative_add_mode(x_original, seg0):
    x = torch.zeros([1, 7])  # TODO mode bit
    x[0, 0:6] = x_original[0, 0:6].detach()
    x[0, 2] = x[0, 2] - seg0[1]
    x[0, 5] = x[0, 5] - seg0[1]
    return x


def get_offset_seg0_seg1(x_sim, seg_list):
    offset = 0
    x = x_sim[0, 0]
    for seg_i in range(len(seg_list)):
        offset += seg_list[seg_i][0]
        if x < offset:
            break
    #assert x < offset
    if x > offset:
        offset = x
        seg_i = len(seg_list) - 1

    seg0 = seg_list[seg_i]
    if seg_i == len(seg_list) - 1:
        seg1 = seg0
    else:
        seg1 = seg_list[seg_i + 1]

    return offset, seg_i, seg0, seg1


def cat_all(list_list):
    return torch.cat(list_list, dim=0)


def cat_all_all(list_list_list):
    flat_list=[]
    for list_list in list_list_list:
        for l in list_list:
            flat_list.append(l)
    return torch.cat(flat_list, dim=0)


def world_to_card(x_abs, offset, floor):
    return torch.stack([
        x_abs[:, 0] - offset,
        x_abs[:, 1],
        x_abs[:, 2] - floor,
        x_abs[:, 3],
        x_abs[:, 4] - offset,
        x_abs[:, 5] - floor,
    ], dim=-1)


def card_to_world(x_card, offset, floor):
    return torch.stack([
        x_card[:, 0] + offset,
        x_card[:, 1],
        x_card[:, 2] + floor,
        x_card[:, 3],
        x_card[:, 4] + offset,
        x_card[:, 5] + floor,
    ], dim=-1)


def card_to_ref(x_card, xd_ref, y_ref):
    return torch.stack([
        x_card[:, 1],
        x_card[:, 2],
        torch.ones_like(x_card[:, 4]) * xd_ref,
        torch.ones_like(x_card[:, 5]) * y_ref,
    ], dim=-1)


def plan_waypoint(gt_ori_x, offset, seg_list, net, clf, actor, roa_est, seg_i, diff_planner, args):
    dbg_ttt1=time.time()
    # TODO make sure the sampled state can land into RoA(target state)
    # more accurate: use init state ~ hops ~> near sample state
    num_grids = 50
    L0, floor0, ceil0, vref0 = seg_list[seg_i]
    if seg_i != len(seg_list)-1:
        L1, floor1, ceil1, vref1 = seg_list[seg_i+1]
    else:
        L1, floor1, ceil1, vref1 = seg_list[seg_i]

    # TODO initial relative state (in seg_i frame, consider the distance & space on the left side, and right side)
    x0_abs = to_item(gt_ori_x[0, 0])
    xd0_abs = to_item(gt_ori_x[0, 1])
    y0_abs = to_item(gt_ori_x[0, 2])
    L_right = offset + L0 - x0_abs  # how far left in seg_i
    xd0 = xd0_abs
    y0 = y0_abs - floor0

    # TODO target state (in seg_i+1 frame)
    xd1 = vref1
    y1_abs = beta_utils.ref_h(floor1, ceil1)
    y1 = y1_abs - floor1

    # TODO to-sample mid waypoint (in seg_i frame, on the edge between seg_i, seg_i+1)
    mid_y_abs_min = max(floor0, floor1) + 1.0 + args.bloat_factor
    mid_y_abs_max = min(ceil0, ceil1) - args.bloat_factor
    assert mid_y_abs_min <= mid_y_abs_max

    x_self = torch.tensor([xd0, y0]).float().unsqueeze(0).repeat(num_grids * num_grids, 1)

    if diff_planner:
        lr = 0.0005
        n_iters=10
        x_self = x_self.cuda()
        mid_xdots = torch.linspace(min(xd0, xd1), max(xd0, xd1), num_grids).cuda().requires_grad_()
        mid_ys = (torch.linspace(mid_y_abs_min, mid_y_abs_max, num_grids) - floor0).cuda().requires_grad_()

        optimizer = torch.optim.RMSprop([mid_xdots, mid_ys], lr=lr)
        for sgd_i in range(n_iters):
            sample_mesh = torch.stack(torch.meshgrid(mid_xdots, mid_ys), dim=-1)
            x_refs = sample_mesh.reshape((num_grids * num_grids, 2))
            init_x = torch.cat([torch.zeros_like(x_self[:, 0:1]), x_self, x_refs], dim=-1)  # (ng * ng, 5)

            # TODO evaluate by the controller, and the dynamics
            xfull_list = [init_x]
            for hop_i in range(5):
                next_x = next_x_full(xfull_list[-1], actor, net, args)
                xfull_list.append(next_x)
            xfull_list = torch.stack(xfull_list, dim=1)  # (ng*ng, nhop+1, 5)
            dL, dL_idx = torch.min(torch.abs(xfull_list[:, :, 0] - L_right), dim=1)
            xfull_new_list = []
            for i in range(xfull_list.shape[0]):
                xfull_new_list.append(xfull_list[i, dL_idx[i], 1:5])
            xfulls = torch.stack(xfull_new_list, dim=0)
            # TODO new states at edge in frame seg_i+1
            xfs1 = torch.stack(
                [xfulls[:, 0],  # xd_future
                 xfulls[:, 1] + floor0 - floor1,  # y_future (in frame seg_i+1)
                 torch.ones_like(xfulls[:, 2]) * xd1,  # ref_xd
                 torch.ones_like(xfulls[:, 3]) * y1  # ref_y (in frame seg_i+1)
                 ], dim=-1)

            # TODO RoA penalty
            v1 = clf(xfs1)
            c1 = roa_est(xfs1[:, 2:])
            penalty = torch.relu(v1 - c1) * 1000 + torch.abs(x_refs[:, 1:2] + floor0 - y1_abs.item())

            loss = torch.min(penalty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss, loss_idx = torch.min(penalty, dim=0)

    else:
        mid_xdots = np.linspace(min(xd0, xd1), max(xd0, xd1), num_grids)
        mid_ys = np.linspace(mid_y_abs_min, mid_y_abs_max, num_grids) - floor0

        # TODO construct the tensor for heading towards the middle point
        sample_mesh = np.meshgrid(mid_xdots, mid_ys)
        x_refs = torch.from_numpy(np.stack(sample_mesh, axis=-1).reshape((num_grids * num_grids, 2))).float()
        init_x = torch.cat([torch.zeros_like(x_self[:, 0:1]), x_self, x_refs], dim=-1)  # (ng * ng, 5)
        if args.gpus is not None:
            x_refs = x_refs.cuda()
            init_x = init_x.cuda()

        # TODO evaluate by the controller, and the dynamics
        xfull_list = [init_x]
        for hop_i in range(5):
            next_x = next_x_full(xfull_list[-1], actor, net, args)
            xfull_list.append(next_x)
        xfull_list = torch.stack(xfull_list, dim=1)  # (ng*ng, nhop+1, 5)
        dL, dL_idx = torch.min(torch.abs(xfull_list[:, :, 0] - L_right), dim=1)
        xfull_new_list = []
        for i in range(xfull_list.shape[0]):
            xfull_new_list.append(xfull_list[i, dL_idx[i], 1:5])
        xfulls = torch.stack(xfull_new_list, dim=0)

        # TODO new states at edge in frame seg_i+1
        xfs1 = torch.stack(
            [xfulls[:, 0],  # xd_future
             xfulls[:, 1] + floor0 - floor1,  # y_future (in frame seg_i+1)
             torch.ones_like(xfulls[:, 2]) * xd1,  # ref_xd
             torch.ones_like(xfulls[:, 3]) * y1   # ref_y (in frame seg_i+1)
             ], dim=-1)

        # TODO RoA penalty
        v1 = clf(xfs1)
        c1 = roa_est(xfs1[:, 2:])
        penalty = torch.relu(v1 - c1) * 1000 + torch.abs(x_refs[:, 1:2] + floor0 - y1_abs.item())
        loss, loss_idx = torch.min(penalty, dim=0)

    # TODO update the reference, and the state info
    wpt_xd = to_item(init_x[loss_idx, 3])
    wpt_y = to_item(init_x[loss_idx, 4])
    wpt_xd_abs = wpt_xd
    wpt_y_abs = wpt_y + floor0
    dbg_ttt2 = time.time()
    return wpt_xd, wpt_y, wpt_xd_abs, wpt_y_abs


def build_net(net_name, path, args, normalize=False):
    smart_path = utils.smart_path(path)
    if "models" in smart_path:
        nn_dir = smart_path.split("models")[0]
    else:
        nn_dir = smart_path
    net_args = np.load(ospj(nn_dir, "args.npz"), allow_pickle=True)['args'].item()

    if normalize:
        net_args.in_means = args.in_means
        net_args.in_stds = args.in_stds
        net_args.out_means = args.out_means
        net_args.out_stds = args.out_stds

    build_nn = {"dyna": Dyna, "clf": CLF, "actor": Actor, "net": Net}
    keys = {"dyna": "model_", "actor": "actor_", "clf": "clf_",  "net": "net_"}
    if net_name in ["clf", "actor"]:
        net = build_nn[net_name](None, net_args)
    else:
        net = build_nn[net_name](net_args)
    net = utils.safe_load_nn(net, path, load_last=args.load_last, key=keys[net_name])
    net = utils.cuda(net, args)
    if hasattr(net, "update_device"):
        net.update_device()
    return net, net_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="beta_fit_dbg")
    parser.add_argument('--random_seed', type=int, default=1007)

    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--normalize', action="store_true", default=False)
    parser.add_argument('--split_ratio', type=float, default=0.95)

    parser.add_argument('--dyna_pretrained_path', type=str, default=None)
    parser.add_argument('--actor_pretrained_path', type=str, default=None)
    parser.add_argument('--clf_pretrained_path', type=str, default=None)
    parser.add_argument('--net_pretrained_path', type=str, default=None)
    parser.add_argument('--load_last', action='store_true', default=False)

    parser.add_argument('--test_for_real', action='store_true', default=False)
    parser.add_argument('--nt', type=int, default=3000)
    parser.add_argument('--dt1', type=float, default=0.001)
    parser.add_argument('--dt2', type=float, default=0.0001)
    parser.add_argument('--l0', type=float, default=1.0)
    parser.add_argument('--g', type=float, default=10.0)
    parser.add_argument('--k', type=float, default=32000)
    parser.add_argument('--M', type=float, default=80)

    parser.add_argument('--num_sim_hops', type=int, default=100)
    parser.add_argument('--bloat_factor', type=float, default=0.1)
    parser.add_argument('--mpc', action='store_true', default=False)
    parser.add_argument('--mpc_max_iters', type=int, default=100)

    parser.add_argument('--viz_freq', type=int, default=200)

    parser.add_argument('--methods', type=str, nargs="+", default=['ours'])  #, ['rl', 'mpc', 'ours'])
    parser.add_argument('--num_trials', type=int, default=1)

    parser.add_argument('--rlp_paths', type=str, nargs="+", default=None)
    parser.add_argument('--auto_rl', action='store_true', default=False)
    parser.add_argument('--load_from', type=str, default=None)

    parser.add_argument('--animation', action='store_true', default=False)
    parser.add_argument('--use_d', action='store_true', default=False)
    parser.add_argument('--from_file', type=str, default=None)

    parser.add_argument('--pret', action='store_true', default=False)
    parser.add_argument('--plot_map_only', action='store_true', default=False)

    parser.add_argument('--multi_ani', action='store_true', default=False)
    parser.add_argument('--select_me', type=int, nargs="+", default=None)

    parser.add_argument("--mbpo_paths", type=str, nargs="+", default=None)
    parser.add_argument("--pets_paths", type=str, nargs="+", default=None)

    args = parser.parse_args()

    args.X_DIM = 2  # (xdot, y)
    args.N_REF = 2  # (xdot', y')
    args.U_DIM = 2  # (theta, P)
    args.load_last = True

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
            args.rlp_paths = utils.gen_rlp_path_list("beta", args.methods)
        args.ours_clf_paths = []
        args.ours_actor_paths = []
        args.ours_roa_paths = []
        aug_mbpo_paths = [None for _ in args.methods]
        aug_pets_paths = [None for _ in args.methods]
        for me_i, method in enumerate(args.methods):
            clf_path, actor_path, roa_path, mbpo_path, pets_path = None, None, None, None, None
            if "ours" in method:
                clf_path = args.clf_pretrained_path
                actor_path = args.actor_pretrained_path
                roa_path = args.net_pretrained_path
            elif "mbpo" in method:
                mbpo_path = args.mbpo_paths[me_i]
            elif "pets" in method:
                pets_path = args.pets_paths[me_i]
            
            args.ours_clf_paths.append(clf_path)
            args.ours_actor_paths.append(actor_path)
            args.ours_roa_paths.append(roa_path)
            if args.mbpo_paths is not None:
                aug_mbpo_paths[me_i] = mbpo_path
            if args.pets_paths is not None:
                aug_pets_paths[me_i] = pets_path

        if args.mbpo_paths is not None:
            args.mbpo_paths = aug_mbpo_paths
        if args.pets_paths is not None:
            args.pets_paths = aug_pets_paths


    tt1 = time.time()
    main()
    tt2 = time.time()

    print("Finished in %.4f seconds" % (tt2 - tt1))
