import os
import numpy as np
import argparse
import roa_utils as utils
import torch
import time
from cgw_train import build_net, build_net_full
from cgw_utils import bipedal_next
from cgw_train import get_gait_data
import cgw_opt_ctr as opt_ctr
from cgw_utils import get_gait_data as get_init_gait_data
import matplotlib.pyplot as plt
import pickle
import cgw_mpc
from os.path import join as ospj

import mbrl_utils

def _avg(a_list, use_torch=False):
    if use_torch:
        return torch.mean(torch.tensor(a_list).float())
    else:
        return np.mean(np.array(a_list))


def filter_condition(i, j):
    accept_1 = i >=7 and j <=5
    accept_2 = i < 7 and j >5
    return not (accept_1 or accept_2)
    # return (i <= 7 and j<=7) or (i>7 and j>7)

def main():
    global args
    if args.load_from is not None:
        old_args = np.load(ospj(args.load_from,"args.npz"), allow_pickle=True)['args'].item()
        old_args.load_from = args.load_from
        old_args.use_d = args.use_d
        old_args.multi_ani = args.multi_ani
        old_args.select_me = args.select_me
        old_args.exp_dir_full = utils.get_exp_dir() + old_args.exp_dir_full.split("/")[-1]
        old_args.viz_dir = utils.get_exp_dir() + "/" + old_args.viz_dir.split("/")[-2] + "/" + old_args.viz_dir.split("/")[-1]
        args = old_args
    else:
        utils.set_seed_and_exp(args)
        args.oracle = opt_ctr.OptimalPolicy(args)

        args.gait_data, args.gait_th = get_gait_data(args)
        args.gait_data_np = [utils.to_np(each_gait_data) for each_gait_data in args.gait_data]
        args.gait_th_np = utils.to_np(args.gait_th)

        clf_list = []
        for me_i, method in enumerate(args.methods):
            if "ours" in method:
                clf_tmp = build_net_full("clf", args.ours_clf_paths[me_i], args)
            else:
                clf_tmp = None
            clf_list.append(clf_tmp)
        # clf = build_net("clf", args)

    # how to design the exp
    # move to the target gait
    n_samples = args.num_samples

    root_x = torch.tensor([[0.17828843, -0.35657686, -0.73985833,  0.61868256] for _ in range(n_samples)]).cuda()
    target_x_bak = torch.tensor([[0.13051651, -0.26103302, -0.66095102, 0.30043977] for _ in range(n_samples)]).cuda()

    all_gait = torch.from_numpy(get_init_gait_data()).float().cuda()
    if args.gait_subset is not None:
        src_gait = all_gait[args.gait_subset]

    if args.tgt_gait_subset is not None:
        tgt_gait = all_gait[args.tgt_gait_subset]

    mask_avg = utils.mask_avg

    # TODO
    if args.rl_path is not None:
        from bp_ppo import PPO
        state_dim = 5
        action_dim = 1
        args.lr_actor = 0.1
        args.lr_critic = 0.1
        args.gamma = 0.1
        args.K_epochs = 80
        args.eps_clip = 0.1
        args.action_std = 0.6
        args.discrete = False
        ppo_agent = PPO(state_dim, action_dim, args)
        ppo_agent.load(args.rl_path)

    if args.load_from is None:
        cache = {
            me_i: {
                "init": [],
                "target": [],
                "x_trajs": [],
                "xf_trajs": [],
                "swi_trajs": [],
                "valid_trajs": [],
                "rmse": [],
                "succ": [],
                "valid": [],
                "fail": [],
                "invalid": [],
                "length": [],
                "runtime": [],
                "runtime_step": [],
                "reward": [],
                "name": method,
        } for me_i, method in enumerate(args.methods)}

        # TODO(yue)
        rlp_list = utils.get_loaded_rl_list(args.rlp_paths, args.methods, auto_rl=args.auto_rl)


    num_src_gait = src_gait.shape[0]
    if args.multi_target:
        num_tgt_gait = tgt_gait.shape[0]
    else:
        num_tgt_gait = 1

    for src_i in range(num_src_gait):
        for tgt_i in range(num_tgt_gait):
            if args.load_from is not None:
                continue

            if args.bipart:
                use_this_config = False
                if (src_i in args.rise_src and tgt_i in args.rise_tgt) or (
                        src_i in args.drop_src and tgt_i in args.drop_tgt):
                    use_this_config=True
                if not use_this_config:
                    continue

            if args.filtered:
                if filter_condition(src_i, tgt_i):
                    continue

            # each gait mode
            if args.multi_target:
                root_x = src_gait[[src_i for _ in range(n_samples)]]
                target_x = tgt_gait[[tgt_i for _ in range(n_samples)]]
            else:
                root_x = src_gait[[src_i for _ in range(n_samples)]]
                target_x = target_x_bak

            # dx near goal
            # dx near each gait
            init_x = root_x + utils.uniform_sample_tsr(n_samples, x_mins=args.dx_mins, x_maxs=args.dx_maxs).cuda()

            # on switch
            init_x[:, 0] = torch.clamp(init_x[:, 0], min=0.01)

            n_try = args.n_try
            n_hypo = 29  # 15
            weight_v = 100
            th_thres = args.th_thres
            acc_v = 0.9  # TODO might need to be flexible learned

            num_swi = 1  # how many swis to make sure it's valid

            zeros = torch.zeros_like(init_x[:, 0:1])
            ones = torch.ones_like(init_x[:, 0:1])

            for me_i, method in enumerate(args.methods):
                dbg_t1 = time.time()

                x = init_x
                x_traj = [[x[i]] for i in range(n_samples)]
                xf_traj = [[torch.zeros_like(x[i, 0:1])] for i in range(n_samples)]
                val_traj = [[torch.ones_like(x[i, 0:1])] for i in range(n_samples)]
                swi_traj = [[x[i]] for i in range(n_samples)]
                val_swi_traj= [[torch.ones_like(x[i, 0:1])] for i in range(n_samples)]
                break_list = [0 for i in range(n_samples)]

                for batch_i in range(n_samples):
                    cache[me_i]["init"].append(init_x[batch_i:batch_i+1])
                    cache[me_i]["target"].append(target_x[batch_i:batch_i + 1])

                # planning phase
                for n_i in range(n_try):
                    dbg_tt1=time.time()

                    xf_tmp = torch.stack([xf_traj[i][-1] for i in range(n_samples)], dim=0)
                    if method == "rl":
                        x_next, x_mode, x_seg, xf_seg, val_seg = bipedal_next(
                            x, None, args, obtain_full=True, xf=xf_tmp, use_rl=True, ppo_agent=ppo_agent,
                            target_th=target_x[:, 0:1])
                    elif "rl-" in method:
                        rlp_policy, rlp_running = rlp_list[me_i]
                        x_next, x_mode, x_seg, xf_seg, val_seg = bipedal_next(
                            x, None, args, obtain_full=True, xf=xf_tmp, use_rl=True, agent=rlp_policy, running=rlp_running,
                            target_th=target_x[:, 0:1])
                    elif "mbpo" in method:  # TODO
                        x_next, x_mode, x_seg, xf_seg, val_seg = bipedal_next(
                            x, None, args, obtain_full=True, xf=xf_tmp, use_mbpo=True, agent=args.mbpo_nets[me_i], running=None,
                            target_th=target_x[:, 0:1])
                    elif "pets" in method:  # TODO
                        x_next, x_mode, x_seg, xf_seg, val_seg = bipedal_next(
                            x, None, args, obtain_full=True, xf=xf_tmp, use_pets=True, agent=args.pets_nets[me_i], running=None,
                            target_th=target_x[:, 0:1])
                    elif "mpc" in method:
                        u_out_list=[]
                        for traj_i in range(x.shape[0]):
                            u_out, _, _, _ = cgw_mpc.plan_gait_mpc(x_init=x[traj_i:traj_i+1,:],
                                                               th1d=target_x[traj_i, 0:1].item(), horizon=100, dt=args.dt,
                                                               num_sim_steps=args.num_sim_steps,
                                                               u_mpc_bound=1.0, u_bound=4.0, args=args)
                            u_out_list.append(u_out)
                        u_out_list = torch.stack(u_out_list, dim=0).type_as(x)
                        x_next, x_mode, x_seg, xf_seg, val_seg = bipedal_next(x, None, args, obtain_full=True, xf=xf_tmp,
                                                                              direct_u=u_out_list)
                    else:
                        if method == "qp":
                            uu = torch.cat([ones, zeros, ones * target_x[0, 0], zeros, zeros], dim=-1)
                        elif method == "hjb":
                            uu = torch.cat([zeros, ones, ones * target_x[0, 0], zeros, zeros], dim=-1)
                        else:
                            assert method in ["ours", "ours-d"]

                            xm = torch.tile(x.unsqueeze(1), [1, n_hypo, 1]).reshape([-1, x.shape[-1]])
                            cth = torch.tile(torch.linspace(0.04, 0.18, n_hypo).unsqueeze(0).unsqueeze(-1),
                                             (n_samples, 1, 1)).reshape([-1, 1]).cuda()
                            loss_thres = 500
                            if method == "ours":  # TODO comment the print out function (see in early version)
                                tar_xm = torch.tile(target_x.unsqueeze(1), [1, n_hypo, 1]).reshape([-1, target_x.shape[-1]])
                                xm_cth = torch.cat([xm, cth], dim=-1)
                                c_v = clf_list[me_i](xm_cth, num_ths=n_samples * n_hypo)
                                loss_v = torch.relu(c_v - acc_v + args.roa_thres)  # 0.05
                                loss_dist = (cth - tar_xm[:, 0:1]) ** 2
                                loss = (loss_v * weight_v + loss_dist).reshape([n_samples, n_hypo])  # + torch.relu(torch.abs(cur_th-cth)-0.01)*100
                                loss = torch.clamp(loss, max=500)
                                loss_item = torch.min(loss, dim=1)[0]
                                valid_m = loss_item < loss_thres
                                loss_mean = torch.sum(loss_item * valid_m) / torch.clamp(torch.sum(valid_m), 1)

                            elif method == "ours-d":
                                lr = 0.00002
                                n_iters = 5
                                cth = cth.requires_grad_()
                                optimizer = torch.optim.RMSprop([cth], lr=lr)
                                for sgd_i in range(n_iters):
                                    tar_xm = torch.tile(target_x.unsqueeze(1), [1, n_hypo, 1]).reshape(
                                        [-1, target_x.shape[-1]])
                                    xm_cth = torch.cat([xm, cth], dim=-1)
                                    c_v = clf_list[me_i](xm_cth, num_ths=n_samples * n_hypo)
                                    loss_v = torch.relu(c_v - acc_v + args.roa_thres)  # 0.05
                                    loss_dist = (cth - tar_xm[:, 0:1]) ** 2
                                    loss = (loss_v * weight_v + loss_dist).reshape([n_samples, n_hypo])
                                    loss_item = torch.min(loss, dim=1)[0]
                                    valid_m = loss_item < loss_thres
                                    loss_mean = torch.sum(loss_item * valid_m) / torch.clamp(torch.sum(valid_m), 1)
                                    optimizer.zero_grad()
                                    loss_mean.backward()
                                    optimizer.step()

                                cth = cth.detach()
                                loss = loss.detach()
                            best_th = torch.argmin(loss, dim=1)
                            th = cth.reshape([n_samples, n_hypo])[range(n_samples), best_th].unsqueeze(-1)
                            uu = torch.cat([ones, zeros, th, zeros, zeros], dim=-1)

                        x_next, x_mode, x_seg, xf_seg, val_seg = bipedal_next(x, uu, args, obtain_full=True, xf=xf_tmp)
                    dbg_tt6 = time.time()
                    x = x_next

                    # collect trajectory-wise info
                    for i in range(n_samples):
                        if break_list[i]==False:
                            x_traj[i] += x_seg[i][1:]
                            xf_traj[i] += xf_seg[i][1:]
                            val_traj[i] += val_seg[i][1:]
                            swi_traj[i].append(x_seg[i][-1])
                            val_swi_traj[i].append(val_seg[i][-1])

                            # if the valid ones are within small error, abort!
                            break_list[i] = torch.logical_or(
                                val_traj[i][-1]==False,
                                torch.norm(x[i, 0:1]-target_x[i, 0:1])<(th_thres if method!="rl" else th_thres*1))
                    if all(break_list):
                        break

                dbg_t2 = time.time()

                for i in range(n_samples):
                    x_traj[i] = torch.stack(x_traj[i], dim=0)
                    xf_traj[i] = torch.stack(xf_traj[i], dim=0)
                    val_traj[i] = torch.stack(val_traj[i], dim=0)
                    swi_traj[i] = torch.stack(swi_traj[i], dim=0)
                    cache[me_i]["x_trajs"].append(x_traj[i])
                    cache[me_i]["xf_trajs"].append(xf_traj[i])
                    cache[me_i]["swi_trajs"].append(swi_traj[i])
                    cache[me_i]["valid_trajs"].append(val_traj[i])

                # visualization (with real sim)
                args.test = False
                args.alpha = 0

                # TODO estimate the performance
                target_x_np = utils.to_np(target_x)
                max_traj_len = np.max([x_traj[i].shape[0] for i in range(n_samples)])
                for i in range(n_samples):
                    success = (torch.norm(x_traj[i][-1, 0] - target_x_np[i, 0]) < th_thres).float()
                    valid = float((val_traj[i][-1, 0]).item() and len(val_swi_traj[i])>=1 and (val_swi_traj[i][num_swi]).item())
                    valid = torch.tensor(valid).type_as(success)
                    n_valids = int(torch.sum(val_traj[i]))
                    length = xf_traj[i][n_valids-1, 0]
                    rmse = compute_rmse(swi_traj[i], target_x[i], success)

                    rmse_cost = rmse
                    invalid_cost = (1 - torch.mean(valid).item()) * 100  # self.args.invalid_cost
                    swi_reward = 100 * float(success) * \
                                 (1 - torch.abs(x_traj[i][-1, 0] - target_x_np[i, 0]) / target_x_np[i, 0])  # self.args.switch_bonus
                    reward = -rmse_cost - invalid_cost + swi_reward


                    cache[me_i]["rmse"].append(rmse)
                    cache[me_i]["succ"].append(success)
                    cache[me_i]["fail"].append(1-success)
                    cache[me_i]["valid"].append(valid)
                    cache[me_i]["invalid"].append(1-valid)
                    cache[me_i]["length"].append(length)
                    cache[me_i]["runtime"].append(torch.tensor((dbg_t2 - dbg_t1) / n_samples).type_as(target_x))
                    cache[me_i]["runtime_step"].append(torch.tensor((dbg_t2 - dbg_t1) / n_samples / max_traj_len).type_as(target_x))

                    cache[me_i]["reward"].append(reward.item())

                    assert n_valids >= 1

                print("[th=%.3f->%.3f %5s] rmse:%.3f(%.3f) succ:%.3f(%.3f)  valid:%.3f(%.3f)  length:%.3f(%.3f) runtime:%.3f(%.3f) step:%.5f(%.5f) reward:%.3f(%.3f)" %
                      (src_gait[src_i, 0], target_x[0, 0], method,
                       mask_avg(cache[me_i]["rmse"][-n_samples:], True), mask_avg(cache[me_i]["rmse"], True),
                       mask_avg(cache[me_i]["succ"][-n_samples:]), mask_avg(cache[me_i]["succ"]),
                       mask_avg(cache[me_i]["valid"][-n_samples:]), mask_avg(cache[me_i]["valid"]),
                       mask_avg(cache[me_i]["length"][-n_samples:]), mask_avg(cache[me_i]["length"]),
                       mask_avg(cache[me_i]["runtime"][-n_samples:]), mask_avg(cache[me_i]["runtime"]),
                       mask_avg(cache[me_i]["runtime_step"][-n_samples:]), mask_avg(cache[me_i]["runtime_step"]),
                       mask_avg(cache[me_i]["reward"][-n_samples:]), mask_avg(cache[me_i]["reward"]),
                       ))

    color_list={"ours": "blue", "ours-d": "royalblue", "qp": "red", "hjb": "purple", "rl": "gray", "rl-a2c": "dimgray", "rl-ddpg":"rosybrown",
        "rl-ppo":"lightskyblue", "rl-sac": "limegreen", "rl-td3":"olive", "rl-trpo": "sienna", "rl-vpg": "indigo",
                "mpc": "indigo", "mbpo":"pink", "pets":"brown"}

    if args.load_from is None:
        np.savez("%s/cache.npz" % (args.exp_dir_full), data=cache)
    else:
        cache = np.load(ospj(args.load_from, "cache.npz"),allow_pickle=True)['data'].item()
    # np.savez("%s/result.npz"%(args.exp_dir_full), data=[succ_list, vali_list, leng_list])

    if args.multi_ani and args.select_me is not None:
        args.methods = [args.methods[me_i] for me_i in args.select_me]
        cache = {ii: cache[me_i] for ii, me_i in enumerate(args.select_me)}
        assert args.methods[-1] == "ours-d"
        args.methods[-1] = "ours"

        # TODO viz for bipedal walker
        trial_freq = 10
        for trial_i in range(len(cache[0]["x_trajs"])):
            if trial_i % trial_freq != 0:
                continue
            for me_i, method in enumerate(args.methods):
                print(trial_i, method)
                plt.figure(figsize=(8, 4))
                fontsize=18
                legend_fontsize = 15
                linewidth = 5
                markersize = 8
                l = 1.0
                leg_width = 3
                head_size = 120
                foot_size = 90
                viz_freq = 40
                x_trajs = cache[me_i]["x_trajs"][trial_i]
                xf_trajs = cache[me_i]["xf_trajs"][trial_i]

                plt.axhline(0, linewidth=linewidth, color="darkgreen", label='ground')
                alpha = 0.5
                for ti in range(len(x_trajs)):
                    xs = x_trajs[ti].detach().cpu().numpy()
                    stance_x = xf_trajs[ti].item()
                    stance_y = 0
                    head_x = stance_x + l * np.sin(-xs[0])
                    head_y = stance_y + l * np.cos(xs[0])
                    swing_x = head_x + l * np.sin(xs[1] + xs[0])
                    swing_y = head_y - l * np.cos(xs[1] + xs[0])
                    if swing_y<-0.5 or head_y<0:
                        max_t = ti - 1
                        break
                    else:
                        max_t = ti

                for ti in range(max_t):
                    if ti % viz_freq != 0 and ti != max_t - 1:
                        continue
                    xs = x_trajs[ti].detach().cpu().numpy()
                    stance_x = xf_trajs[ti].item()
                    stance_y = 0
                    head_x = stance_x + l * np.sin(-xs[0])
                    head_y = stance_y + l * np.cos(xs[0])
                    swing_x = head_x + l * np.sin(xs[1] + xs[0])
                    swing_y = head_y - l * np.cos(xs[1] + xs[0])

                    alpha = 1.0 - 0.75 * ti / max_t
                    first = ti == 0
                    plt.plot([stance_x, head_x], [stance_y, head_y], label="stance leg" if first else None,
                             color="blue", linewidth=leg_width, alpha=alpha)
                    plt.plot([swing_x, head_x], [swing_y, head_y], label="swing leg" if first else None, color="red",
                             linewidth=leg_width, alpha=alpha)
                    plt.scatter([head_x], [head_y], label="head" if first else None, color="black", s=head_size,
                                alpha=alpha, zorder=10000)
                    plt.scatter([stance_x], [stance_y], label="stance foot" if first else None, color="darkblue",
                                s=foot_size, alpha=alpha, zorder=10000)
                    plt.scatter([swing_x], [swing_y], label="swing foot" if first else None, color="brown", s=foot_size,
                                alpha=alpha, zorder=10000)
                plt.axis("scaled")
                plt.xlim(-0.5, 3.0)
                plt.ylim(-0.25, 1.5)
                plt.xlabel("x (m)", fontsize=fontsize)
                plt.ylabel("y (m)", fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.legend(fontsize=legend_fontsize, loc="upper right")
                if args.multi_target:
                    plt.savefig("%s/cgw_simx_%04d_%s.png" % (args.viz_dir, trial_i, method), bbox_inches='tight',
                                pad_inches=0.02)
                else:
                    plt.savefig("%s/cgw_sim_%04d_%s.png" % (args.viz_dir, trial_i, method), bbox_inches='tight', pad_inches=0.02)
                plt.close()
        return

    if args.from_file:
        utils.gen_plot_rl_std_curve(
            args.ds_list, args.methods, cache, "%s/std_walker_curve.png" % (args.exp_dir_full),
            loc="upper left")
        return

    # TODO change plotting orders here
    cs_d = utils.get_cs_d()
    # np.savez("%s/stat.npz"%(args.exp_dir_full), succ_list=succ_list, vali_list=vali_list, leng_list=leng_list)
    for metric in ["rmse", "succ", "valid", "fail", "invalid", "length", "runtime", "runtime_step"]:
        masked = metric == "rmse"
        utils.metric_bar_plot(cache, metric, args, header="_walker" + ("Mul" if args.multi_target else ""),
                              index=True, masked=masked, cs_d=cs_d, mode="walker", rl_merged=True)
        utils.metric_bar_plot(cache, metric, args, header="_walker" + ("Mul" if args.multi_target else ""),
                              index=True, masked=masked, cs_d=cs_d, mode="walker", rl_merged=False)

    if args.one_line:
        plt.plot(range(len(args.methods)), [utils.to_np(mask_avg(cache[me_i]["succ"])) for me_i,method in enumerate(args.methods)])
        plt.legend()
        plt.xlabel("theta")
        plt.ylabel("success rate")
        plt.savefig("%s/succ_curve.png" % (args.viz_dir))
        plt.close()

    else:
        if args.multi_target:
            if args.load_from is None:
                for metric in ["succ", "rmse", "valid"]:
                    for me_i, method in enumerate(args.methods):
                        ll = []
                        cnt_idx = 0
                        for i in range(num_src_gait):
                            for j in range(num_tgt_gait):
                                if args.filtered:
                                    if filter_condition(i, j):
                                        if metric in ["succ", "valid"]:
                                            ll.append(0.0)
                                        else:
                                            ll.append(5.0)
                                        continue
                                b_s = cnt_idx * n_samples
                                b_e = (cnt_idx + 1) * n_samples
                                val = utils.to_np(mask_avg(cache[me_i][metric][b_s: b_e], masked=metric=="rmse"))
                                ll.append(val)
                                cnt_idx += 1
                        mat = np.array(ll).reshape((num_src_gait, num_tgt_gait))
                        plt.imshow(mat, origin="lower")
                        plt.xlabel("tgt_gait")
                        plt.ylabel("src_gait")
                        ax = plt.gca()
                        x_positions = np.linspace(start=0, stop=num_tgt_gait, num=num_tgt_gait, endpoint=False)
                        y_positions = np.linspace(start=0, stop=num_src_gait, num=num_src_gait, endpoint=False)
                        jump_x = 0.0
                        jump_y = 0.0
                        for y_index, y in enumerate(y_positions):
                            for x_index, x in enumerate(x_positions):
                                label = mat[y_index, x_index]
                                text_x = x + jump_x
                                text_y = y + jump_y
                                ax.text(text_x, text_y, "%.3f"%(label), color='black', ha='center', va='center')
                        plt.colorbar()
                        if metric == "succ":
                            plt.clim(0, 1.0)
                        elif metric == "rmse":
                            plt.clim(0, 5.0)
                        elif metric == "valid":
                            plt.clim(0, 1.0)
                        plt.savefig("%s/mat_%s_%s.png" % (args.viz_dir, metric, method), bbox_inches='tight', pad_inches=0.1)
                        plt.close()
        else:
            for me_i, method in enumerate(args.methods):
                plt.plot([src_gait[i, 0].item() for i in range(num_src_gait)],
                         [utils.to_np(mask_avg(cache[me_i]["succ"][i * n_samples: (i+1)*n_samples])) for i in range(num_src_gait)],
                         color=color_list[method], label=method)
            plt.legend()
            plt.xlabel("theta")
            plt.ylabel("success rate")
            plt.savefig("%s/succ_curve.png" % (args.viz_dir))
            plt.close()


def compute_rmse(swi_traj, target_x, success):
    if len(swi_traj) == 0:
        return torch.tensor(-1.0).type_as(target_x)
    else:
        # if success<0.5:
        #     return torch.tensor(-1.0).type_as(target_x)
        s = 0
        valid_len = 0
        need_break = False
        l1_angle = np.pi/4
        l2_angle = np.pi/2
        for val_swi in swi_traj:
            # print(val_swi.shape, target_x.shape)
            if val_swi[0].item()<0 or val_swi[0].item()>l1_angle or \
                    val_swi[1].item() < -l2_angle or val_swi[1].item() > 0:
                val_swi[0] = torch.clamp(val_swi[0], 0, l1_angle)
                val_swi[1] = torch.clamp(val_swi[1], -l2_angle, 0)
                need_break=True
            valid_len+=1
            assert val_swi.shape == target_x.shape
            # s = s + torch.norm(val_swi[:2] - target_x[:2])
            s = s + torch.norm(val_swi - target_x)
            if need_break:
                break
        return s / valid_len
        # return s / len(swi_traj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="bp_train_dbg")
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--clf_pretrained_path', type=str, default=None)
    parser.add_argument('--actor_pretrained_path', type=str, default=None)
    parser.add_argument('--rl_path', type=str, default=None)
    parser.add_argument('--print_sim_freq', type=int, default=3)

    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--dx_mins', type=float, nargs="+", default=[-0.03, -0.06, -0.25, -0.5])
    parser.add_argument('--dx_maxs', type=float, nargs="+", default=[0.03, 0.06, 0.25, 0.5])
    parser.add_argument('--use_qp', action='store_true', default=False)
    parser.add_argument('--use_hjb', action='store_true', default=False)
    parser.add_argument('--roa_thres', type=float, default=0.2)
    parser.add_argument('--th_thres', type=float, default=0.0001)
    parser.add_argument('--n_try', type=int, default=10)
    parser.add_argument('--gait_idx_begin', type=int, default=0)
    parser.add_argument('--gait_idx_end', type=int, default=29)
    parser.add_argument('--n_theta', type=int, default=29)
    parser.add_argument('--gait_idx_indices', nargs="+", type=int, default=None)
    parser.add_argument('--clf_hiddens', type=int, nargs="+", default=[64, 64])

    parser.add_argument('--methods', type=str, nargs="+", default=["ours", "rl", "qp", "hjb"])
    parser.add_argument('--rlp_paths', type=str, nargs="+", default=["ours", "rl", "qp", "hjb"])
    parser.add_argument('--gait_subset', type=int, nargs="+", default=None)
    parser.add_argument('--one_line', action='store_true', default=False)

    parser.add_argument('--multi_target', action='store_true', default=False)
    parser.add_argument('--tgt_gait_subset', type=int, nargs="+", default=None)

    parser.add_argument('--filtered', action='store_true', default=False)
    parser.add_argument('--auto_rl', action='store_true', default=False)
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--mpc_max_iters', type=int, default=100)
    parser.add_argument('--use_d', action='store_true', default=False)
    parser.add_argument('--from_file', type=str, default=None)

    parser.add_argument('--bipart', action='store_true', default=False)
    parser.add_argument('--rise_src', type=int, nargs="+", default=[])
    parser.add_argument('--rise_tgt', type=int, nargs="+", default=[])
    parser.add_argument('--drop_src', type=int, nargs="+", default=[])
    parser.add_argument('--drop_tgt', type=int, nargs="+", default=[])

    parser.add_argument('--multi_ani', action='store_true', default=False)
    parser.add_argument('--select_me', type=int, nargs="+", default=None)

    parser.add_argument("--mbpo_paths", type=str, nargs="+", default=None)
    parser.add_argument("--pets_paths", type=str, nargs="+", default=None)
    parser.add_argument("--ours_clf_paths", type=str, nargs="+", default=None)
    # stability criteria

    args = parser.parse_args()
    args.load_last = True
    args.nt = 200
    args.dt = 0.01
    args.num_sim_steps = 2
    args.fine_switch = True
    args.constant_g = False
    args.changed_dynamics = False
    args.qp_bound = 4
    args.hjb_u_bound = 4
    args.reset_q1_threshold = -0.03
    args.X_DIM = 4
    args.N_REF = 1
    args.U_DIM = 1

    args.mbpo_nets = [None] * len(args.methods)
    args.pets_nets = [None] * len(args.methods)

    if args.from_file:
        data_from_file = utils.parse_from_file(args.from_file, mode="walker")
        args.rlp_paths = data_from_file["rlp_paths"]
        args.ours_actor_paths = data_from_file["actor_paths"]
        args.ours_clf_paths = data_from_file["clf_paths"]
        args.ours_roa_paths = data_from_file["roa_paths"]
        args.methods = data_from_file["methods"]
        args.ds_list = data_from_file["ds_list"]
    else:
        if args.ours_clf_paths is not None:
            aug_clf_paths = []
            for mi, method_name in enumerate(args.methods):
                if method_name == "ours":
                    aug_clf_paths.append(args.ours_clf_paths[0])
                else:
                    aug_clf_paths.append(None)
            args.ours_clf_paths = aug_clf_paths

        if args.auto_rl:
            args.rlp_paths = utils.gen_rlp_path_list("bp", args.methods)

        if args.mbpo_paths is not None:
            for mi, mbpo_path in enumerate(args.mbpo_paths):
                if mbpo_path != "None":
                    args.mbpo_nets[mi] = mbrl_utils.get_mbpo_models(mbpo_path, args)

        if args.pets_paths is not None:
            for mi, pets_path in enumerate(args.pets_paths):
                if pets_path != "None":
                    args.pets_nets[mi] = mbrl_utils.get_pets_models(pets_path, args)
            

    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.4f seconds" % (t2 - t1))