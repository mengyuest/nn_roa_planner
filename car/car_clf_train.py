import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from scipy import linalg
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from matplotlib import ticker, cm

import car_roa_viz as rviz
import car_utils as sscar_utils

sys.path.append("../")
import roa_nn
import roa_utils as utils


def plot(v, metric, label, func):
    order = np.argsort(metric).astype(int)
    plt.bar(range(len(v)), v[order], width=1.0, color=func(label[order]))

class MockArgs:
    pass


def init_sampling(args, num_samples=None, sample_ratio=1.0):
    if num_samples is None:
        num_samples = args.num_samples
    # Sampling sub_num
    init_states = utils.uniform_sample_tsr(num_samples, args.s_mins, args.s_maxs, sample_ratio)
    init_mus = utils.rand_choices_tsr(num_samples, args.mu_choices)
    init_refs = utils.uniform_sample_tsr(num_samples, args.ref_mins, args.ref_maxs)
    init_idx = torch.tensor(range(num_samples)).float().unsqueeze(-1)
    cfg_tensor = torch.cat([init_mus, init_refs, init_idx], axis=-1)
    # (N_CONFIGS, N_SAMPLES, X_DIM + N_REF + 1)
    tensor = torch.cat([init_states, cfg_tensor], axis=-1)
    return tensor, cfg_tensor


def to_cuda(*tensor_list):
    tensor_cuda_list = []
    for tensor in tensor_list:
        tensor_cuda = tensor.cuda()
        tensor_cuda_list.append(tensor_cuda)
    return tensor_cuda_list


def init_samp_with_P_K(car_params, args=None, quiet=False, no_p_k=False):
    if quiet==False:
        print("NEW SAMPLING!")
    init_s, params = init_sampling(args)
    if no_p_k==False:
        lqr_P, lqr_K = solve_for_P_K(params, car_params, args)
    else:
        lqr_P_cuda, lqr_K_cuda = None, None
    if args.gpus is not None:
        init_s = init_s.cuda()
        params = params.cuda()
        if no_p_k == False:
            lqr_P_cuda = lqr_P.cuda()
            lqr_K_cuda = lqr_K.cuda()
    return init_s, params, lqr_P_cuda, lqr_K_cuda


def lqr_worker(input):
    i, A, B, Q, R = input
    K = sscar_utils.lqr(A, B, Q, R)
    return i, K


def lqr_parallel(A_list, B_list, Q, R, args):
    with Pool(processes=args.num_workers) as pool:
        input = [(i, A_list[i], B_list[i], Q, R) for i in range(len(A_list))]
        res = pool.map(lqr_worker, input)
    return [x[1] for x in sorted(res, key=lambda x: x[0])]


def cont_lyap_worker(input):
    i, A, Q = input
    P = linalg.solve_continuous_lyapunov(A, -Q)
    return i, P


def cont_lyap_worker_parallel(A_list, Q, args):
    with Pool(processes=args.num_workers) as pool:
        input = [(i, A_list[i], Q) for i in range(len(A_list))]
        res = pool.map(cont_lyap_worker, input)
    return [x[1] for x in sorted(res, key=lambda x: x[0])]

def to_np(tensor):
    return tensor.detach().cpu().numpy()


def solve_for_P_K(params, car_params, args, single=False):
    t1 = time.time()
    goal_s = torch.zeros(params.shape[0], args.X_DIM)
    x0 = sscar_utils.compute_x0_from_s(goal_s, car_params, params, args, K_hist={}, pure_s=True)
    Act = sscar_utils.compute_Act_b(to_np(x0), car_params, params, args)
    Bct = sscar_utils.compute_Bct_b(x0, car_params, params, args)
    A = np.tile(np.eye(args.X_DIM), (params.shape[0], 1, 1)) + args.controller_dt * Act
    B = args.controller_dt * Bct
    Q = np.eye(args.X_DIM)
    R = np.eye(args.U_DIM)

    if single:
        K_list = [sscar_utils.lqr(A[0], B[0], Q, R)] * A.shape[0]
    else:
        K_list = lqr_parallel(A, B, Q, R, args)
    K_np = np.stack(K_list, axis=0)

    Acl_T = np.transpose(Act - np.einsum('BNi,BiM->BNM', Bct, K_np), (0, 2, 1))
    if single:
        P_list = [linalg.solve_continuous_lyapunov(Acl_T[0], -Q)] * A.shape[0]
    else:
        P_list = cont_lyap_worker_parallel(Acl_T, Q, args)
    P_np = np.stack(P_list, axis=0)
    P_all = torch.from_numpy(P_np).float()
    K_all = torch.from_numpy(K_np).float()
    return P_all, K_all


def to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def next_step(x, policy, car_params, params, is_lqr, lqr_K_cuda, args, detached=False, directed_u=False):
    dbg_t1=time.time()
    if len(x.shape) == 3:
        reshape = True
        N, T, _ = x.shape
        x = x.reshape(N * T, -1)
    else:
        reshape = False

    val_f = sscar_utils._f(x[:, :args.X_DIM], car_params, params)
    val_g = sscar_utils._g(x[:, :args.X_DIM], car_params, params)
    if is_lqr:
        u = sscar_utils.cal_lqr(x[:, :args.X_DIM], car_params, params, lqr_K_cuda, args)
    else:
        if directed_u:
            u = policy
        elif detached:
            u = policy(x.detach()).detach()
        else:
            u = policy(x)

    x_dot = val_f + torch.bmm(val_g, u.unsqueeze(-1))
    x_dot = x_dot.squeeze(-1)
    new_x = x.clone().detach()
    new_x[:, :args.X_DIM] = x[:, :args.X_DIM] + x_dot * args.dt

    if args.clip_angle:
        new_x = torch.cat([
            new_x[:, 0:2],
            to_pi(new_x[:, 2:3]),
            new_x[:, 3:4],
            to_pi(new_x[:, 4:5]),
            torch.clamp(new_x[:, 5:6], -100, 100),
            to_pi(new_x[:, 6:7]),
            new_x[:, args.X_DIM:]
        ], dim=-1)
    if reshape:
        new_x = new_x.reshape(N, T, -1)
        u = u.reshape(N, T, -1)
    dbg_t2 = time.time()
    return new_x, u


def get_trajectories(init_s, policy, car_params, params, is_lqr=False, lqr_K_cuda=None, args=None, detached=False,
                     tail=False, manual_t=None):
    s_list = [init_s]
    u_list = []
    if manual_t is None:
        nt = args.nt
    else:
        nt = manual_t
    for i in range(nt):
        x = s_list[-1]
        new_x, u = next_step(x, policy, car_params, params, is_lqr, lqr_K_cuda, args, detached=detached)
        if tail:
            s_list = [new_x]
        else:
            s_list.append(new_x)
            u_list.append(u)
    if tail:
        return new_x, None
    else:
        return torch.stack(s_list, axis=1), torch.stack(u_list, axis=1)


def next_step_nn(x, policy, car_params, params, args, detached=False):
    return next_step(x, policy, car_params, params, is_lqr=False, lqr_K_cuda=None, args=args, detached=detached)


def get_traj_nn(init_s, policy, car_params, params, args, detached=False, tail=False, manual_t=None):
    return get_trajectories(init_s, policy, car_params, params, is_lqr=False, lqr_K_cuda=None, args=args,
                            detached=detached, tail=tail, manual_t=manual_t)


def get_traj_lqr(init_s, lqr_K_cuda, car_params, params, args, detached=False, tail=False, manual_t=None):
    return get_trajectories(init_s, None, car_params, params, is_lqr=True, lqr_K_cuda=lqr_K_cuda, args=args,
                            detached=detached, tail=tail, manual_t=manual_t)


def pretrain_clf_multi(clf, init_s, lqr_trs, init_cfg, lqr_P):
    train_lr = args.pretrain_clf_setups[0]
    train_epochs = int(args.pretrain_clf_setups[1])
    print_freq = int(args.pretrain_clf_setups[2])
    viz_freq = int(args.pretrain_clf_setups[3])
    split = 0.8

    N, T, _ = lqr_trs.shape
    n_train, n_test = int(args.num_samples * split), N - int(args.num_samples * split)

    if args.clf_pretrain_traj:
        train_xs, test_xs = utils.split_tensor(lqr_trs, n_train)
        train_xs = train_xs.reshape(n_train * T, -1)
        test_xs = test_xs.reshape(n_test * T, -1)
        init_p = lqr_P.unsqueeze(1).repeat(1, T, 1, 1)
        train_ps, test_ps = utils.split_tensor(init_p, n_train)
        train_ps = train_ps.reshape(n_train * T, args.X_DIM, args.X_DIM)
        test_ps = test_ps.reshape(n_test * T, args.X_DIM, args.X_DIM)
    else:
        init_p = lqr_P  # (BN, X_DIM, X_DIM)
        train_xs, test_xs = utils.split_tensor(init_s, n_train)
        train_ps, test_ps = utils.split_tensor(init_p, n_train)
    criterion = nn.MSELoss()
    train_losses, test_losses = utils.get_n_meters(2)
    if args.new_clf_pretraining:
        train_zero_losses, test_zero_losses, train_dec_losses, test_dec_losses, train_cls_losses, test_cls_losses = utils.get_n_meters(6)

    optimizer = torch.optim.RMSprop(clf.parameters(), train_lr)
    quad_batch = lambda x, pp: utils.quadratic_multi(x[:, :args.X_DIM], pp) / 2
    quad = lambda x, pp: utils.quadratic(x[:, :args.X_DIM], pp) / 2
    relu = torch.relu

    if args.clf_pretrain_traj:
        total_n_train = n_train * T
        total_n_test = n_test * T
        train_len = 8000
        test_len = 2000
    else:
        total_n_train = n_train * 1
        total_n_test = n_test * 1
        train_len = 1000
        test_len = 1000

    test_indices = torch.randperm(total_n_test)[:test_len]
    for epi in range(train_epochs):
        if args.always_new_sampling and epi % args.new_sampling_freq == 0:
            init_s, params, init_p, init_k = init_samp_with_P_K(car_params, args, True)
            if args.clf_pretrain_traj:
                lqr_trs, lqr_us = get_traj_lqr(init_s, init_k, car_params, params, args)
                train_xs, test_xs = utils.split_tensor(lqr_trs, n_train)
                train_xs = train_xs.reshape(n_train * T, -1)
                test_xs = test_xs.reshape(n_test * T, -1)
                init_p = init_p.unsqueeze(1).repeat(1, T, 1, 1)
                train_ps, test_ps = utils.split_tensor(init_p, n_train)
                train_ps = train_ps.reshape(n_train * T, args.X_DIM, args.X_DIM)
                test_ps = test_ps.reshape(n_test * T, args.X_DIM, args.X_DIM)

                marked, roa_idx = analyze_traj_u(lqr_trs, lqr_us, "LQR controller", header="pretrain_clf_lqr", skip_print=True, skip_viz=True, args=args)
                train_mask, test_mask = utils.split_tensor(marked, n_train)  # (n, )
                train_traj_mask = train_mask.unsqueeze(1).repeat(1, T)
                test_traj_mask = test_mask.unsqueeze(1).repeat(1, T)

                if args.new_clf_pretraining:
                    train_xs = train_xs.reshape(n_train, T, -1)
                    test_xs = test_xs.reshape(n_test, T, -1)
            else:
                train_xs, test_xs = utils.split_tensor(init_s, n_train)
                train_ps, test_ps = utils.split_tensor(init_p, n_train)

        if args.clf_pretrain_mask:
            if epi % 50 == 0:
                train_indices = torch.randperm(total_n_train)[:train_len]
        else:
            train_indices = torch.tensor(range(total_n_train))
            test_indices = torch.tensor(range(total_n_test))

        if args.new_clf_pretraining:
            train_zo = train_xs[:, 0, :].detach().clone()
            train_zo[:, :args.X_DIM] = 0
            test_zo = test_xs[:, 0, :].detach().clone()
            test_zo[:, :args.X_DIM] = 0
            v_est_train = clf(train_xs)
            v_zo_train = clf(train_zo)
            v_est_test = clf(test_xs)
            v_zo_test = clf(test_zo)

            loss_zero_train = torch.mean(torch.square(v_zo_train))
            loss_zero_test = torch.mean(torch.square(v_zo_test))
            loss_dec_train = mask_loss(torch.zeros_like(train_traj_mask[:, 1:]),
                                       relu(v_est_train[:, 1:, 0] - (1-args.alpha_v) * v_est_train[:, :-1, 0] + args.v_margin),
                                       train_traj_mask[:, 1:])
            loss_dec_test = mask_loss(torch.zeros_like(test_traj_mask[:, 1:]),
                                       relu(v_est_test[:, 1:, 0] - (1-args.alpha_v) * v_est_test[:, :-1, 0] + args.v_margin),
                                       test_traj_mask[:, 1:])
            loss_cls_train = mask_loss(relu(1 - v_est_train[:, 0, 0]), relu(v_est_train[:, 0, 0] - 1), train_mask)
            loss_cls_test = mask_loss(relu(1 - v_est_test[:, 0, 0]), relu(v_est_test[:, 0, 0] - 1), test_mask)

            loss_zero_train = loss_zero_train * args.pre_clf_zero_weight
            loss_zero_test = loss_zero_test * args.pre_clf_zero_weight
            loss_dec_train = loss_dec_train * args.pre_clf_dec_weight
            loss_dec_test = loss_dec_test * args.pre_clf_dec_weight
            loss_cls_train = loss_cls_train * args.pre_clf_cls_weight
            loss_cls_test = loss_cls_test * args.pre_clf_cls_weight

            loss_train = loss_zero_train + loss_dec_train + loss_cls_train
            loss_test = loss_zero_test + loss_dec_test + loss_cls_test

            utils.update(train_zero_losses, loss_zero_train)
            utils.update(test_zero_losses, loss_zero_test)
            utils.update(train_dec_losses, loss_dec_train)
            utils.update(test_dec_losses, loss_dec_test)
            utils.update(train_cls_losses, loss_cls_train)
            utils.update(test_cls_losses, loss_cls_test)
            utils.update(train_losses, loss_train)
            utils.update(test_losses, loss_test)

            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epi % print_freq == 0:
                print("%04d/%04d train[%5.2f%%]: %.4f (%.4f) zo: %.4f dec: %.4f cls: %.4f  test[%5.2f%%]: %.4f (%.4f) zo: %.4f dec: %.4f cls: %.4f" % (
                    epi, train_epochs, torch.mean(train_mask.float()).item() * 100,
                    train_losses.val, train_losses.avg, train_zero_losses.val, train_dec_losses.val, train_cls_losses.val,
                    torch.mean(test_mask.float()).item() * 100, test_losses.val, test_losses.avg,
                    test_zero_losses.val, test_dec_losses.val, test_cls_losses.val,
                ))

        else:
            v_est_train = clf(train_xs[train_indices])
            v_gt_train = quad_batch(train_xs[train_indices], train_ps[train_indices])
            v_est_test = clf(test_xs[test_indices])
            v_gt_test = quad_batch(test_xs[test_indices], test_ps[test_indices])
            loss_train = criterion(v_est_train, v_gt_train)
            loss_test = criterion(v_est_test, v_gt_test)

            utils.update(train_losses, loss_train)
            utils.update(test_losses, loss_test)
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epi % print_freq == 0:
                print("%04d/%04d train: %.4f (%.4f)  test: %.4f (%.4f)" % (
                    epi, train_epochs, train_losses.val, train_losses.avg, test_losses.val, test_losses.avg))

        if epi == 0 or epi == train_epochs - 1 or epi % viz_freq == 0:
            if args.clf_pretrain_traj:
                if args.new_clf_pretraining:
                    viz_test_xs = test_xs
                else:
                    viz_test_xs = test_xs.reshape(n_test, T, -1)
                    v_est_test = clf(viz_test_xs)
                n_trajs, nt, _ = viz_test_xs.shape
                ts = range(nt)
                lqr_idx0 = torch.where(test_mask.float() == 0)[0]
                lqr_idx1 = torch.where(test_mask.float() == 1)[0]
                v_est_test_np = to_np(v_est_test)

                v_p_test = quad_batch(viz_test_xs[:, :, :args.X_DIM].reshape(n_test * T, args.X_DIM), test_ps)
                v_p_test = v_p_test.reshape(n_test, T, 1)
                v_p_test_np = to_np(v_p_test)

                v_norm_test = torch.norm(viz_test_xs[:, :, :args.X_DIM], dim=-1, keepdim=True)
                v_norm_test_np = to_np(v_norm_test)

                ax1 = plt.subplot(3, 2, 1)
                ax1.set_title("Unstable, NN")
                for tr_i in lqr_idx0:
                    plt.plot(ts, v_est_test_np[tr_i, :, 0], color="red", alpha=0.25)
                ax2 = plt.subplot(3, 2, 2)
                ax2.set_title("Stable, NN")
                for tr_i in lqr_idx1:
                    plt.plot(ts, v_est_test_np[tr_i, :, 0], color="blue", alpha=0.25)
                ax3 = plt.subplot(3, 2, 3)
                ax3.set_title("Unstable, Quadratic")
                for tr_i in lqr_idx0:
                    plt.plot(ts, v_p_test_np[tr_i, :, 0], color="red", alpha=0.25)
                ax4 = plt.subplot(3, 2, 4)
                ax4.set_title("Stable, Quadratic")
                for tr_i in lqr_idx1:
                    plt.plot(ts, v_p_test_np[tr_i, :, 0], color="blue", alpha=0.25)

                ax5 = plt.subplot(3, 2, 5)
                ax5.set_title("Unstable, Norm")
                for tr_i in lqr_idx0:
                    plt.plot(ts, v_norm_test_np[tr_i, :, 0], color="red", alpha=0.25)
                ax6 = plt.subplot(3, 2, 6)
                ax6.set_title("Stable, Norm")
                for tr_i in lqr_idx1:
                    plt.plot(ts, v_norm_test_np[tr_i, :, 0], color="blue", alpha=0.25)
                plt.savefig("%s/curve_e%05d.png"%(args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
                plt.close()

            for cfg_i in [0, 1, 2, 3]:
                cfg = init_cfg[cfg_i]
                if args.clf_pretrain_traj:
                    viz_P = init_p[cfg_i, 0]
                else:
                    viz_P = init_p[cfg_i]
                if args.no_viz:
                    continue
                rviz.plot_v_heatmap_whole(
                    ind_list=[[0, 1], [2, 3], [4, 5], [5, 6]],
                    f_list=[lambda x, pp: clf(x), lambda x, pp: quad(x, pp), lambda x, pp: clf(x) - quad(x, pp)],
                    img_path="%s/s0_lyapunov_e%04d_cfg%04d.png" % (args.viz_dir, epi, cfg_i),
                    args=args, viz_config=cfg, viz_P=viz_P, title="mu=%.3f v=%.3f w=%.3f" % (cfg[0], cfg[1], cfg[2]))
    return clf


def mask_loss(loss0, loss1, mask):
    assert loss0.shape == loss1.shape == mask.shape
    return utils.mask_mean(loss0, 1 - mask.float()) + utils.mask_mean(loss1, mask.float())

def pretrain_actor_multi(actor, init_s, lqr_trs, params, lqr_K_cuda):
    train_lr = args.pretrain_actor_setups[0]
    train_epochs = int(args.pretrain_actor_setups[1])  # TODO(temp) 5000
    print_freq = int(args.pretrain_actor_setups[2])
    viz_freq = int(args.pretrain_actor_setups[3])
    split = 0.8
    N = args.num_samples
    T = args.nt + 1
    if args.actor_pretrain_traj:
        n_train = int(N * T * split)
        traj = lqr_trs.reshape(N * T, args.X_DIM + args.N_REF + 1)
        params_m = params.unsqueeze(1).repeat([1, T, 1]).reshape(N * T, args.N_REF + 1)
        lqr_K_m = lqr_K_cuda.unsqueeze(1).repeat([1, T, 1, 1]).reshape(N * T, args.U_DIM, args.X_DIM)
        us = sscar_utils.cal_lqr(traj[:, :args.X_DIM], car_params, params_m, lqr_K_m, args).cuda()
        train_xs, test_xs = utils.split_tensor(traj, n_train)
        train_ks, test_ks = utils.split_tensor(lqr_K_m, n_train)
        train_us, test_us = utils.split_tensor(us, n_train)
    else:
        n_train = int(N * split)
        us = sscar_utils.cal_lqr(init_s[:, :args.X_DIM], car_params, params, lqr_K_cuda, args).cuda()
        train_xs, test_xs = utils.split_tensor(init_s, n_train)
        train_ks, test_ks = utils.split_tensor(lqr_K_cuda, n_train)
        train_us, test_us = utils.split_tensor(us, n_train)

    if args.actor_pretrain_rel_loss:
        criterion = lambda x, y: torch.mean((y - x) * (y - x) / torch.clip(torch.abs(y), 1e-4, 1e4))
    else:
        criterion = nn.MSELoss()

    crit_mat = nn.MSELoss()
    train_losses, test_losses, train_u_losses, test_u_losses, train_k_losses, test_k_losses = utils.get_n_meters(6)
    optimizer = torch.optim.RMSprop(actor.parameters(), train_lr)

    train_len = 8000
    test_len = 2000

    for epi in range(train_epochs):
        if args.always_new_sampling and epi % 100 == 0:
            init_s, params, lqr_P_cuda, lqr_K_cuda = init_samp_with_P_K(car_params, args)
            lqr_trs, _ = get_traj_lqr(init_s, lqr_K_cuda, car_params, params, args)
            traj = lqr_trs.reshape(N * T, args.X_DIM + args.N_REF + 1)
            params_m = params.unsqueeze(1).repeat([1, T, 1]).reshape(N * T, args.N_REF + 1)
            lqr_K_m = lqr_K_cuda.unsqueeze(1).repeat([1, T, 1, 1]).reshape(N * T, args.U_DIM, args.X_DIM)
            us = sscar_utils.cal_lqr(traj[:, :args.X_DIM], car_params, params_m, lqr_K_m, args).cuda()
            train_xs, test_xs = utils.split_tensor(traj, n_train)
            train_ks, test_ks = utils.split_tensor(lqr_K_m, n_train)
            train_us, test_us = utils.split_tensor(us, n_train)

        if args.actor_pretrain_mask:
            if epi % 10 == 0:
                train_indices = torch.randperm(train_xs.shape[0])[:train_len]
                test_indices = torch.randperm(test_xs.shape[0])[:test_len]
            u_nn_train, k_nn_train = actor(train_xs[train_indices], return_K=True)
            u_nn_test, k_nn_test = actor(test_xs[test_indices], return_K=True)
            loss_train_u = criterion(u_nn_train, train_us[train_indices]) * args.pre_actor_u_weight
            loss_test_u = criterion(u_nn_test, test_us[test_indices]) * args.pre_actor_u_weight
            loss_train_k = crit_mat(k_nn_train, train_ks[train_indices]) * args.pre_actor_k_weight
            loss_test_k = crit_mat(k_nn_test, test_ks[test_indices]) * args.pre_actor_k_weight
        else:
            u_nn_train, k_nn_train = actor(train_xs, return_K=True)
            u_nn_test, k_nn_test = actor(test_xs, return_K=True)
            loss_train_u = criterion(u_nn_train, train_us) * args.pre_actor_u_weight
            loss_test_u = criterion(u_nn_test, test_us) * args.pre_actor_u_weight
            loss_train_k = crit_mat(k_nn_train, train_ks) * args.pre_actor_k_weight
            loss_test_k = crit_mat(k_nn_test, test_ks) * args.pre_actor_k_weight

        loss_train = loss_train_u + loss_train_k
        loss_test = loss_test_u + loss_test_k

        utils.update(train_losses, loss_train)
        utils.update(test_losses, loss_test)
        utils.update(train_u_losses, loss_train_u)
        utils.update(test_u_losses, loss_test_u)
        utils.update(train_k_losses, loss_train_k)
        utils.update(test_k_losses, loss_test_k)

        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epi % print_freq == 0:
            print("%04d/%04d train: %.4f (%.4f) u: %.4f k: %.4f  test: %.4f (%.4f) u: %.4f k: %.4f" % (
                epi, train_epochs, train_losses.val, train_losses.avg, train_u_losses.val, train_k_losses.val,
                test_losses.val, test_losses.avg, test_u_losses.val, test_k_losses.val))

        if epi % 100 == 0:
            torch.save(actor.state_dict(), "%s/actor_model_%06d.ckpt" % (args.model_dir, epi))

    return actor


def compute_lyapunov_loss(trs, mask, v_est, prev_v):
    mask = mask.float()
    non_mask = 1 - mask
    # loss_cls = utils.mask_mean(v_est[:, 0] - 1.0, mask) - utils.mask_mean(v_est[:, 0] - 1.0, non_mask)
    loss_cls = utils.mask_mean(torch.relu(v_est[:, 0] - 1.0), mask[:, 0]) \
               + utils.mask_mean(torch.relu(1.0 - v_est[:, 0]), non_mask[:, 0])
    loss_roa = utils.mask_mean(torch.relu(v_est[:, 1:] - (1 - args.alpha_v) * v_est[:, :-1] + args.joint_v_margin), mask[:, 1:])
    loss_monot = utils.mask_mean(torch.square(v_est[:, :-1] - prev_v[:, 1:]), mask[:, 1:])

    loss_cls = loss_cls * args.weight_cls
    loss_roa = loss_roa * args.weight_roa
    loss_monot = loss_monot * args.weight_monot
    loss = loss_cls + loss_roa + loss_monot
    return loss, loss_cls, loss_roa, loss_monot

def plt_mask_scatter(data_x, data_y, mask, color_list, scale_list, label_list=None):
    for i in range(2):
        idx = torch.where(mask==i)[0]
        if label_list is not None:
            plt.scatter(to_np(data_x[idx]), to_np(data_y[idx]), color=color_list[i], s=scale_list[i], label=label_list[i])
        else:
            plt.scatter(to_np(data_x[idx]), to_np(data_y[idx]), color=color_list[i], s=scale_list[i])


def train_clf(clf, marked, roa_idx, init_s, trs, init_cfg, lqr_P, outer_i):
    train_lr = args.clf_setups[0]  # TODO 0.001
    train_epochs = int(args.clf_setups[1])  # TODO(temp) 1000
    print_freq = int(args.clf_setups[2])
    viz_freq = int(args.clf_setups[3])
    split = 0.8

    N, T, _ = trs.shape
    n_train, n_test = int(N * split), N - int(N * split)

    train_trs, test_trs = utils.split_tensor(trs, n_train)
    train_xs = train_trs.detach()
    test_xs = test_trs.detach()
    if args.cover_c:
        least_num = int(args.cover_least_ratio * N) * 2
        if torch.sum(marked.float()) < least_num:
            new_marked = torch.norm(trs[:, 0, :args.X_DIM], dim=-1) < \
                         torch.sort(torch.norm(trs[:, 0, :args.X_DIM], dim=-1))[0][least_num]
            print("CLF %.2f%% -> %.2f%% masked samples" % (
            torch.sum(marked.float(), dim=0).item() / N * 100,
            torch.sum(new_marked.float(), dim=0).item() / N * 100))
            mask = new_marked.unsqueeze(-1).unsqueeze(-1).repeat(1, T, 1)
        else:
            mask = marked.unsqueeze(-1).unsqueeze(-1).repeat(1, T, 1)
            print("CLF %.2f%% masked samples" % (torch.sum(marked.float(), dim=0).item() / N * 100))
    else:
        mask = marked.unsqueeze(-1).unsqueeze(-1).repeat(1, T, 1)
    train_mask, test_mask = utils.split_tensor(mask, n_train)
    train_losses, train_losses_cls, train_losses_roa, train_losses_monot, \
    test_losses, test_losses_cls, test_losses_roa, test_losses_monot = utils.get_n_meters(8)
    optimizer = torch.optim.RMSprop(clf.parameters(), train_lr)
    quadratic_pp = lambda x, pp: utils.quadratic(x, pp) / 2

    prev_v_train = {}
    prev_v_test = {}
    b_i = 0
    for epi in range(train_epochs):
        if epi % 10 == 0:
            if args.train_clf_batch:
                batch_size = int(n_train * 0.25)
                rand_idx = torch.randperm(n_train)[:batch_size]
            else:
                rand_idx = torch.tensor(range(n_train))
        v_est_train = clf(train_xs)
        v_est_test = clf(test_xs)
        if epi == 0:
            prev_v_train[b_i] = v_est_train.detach()
            prev_v_test[b_i] = v_est_test.detach()
        assert len(train_xs.shape) == 3
        assert len(v_est_train.shape) == 3
        train_loss, train_loss_cls, train_loss_roa, train_loss_monot = \
            compute_lyapunov_loss(train_xs[rand_idx], train_mask[rand_idx], v_est_train[rand_idx],
                                  prev_v_train[b_i][rand_idx])
        test_loss, test_loss_cls, test_loss_roa, test_loss_monot = \
            compute_lyapunov_loss(test_xs, test_mask, v_est_test, prev_v_test[b_i])

        utils.update(train_losses, train_loss)
        utils.update(train_losses_cls, train_loss_cls)
        utils.update(train_losses_roa, train_loss_roa)
        utils.update(train_losses_monot, train_loss_monot)

        utils.update(test_losses, test_loss)
        utils.update(test_losses_cls, test_loss_cls)
        utils.update(test_losses_roa, test_loss_roa)
        utils.update(test_losses_monot, test_loss_monot)

        if args.debug_viz:
            if epi % 100 == 0:
                scale = 1
                plt.figure(figsize=(8, 8))
                plt.subplot(2, 2, 1)
                plt_mask_scatter(train_xs[rand_idx, 0, 0], train_xs[rand_idx, 0, 1], train_mask[rand_idx],
                                 color_list=["red", "blue"], scale_list=[scale, scale])

                plt.subplot(2, 2, 2)
                plt_mask_scatter(test_xs[:, 0, 0], test_xs[:, 0, 1], test_mask,
                                 color_list=["red", "blue"], scale_list=[scale, scale])

                plt.subplot(2, 2, 3)
                # sort by v value (train set)
                plot(to_np(v_est_train[rand_idx, 0, 0]), to_np(v_est_train[rand_idx, 0, 0]),
                          to_np(train_mask[rand_idx, 0, 0]), func=lambda x_list: ["blue" if x == 1 else "red" for x in x_list])

                plt.subplot(2, 2, 4)
                plot(to_np(v_est_test[:, 0, 0]), to_np(v_est_test[:, 0, 0]), to_np(test_mask[:, 0, 0]),
                          func=lambda x_list: ["blue" if x == 1 else "red" for x in x_list])

                plt.savefig("%s/debug_%04d_%04d.png" % (args.viz_dir, outer_i, epi), bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epi % print_freq == 0:
            print("LYAPU %04d/%04d train[%5.2f%%]: %.4f (%.4f) c:%.4f r:%.4f m:%.4f  test[%5.2f%%]: %.4f (%.4f) c:%.4f r:%.4f m:%.4f M:%.4f" % (
                epi, train_epochs, to_np(torch.mean(train_mask.float()))*100, train_losses.val, train_losses.avg,
                train_losses_cls.val, train_losses_roa.val, train_losses_monot.val, to_np(torch.mean(test_mask.float()))*100,
                test_losses.val, test_losses.avg, test_losses_cls.val, test_losses_roa.val, test_losses_monot.val,
                to_np(torch.mean(v_est_test)),
            ))

        if epi == 0 or epi == train_epochs - 1 or epi % viz_freq == 0:
            if args.no_viz:
                continue
            for rand_i in [0, 1, 2, 3]:  # [0, 5, 10, 15]:
                rviz.plot_v_heatmap_whole(ind_list=[[0, 1], [2, 3], [4, 5], [5, 6]],
                                          f_list=[lambda x, pp: clf(x),
                                                  lambda x, pp: quadratic_pp(x[:, :args.X_DIM], pp),
                                                  lambda x, pp: clf(x) - quadratic_pp(x[:, :args.X_DIM], pp)],
                                          img_path="%s/s%04d_lyapunov_e%04d_i%04d.png" % (args.viz_dir, outer_i, epi, rand_i),
                                          args=args, viz_config=init_cfg[rand_i], viz_P=lqr_P[rand_i])
    return clf


def train_actor(actor, clf, init_v, params, marked, roa_idx, init_s, trs, c_in, c_out, c_in_mask, c_out_mask, outer_i,
                mid_i, lqr_K_cuda, new_set=None, dbg_init_s=None, dbg_params=None):
    train_lr = args.actor_setups[0]  # TODO 0.001
    train_epochs = int(args.actor_setups[1])  # TODO(temp) 1000
    print_freq = int(args.actor_setups[2])
    viz_freq = int(args.actor_setups[3])

    split = 0.8

    N, T, _ = trs.shape
    n_train, n_test = int(N * split), N - int(N * split)
    repeat = lambda x, n_rep: x.squeeze(1).repeat(1, n_rep, 1).reshape(-1, x.shape[-1])
    next_x = lambda x, _par: next_step_nn(x, actor, car_params, _par, args)[0]
    dv_nn_mid = lambda x, _par: torch.relu(clf(next_x(x.detach(), _par)) - (1 - args.alpha_v) * clf(x.detach()) + args.joint_v_margin)

    if args.train_actor_resampling:
        new_v, new_params, _, _, _, new_trs, _ = new_set
        train_trs, test_trs = utils.split_tensor(new_trs, n_train)
        train_params, test_params = utils.split_tensor(new_params, n_train)
    else:
        # TODO (temp) all those data handling techniques should merge to one function
        train_trs, test_trs = utils.split_tensor(trs, n_train)
        train_params, test_params = utils.split_tensor(params, n_train)

    train_xs = train_trs.reshape(-1, args.X_DIM + args.N_REF + 1)
    test_xs = test_trs.reshape(-1, args.X_DIM + args.N_REF + 1)
    train_c_in_mask, test_c_in_mask = utils.split_tensor(c_in_mask, n_train)
    train_c_out_mask, test_c_out_mask = utils.split_tensor(c_out_mask, n_train)
    train_cin_trj_mask = repeat(train_c_in_mask, T)
    train_cout_trj_mask = repeat(train_c_out_mask, T)
    test_cin_trj_mask = repeat(test_c_in_mask, T)
    test_cout_trj_mask = repeat(test_c_out_mask, T)
    train_params = repeat(train_params, T)
    test_params = repeat(test_params, T)

    train_losses, train_losses_cin, train_losses_cout, \
    test_losses, test_losses_cin, test_losses_cout = utils.get_n_meters(6)
    optimizer = torch.optim.RMSprop(actor.parameters(), train_lr)

    if args.statewise_actor_training:
        ratioes, = utils.get_n_meters(1)

    for epi in range(train_epochs):
        if args.statewise_actor_training:
            if epi % 100 == 0:
                if args.graduate_growing:
                    sample_ratio = 1.0 * (outer_i+1) / args.roa_iter * 10
                    sample_ratio = min(sample_ratio, 1.0)
                else:
                    sample_ratio = 1.0
                sat_s, sat_cfg = init_sampling(args, num_samples=100*args.num_samples, sample_ratio=sample_ratio)
                sat_s, sat_cfg = to_cuda(sat_s, sat_cfg)
                new_split = 0.99
                tr_s, te_s = utils.split_tensor(sat_s, int(new_split * sat_s.shape[0]))
                tr_cfg, te_cfg = utils.split_tensor(sat_cfg, int(new_split * sat_s.shape[0]))
            if epi % 1 == 0:
                train_len = te_s.shape[0]
                train_indices = torch.randperm(tr_s.shape[0])[:train_len]
            tr_mid = dv_nn_mid(tr_s[train_indices], tr_cfg[train_indices])
            te_mid = dv_nn_mid(te_s, te_cfg)
            loss_tr = torch.mean(tr_mid)
            loss_te = torch.mean(te_mid)

            utils.update(train_losses, loss_tr)
            utils.update(test_losses, loss_te)
            if epi==0:
                utils.update(ratioes, 0 * loss_te)
            else:
                utils.update(ratioes, torch.mean(new_marked.float()))

            loss_tr.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epi % print_freq == 0:
                print("ACTOR %04d/%04d train: %.4f (%.4f)  test: %.4f (%.4f)  ROA: %.4f (%.4f)" %(
                    epi, train_epochs, train_losses.val, train_losses.avg, test_losses.val, test_losses.avg,
                    ratioes.val, ratioes.avg
                ))

        else:
            train_mid = dv_nn_mid(train_xs, train_params)
            loss_train_cin = utils.mask_mean(train_mid, train_cin_trj_mask) * args.weight_cin
            loss_train_cout = utils.mask_mean(train_mid, train_cout_trj_mask) * args.weight_cout
            loss_train = loss_train_cin + loss_train_cout

            test_mid = dv_nn_mid(test_xs, test_params)
            loss_test_cin = utils.mask_mean(test_mid, test_cin_trj_mask) * args.weight_cin
            loss_test_cout = utils.mask_mean(test_mid, test_cout_trj_mask) * args.weight_cout

            loss_test = loss_test_cin + loss_test_cout

            utils.update(train_losses, loss_train)
            utils.update(train_losses_cin, loss_train_cin)
            utils.update(train_losses_cout, loss_train_cout)
            utils.update(test_losses, loss_test)
            utils.update(test_losses_cin, loss_test_cin)
            utils.update(test_losses_cout, loss_test_cout)

            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epi % print_freq == 0:
                print("ACTOR %04d/%04d train: %.4f (%.4f) cin[%5.2f%%]:%.4f cout[%5.2f%%]:%.4f  test: %.4f (%.4f) cin[%5.2f%%]:%.4f cout[%5.2f%%]:%.4f" % (
                    epi, train_epochs, train_losses.val, train_losses.avg, torch.mean(train_cin_trj_mask.float())*100,
                    train_losses_cin.val, torch.mean(train_cout_trj_mask.float())*100, train_losses_cout.val,
                    test_losses.val, test_losses.avg, torch.mean(test_cin_trj_mask.float())*100,
                    test_losses_cin.val, torch.mean(test_cout_trj_mask.float())*100, test_losses_cout.val,
                ))

        if epi == 0 or (
                epi == train_epochs - 1 and train_epochs != 1) or epi % viz_freq == 0 or args.train_actor_print_every:

            new_trs, new_us = get_traj_nn(init_s, actor, car_params, params, args)
            new_marked, new_roa_idx = analyze_traj_u(new_trs, new_us,
                                                     "NN controller (epoch=%02d, mid=%02d, iter=%02d)" % (
                                                     outer_i, mid_i, epi),
                                                     header="s%02d_o%02d_hist_policy_e%04d" % (outer_i, mid_i, epi),
                                                     skip_print=False, skip_viz=False, concise=True, args=args, trs_v=clf(new_trs))

            if args.no_viz:
                continue

            rviz.plot_sim_whole(ind_list=[[0, 1], [2, 3], [4, 5], [5, 6]], trs=new_trs, marked=new_marked,
                                img_path="%s/s%04d_mid%0d_sim_e%04d_%s.png" % (args.viz_dir, outer_i, mid_i, epi, "t%04d"),
                                args=args)

    return actor


def get_batch_ratio(marked):
    b_marked = marked.reshape((args.N_CFGS, -1))
    b_ratio = torch.mean(b_marked.float(), dim=-1)
    return b_ratio


def get_concised_roa_ratios(ratios):
    n_ratios = len(ratios)
    if n_ratios < 6:
        return ratios
    new_ratios = sorted(ratios)
    return [new_ratios[0], new_ratios[1],
            new_ratios[n_ratios // 2 - 1], new_ratios[n_ratios // 2],
            new_ratios[-2], new_ratios[-1]]


def roa_criterion(x, args, trs_v=None):
    if args.use_v_for_stable:
        if args.relative_roa_scale is not None:
            if args.traj_roa_alpha is not None:
                assert len(trs_v.shape)==3 and trs_v.shape[1] >= 20
                dec = trs_v[:,1:,0] <= (1-args.traj_roa_alpha) * trs_v[:, :-1, 0] + 0.001
                dec = torch.all(dec, dim=-1)
                return torch.logical_or(dec,
                        torch.logical_and(trs_v[:,0,0]<args.max_v_thres, trs_v[:,-1,0]<args.max_v_thres))
            else:
                return torch.logical_or(trs_v[:,-1,0] < args.relative_roa_scale * trs_v[:,0,0],
                                    torch.logical_and(trs_v[:,0,0]<args.max_v_thres, trs_v[:,-1,0]<args.max_v_thres))
        else:
            return trs_v[:,-1,0] < args.cutoff_norm
    else:
        if args.relative_roa_scale is not None:
            return torch.norm(x[:, -1, :args.X_DIM], dim=-1) < \
                   args.relative_roa_scale * torch.norm(x[:, 0, :args.X_DIM], dim=-1)
        else:
            return torch.norm(x[:, -1, :args.X_DIM], dim=-1) < args.cutoff_norm


def analyze_traj_u(trs, us, title, header, skip_print=False, skip_viz=False, concise=False, args=None, trs_v=None):
    marked = roa_criterion(trs, args, trs_v)
    if args.check_lqr_sim:
        if not args.no_viz:
            rviz.visualization_for_lqr(trs, marked, args)
    # Statistics about the LQR trajectories
    ratio = torch.mean(marked.float()).detach().cpu().item()
    roa_idx = marked.float().nonzero()
    if skip_print == False:
        if concise == False:
            print(utils.center_title(title, "-", 40))
        if args.traj_roa_alpha is not None:
            c_in, c_out = levelset_linesearch(trs_v[:,0], marked, None, None, None, None, None, args)
            c_mask = (trs_v[:,0]<c_in).float()
            print("ROA:  %.2f%%   [%.2f%%]" % (ratio * 100, torch.mean(c_mask) * 100))
        else:
            print("ROA:  %.2f%%" % (ratio * 100))
        if concise == False:
            roa_ratios = get_batch_ratio(marked)
            stat_roa_ratios = get_concised_roa_ratios(roa_ratios)
            print("omega  min: %.4f  max: %.4f ||  accel  min: %.4f  max: %.4f" % (
                torch.min(us[:, :, 0]), torch.max(us[:, :, 0]), torch.min(us[:, :, 1]), torch.max(us[:, :, 1])))
    if skip_viz == False:
        if not args.no_viz:
            rviz.plot_policy_hist(us[:, :, 0].flatten(), "omega", "%s/%s_0w.png" % (args.viz_dir, header))
            rviz.plot_policy_hist(us[:, :, 1].flatten(), "accel", "%s/%s_1a.png" % (args.viz_dir, header))
    return marked, roa_idx


def levelset_linesearch(init_v, marked, roa_idx, init_s, po_trs, out_i, mid_i, args):
    if torch.where(marked == True)[0].shape[0] > 0:
        c_in1 = torch.max(init_v[torch.where(marked == True)]).item()
        if torch.where(marked == False)[0].shape[0] > 0:
            c_in2 = torch.min(init_v[torch.where(marked == False)]).item()
        else:
            c_in2 = c_in1
        c_in = min(c_in1, c_in2)
        c_out = c_in * args.c_multiplier
    else:
        c_in = 0.0
        c_out = 0.01
    return c_in, c_out


def compute_and_plot_roa_vref(init_s):
    # TODO (temp) needs some work to make lqr_K work for this case (single lqr_K)
    v_list = []
    r_list = []
    for v_i, v_ref in enumerate(np.linspace(0.1, 35, 30)):
        params = torch.tensor([args.mu_choices[0], v_ref, 0.0]).float()
        _, lqr_K = solve_for_P_K(params, car_params, args)
        lqr_K_cuda = lqr_K.cuda()
        lqr_trs, _ = get_traj_lqr(init_s, lqr_K_cuda, car_params, params, args)
        marked = torch.norm(lqr_trs[:, -1, :args.X_DIM], dim=-1) < args.cutoff_norm
        ratio = torch.mean(marked.float()).detach().cpu().item()
        print(v_i, v_ref, ratio)
        v_list.append(v_ref)
        r_list.append(ratio)

    plt.plot(v_list, r_list, linewidth=2, color="navy")
    plt.xlabel("Reference velocity (m/s)")
    plt.ylabel("Normalized volume of the RoA")
    plt.savefig("%s/plot.png" % (args.viz_dir), bbox_inches='tight', pad_inches=0.1)
    plt.close()


def roa_sampling(init_s, init_v, c_in, c_out, params, args, clf):
    N = args.num_samples
    val_s = init_s[torch.where(init_v <= c_out * args.roa_multiplier)[0]]
    val_par = params[torch.where(init_v <= c_out * args.roa_multiplier)[0]]
    val_v = init_v[torch.where(init_v <= c_out * args.roa_multiplier)[0]]
    curr_min = to_np(torch.min(init_s, dim=0)[0])
    curr_max = to_np(torch.max(init_s, dim=0)[0])
    v_list = [val_v]
    val_list = [val_s]
    val_cfg_list = [val_par]
    val_num = val_s.shape[0]
    num_try = 0

    is_covered = False

    if args.train_actor_use_all:
        init_states = utils.uniform_sample_tsr(N, curr_min[:args.X_DIM], curr_max[:args.X_DIM])
        init_mus = utils.rand_choices_tsr(N, args.mu_choices)
        init_refs = utils.uniform_sample_tsr(N, curr_min[args.X_DIM + 1:args.X_DIM + args.N_REF],
                                             curr_max[args.X_DIM + 1:args.X_DIM + args.N_REF])
        init_idx = torch.tensor(range(N)).float().unsqueeze(-1)
        new_cfg = torch.cat([init_mus, init_refs, init_idx], axis=-1)
        new_s = torch.cat([init_states, new_cfg], axis=-1)
        new_s = new_s.cuda()
        v_ten = clf(new_s)
        c_in_mask = (v_ten < c_in)
        c_out_mask = (v_ten >= c_in)
        print("##################")
        print("Resample ALLLLLLLL")
        print("##################")
        return new_s.detach().cpu(), new_cfg, v_ten, c_in, c_out, c_in_mask, c_out_mask, is_covered

    if args.cover_c:
        new_s, new_cfg = init_sampling(args)
        new_s = new_s.cuda()
        new_cfg = new_cfg.cuda()
        new_v = clf(new_s)
        new_val_idx = new_s[torch.where(new_v <= c_out * args.roa_multiplier)[0]]
        least_ratio = args.cover_least_ratio
        least_num = int(least_ratio * args.num_samples)
        if new_val_idx.shape[0] < least_num:
            _, sorted_indices = torch.sort(new_v, dim=0)

            new_c_in = new_v[sorted_indices[least_num]].item()
            new_c_out = new_v[sorted_indices[least_num * 2]].item()

            new_c_in_mask = torch.zeros_like(new_v)
            new_c_out_mask = torch.zeros_like(new_v)
            new_c_in_mask[sorted_indices[least_num]] = 1.0
            new_c_out_mask[sorted_indices[least_num * 2]] = 1.0
            is_covered = True

            print("#### COVERED:  %.2f%% points Ci:%.4f Co:%.4f -> Ci:%.4f Co:%.4f min:%.4f max:%.4f" % (
                2.0 * least_num / args.num_samples * 100, c_in, c_out, new_c_in, new_c_out, torch.min(new_v).item(),
                torch.max(new_v).item()
            ))
            return new_s.detach().cpu(), \
                   new_cfg.detach().cpu(), \
                   new_v.detach().cpu(), new_c_in, new_c_out, new_c_in_mask, new_c_out_mask, is_covered
        else:
            c_in_mask = (new_v < c_in)
            c_out_mask = (torch.logical_and(new_v >= c_in, new_v < c_out))
            print("#### LevelSet: %.2f%% points Ci:%.4f Co:%.4f min:%.4f max:%.4f" % (
                2.0 * least_num / args.num_samples * 100, c_in, c_out, torch.min(new_v).item(),
                torch.max(new_v).item()))
            return new_s.detach().cpu(), \
                   new_cfg.detach().cpu(), \
                   new_v.detach().cpu(), c_in, c_out, c_in_mask, c_out_mask, is_covered

    else:
        while val_num < N and num_try < 10000:
            num_try += 1
            init_states = utils.uniform_sample_tsr(N, curr_min[:args.X_DIM], curr_max[:args.X_DIM])
            init_mus = utils.rand_choices_tsr(N, args.mu_choices)
            init_refs = utils.uniform_sample_tsr(N, curr_min[args.X_DIM + 1:args.X_DIM + args.N_REF],
                                                 curr_max[args.X_DIM + 1:args.X_DIM + args.N_REF])
            init_idx = torch.tensor(range(N)).float().unsqueeze(-1)
            new_cfg = torch.cat([init_mus, init_refs, init_idx], axis=-1)
            new_s = torch.cat([init_states, new_cfg], axis=-1)
            new_s = new_s.cuda()
            new_cfg = new_cfg.cuda()
            new_v = clf(new_s)
            new_val_idx = torch.where(new_v <= c_out * args.roa_multiplier)[0]
            if new_val_idx.shape[0] > 0:
                v_list.append(new_v[new_val_idx])
                val_list.append(new_s[new_val_idx])
                val_cfg_list.append(new_cfg[new_val_idx])
                val_num += new_val_idx.shape[0]

        if num_try == 10000 and val_num < N:
            print("RESET")
            tensor = init_s
            cfg_tensor = params
            v_ten = init_v
        else:
            v_ten = torch.cat(v_list, dim=0)
            tensor = torch.cat(val_list, dim=0)
            cfg_tensor = torch.cat(val_cfg_list, dim=0)
            tensor = tensor[:N]
            cfg_tensor = cfg_tensor[:N]
            v_ten = v_ten[:N]

        c_in_mask = (v_ten < c_in)
        c_out_mask = (torch.logical_and(v_ten >= c_in, v_ten < c_out))

        old_ratio = torch.mean((val_v <= c_in).float())
        new_ratio = torch.mean((v_ten <= c_in).float())
        print(
            "Resample %d times, with %5d(%.3f)->%5d(%.3f) valids" % (num_try, val_s.shape[0], old_ratio, N, new_ratio))
        return tensor.detach().cpu(), cfg_tensor.detach().cpu(), v_ten, c_in, c_out, c_in_mask, c_out_mask, is_covered


def test_lqr_roa(init_s, params, lqr_K_cuda):
    lqr_trs, lqr_us = get_traj_lqr(init_s, lqr_K_cuda, car_params, params, args)
    marked, roa_idx = analyze_traj_u(lqr_trs, lqr_us, "LQR controller", header="s0_hist_lqr", skip_print=False,
                                     skip_viz=True, args=args)


def get_bdry_points(points, v, c, dim):
    bdry = points[torch.where((v - c) * (v) < 0)[0]]
    bdry = bdry[:, [dim[0], dim[1]]]
    bdry = to_np(bdry)
    hull = ConvexHull(points=bdry)
    return bdry[hull.vertices]


def prep_viz_points(s_mins, s_maxs, ind0, ind1, cfg, nx, ny, args):
    mu, vref, wref = cfg
    new_s_mins = np.zeros_like(s_mins)
    new_s_maxs = np.zeros_like(s_maxs)
    new_s_mins[ind0] = s_mins[ind0]
    new_s_mins[ind1] = s_mins[ind1]
    new_s_maxs[ind0] = s_maxs[ind0]
    new_s_maxs[ind1] = s_maxs[ind1]

    lin0 = torch.linspace(s_mins[ind0], s_maxs[ind0], nx)
    lin1 = torch.linspace(s_mins[ind1], s_maxs[ind1], ny)
    mesh = torch.stack(torch.meshgrid(lin0, lin1), dim=-1).reshape((nx * ny, 2))
    init_states = torch.zeros(nx*ny, 7)
    init_states[:, ind0] = mesh[:, 0]
    init_states[:, ind1] = mesh[:, 1]
    ones = torch.ones_like(init_states[:, 0:1])
    tensor = torch.cat([
        init_states, ones * mu, ones * vref, ones * wref, torch.tensor(range(nx * ny)).float().unsqueeze(-1)
    ], dim=-1)
    return tensor.cuda()



def plot_roa_act_lqr(clf, actor, actor_2, clf_2, iter_list, actor_list, clf_list, car_params, args):
    str_list=[r"$x_{err}$ (m)",
              r"$y_{err}$ (m)",
              r"$\delta_{err}$ (rad)",
              r"$v_{err}$ (m/s)",
              r"$\psi_{err} (rad)$",
              r"$\dot{\psi}_{err}$ (rad/s)",
              r"$\beta_{err}$ (rad)"]
    # cfg_list = [(1.0, 10.0, 0.0), (0.1, 10.0, 0.0)]
    if args.multi_initials is not None:
        cfg_list = [
            (1.0, 3.0, 0.0),
            (1.0, 5.0, 0.0),
            (1.0, 10.0, 0.0),
            (1.0, 15.0, 0.0),
            (1.0, 20.0, 0.0),
            (1.0, 25.0, 0.0),
            (1.0, 30.0, 0.0),
            (0.1, 3.0, 0.0),
            (0.1, 5.0, 0.0),
            (0.1, 10.0, 0.0),
            (0.1, 15.0, 0.0),
            (0.1, 20.0, 0.0),
            (0.1, 25.0, 0.0),
            (0.1, 30.0, 0.0)]
    else:
        cfg_list = [(1.0, 10.0, 0.0), (0.1, 10.0, 0.0)]
    dim_list = [[0, 1], [2, 3], [4, 5], [5, 6]]
    gain_list = [2, 3, 2, 2]
    for cfg_i, cfg in enumerate(cfg_list):
        for dim_i, dim in enumerate(dim_list):
            mock_args = MockArgs()
            mock_args.num_workers = args.num_workers
            mock_args.num_samples = args.num_samples
            if clf_list is not None:
                mock_args.s_mins = args.s_mins
                mock_args.s_maxs = args.s_maxs
            else:
                mock_args.s_mins = [0.0] * 7
                mock_args.s_maxs = [0.0] * 7
                mock_args.s_mins[dim[0]] = args.s_mins[dim[0]] * gain_list[dim_i]
                mock_args.s_mins[dim[1]] = args.s_mins[dim[1]] * gain_list[dim_i]
                mock_args.s_maxs[dim[0]] = args.s_maxs[dim[0]] * gain_list[dim_i]
                mock_args.s_maxs[dim[1]] = args.s_maxs[dim[1]] * gain_list[dim_i]
            mock_args.mu_choices = [cfg[0]]
            mock_args.ref_mins = [cfg[1], cfg[2]]
            mock_args.ref_maxs = [cfg[1], cfg[2]]
            mock_args.X_DIM = args.X_DIM
            mock_args.N_REF = args.N_REF
            mock_args.U_DIM = args.U_DIM
            mock_args.controller_dt = args.controller_dt
            mock_args.gpus = args.gpus
            mock_args.tanh_w_gain = args.tanh_w_gain
            mock_args.tanh_a_gain = args.tanh_a_gain

            init_s, params, init_p, init_k = init_samp_with_P_K(car_params, args=mock_args, quiet=False)

            lqr_trs, lqr_us = get_traj_lqr(init_s, init_k, car_params, params, args)
            nn_trs, nn_us = get_traj_nn(init_s, actor, car_params, params, args)
            if clf_2 is not None:
                nn2_trs, nn2_us = get_traj_nn(init_s, actor_2, car_params, params, args)
            N, T, _ = lqr_trs.shape

            lqr_vq = (utils.quadratic(lqr_trs.reshape(N*T,-1)[:, :args.X_DIM], init_p[0]) / 2 ) .reshape(N, T, 1)
            nn_vn = clf(nn_trs)
            if clf_2 is not None:
                nn2_vn = clf_2(nn2_trs)


            nn_marked, nn_roa_idx = analyze_traj_u(nn_trs, nn_us, "NN Controller", "DEBUG", skip_print=False, skip_viz=True, args=args, trs_v=nn_vn)
            nn_c_in, nn_c_out = levelset_linesearch(nn_vn[:,0,0], nn_marked, nn_roa_idx, init_s, nn_trs, 0, 0, args)
            nn_c_in_idx = torch.where(nn_vn[:, 0, 0] <= nn_c_in)[0]
            nn_c_in_points = torch.stack([init_s[nn_c_in_idx, dim[0]], init_s[nn_c_in_idx, dim[1]]], dim=-1)
            print("#NN  ", nn_c_in_points.shape)


            if clf_2 is not None:
                nn2_marked, nn2_roa_idx = analyze_traj_u(nn2_trs, nn2_us, "NN Controller2", "DEBUG", skip_print=False,
                                                       skip_viz=True, args=args, trs_v=nn2_vn)
                nn2_c_in, nn2_c_out = levelset_linesearch(nn2_vn[:, 0, 0], nn2_marked, nn2_roa_idx, init_s, nn2_trs, 0, 0, args)
                nn2_c_in_idx = torch.where(nn2_vn[:, 0, 0] <= nn2_c_in)[0]
                nn2_c_in_points = torch.stack([init_s[nn2_c_in_idx, dim[0]], init_s[nn2_c_in_idx, dim[1]]], dim=-1)
                print("#NN2  ", nn2_c_in_points.shape)


            lqr_marked, lqr_roa_idx = analyze_traj_u(lqr_trs, lqr_us, "LQR Baseline", "DEBUG", skip_print=False,
                                                     skip_viz=True, args=args, trs_v=lqr_vq)
            lqr_c_in, lqr_c_out = levelset_linesearch(lqr_vq[:, 0, 0], lqr_marked, lqr_roa_idx, init_s, lqr_trs, 0, 0,
                                                      args)
            lqr_c_in_idx = torch.where(lqr_vq[:, 0, 0] <= lqr_c_in)[0]
            lqr_c_in_points = torch.stack([init_s[lqr_c_in_idx, dim[0]], init_s[lqr_c_in_idx, dim[1]]], dim=-1)
            print("#LQR ", lqr_c_in_points.shape)

            if clf_list is not None:
                c_list = []
                for clf_i in range(len(clf_list)):
                    nn_i_trs, nn_i_us = get_traj_nn(init_s, actor_list[clf_i], car_params, params, args)
                    nn_i_vn = clf_list[clf_i](nn_i_trs)
                    nn_i_marked, nn_i_roa_idx = analyze_traj_u(nn_i_trs, nn_i_us, "NN Controller", "DEBUG",
                                                               skip_print=False,
                                                               skip_viz=True, args=args, trs_v=nn_i_vn)
                    nn_i_c_in, nn_i_c_out = levelset_linesearch(nn_i_vn[:, 0, 0], nn_i_marked, nn_i_roa_idx, init_s,
                                                                nn_i_trs, 0, 0,
                                                                args)
                    nn_i_c_in_idx = torch.where(nn_i_vn[:, 0, 0] <= nn_i_c_in)[0]
                    nn_i_c_in_points = torch.stack([init_s[nn_i_c_in_idx, dim[0]], init_s[nn_i_c_in_idx, dim[1]]],
                                                   dim=-1)
                    c_i_ratio = nn_i_c_in_idx.shape[0] / N
                    c_list.append(c_i_ratio)

                plt.plot(iter_list, c_list, label="Ours", color="blue")
                plt.axhline(lqr_c_in_idx.shape[0]/N, label="LQR", color="red", linestyle="--")
                plt.legend()
                plt.xlabel("Iterations")
                plt.ylabel("Relative RoA volume")
                plt.savefig("%s/iter_roa_dim_%d_%d.png" % (args.viz_dir, cfg_i, dim_i), bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()

            # plot c level set
            mock_args.num_samples = args.num_samples * 10
            NN = mock_args.num_samples
            s_viz, p_viz, _, _ = init_samp_with_P_K(car_params, args=mock_args, quiet=False, no_p_k=True)

            lqr_v_viz = (utils.quadratic(s_viz[:, :args.X_DIM], init_p[0]) / 2).reshape(NN, 1)
            nn_v_viz = clf(s_viz)
            if clf_2 is not None:
                nn2_v_viz = clf_2(s_viz)

            ax = plt.gca()
            if clf_2 is not None:
                print(lqr_c_in, nn_c_in, nn2_c_in)
            else:
                print(lqr_c_in, nn_c_in)

            import matplotlib as mpl
            SMALL_SIZE = 20
            MEDIUM_SIZE = 32
            BIGGER_SIZE = 48
            plt_d = {"font.size": SMALL_SIZE,
                     "axes.titlesize": SMALL_SIZE,
                     "axes.labelsize": SMALL_SIZE,
                     "xtick.labelsize": SMALL_SIZE,
                     "ytick.labelsize": SMALL_SIZE,
                     "legend.fontsize": SMALL_SIZE,
                     "figure.titlesize": SMALL_SIZE}
            trun_cmap = utils.truncate_colormap(plt.get_cmap('magma'), 0.0, 1.0)
            with mpl.rc_context(plt_d):
                # plot the contour
                ctr_nx = 100
                ctr_ny = 100
                viz_v_pts = prep_viz_points(mock_args.s_mins, mock_args.s_maxs, dim[0], dim[1], cfg, ctr_nx, ctr_ny, args)
                NNN = viz_v_pts.shape[0]
                viz_v_clf = clf(viz_v_pts)
                plt.contourf(to_np(viz_v_pts[:, dim[0]]).reshape((ctr_nx, ctr_ny)),
                             to_np(viz_v_pts[:, dim[1]]).reshape((ctr_nx, ctr_ny)),
                             to_np(viz_v_clf[:, 0]).reshape((ctr_nx, ctr_ny)),
                             levels=30, cmap=trun_cmap, alpha=0.7, #locator=ticker.LogLocator(base=3, numticks=20),
                             )
                lqr_bdry = get_bdry_points(s_viz, lqr_v_viz, lqr_c_in, dim)
                nn_bdry = get_bdry_points(s_viz, nn_v_viz, nn_c_in, dim)
                if clf_2 is not None:
                    nn2_bdry = get_bdry_points(s_viz, nn2_v_viz, nn2_c_in, dim)
                ax.add_patch(Polygon(nn_bdry, antialiased=True, fill=False, facecolor="blue",
                                     edgecolor="blue", linestyle="--", linewidth=2.5, label="Ours", alpha=1.0))
                if clf_2 is not None:
                    ax.add_patch(Polygon(nn2_bdry, antialiased=True, fill=False, facecolor="green",
                                         edgecolor="green", linestyle="--", linewidth=2.5, label="CLF", alpha=1.0))
                ax.add_patch(Polygon(lqr_bdry, antialiased=True, fill=False, facecolor="red",
                                     edgecolor="red", linestyle="--", linewidth=2.5, label="LQR", alpha=1.0))

                plt.axis("scaled")
                plt.legend(fontsize=int(SMALL_SIZE*0.75))
                plt.xlabel(str_list[dim[0]], fontsize=SMALL_SIZE)
                plt.ylabel(str_list[dim[1]], fontsize=SMALL_SIZE)
                plt.xticks(fontsize=SMALL_SIZE)
                plt.yticks(fontsize=SMALL_SIZE)
                plt.xlim(mock_args.s_mins[dim[0]], mock_args.s_maxs[dim[0]])
                plt.ylim(mock_args.s_mins[dim[1]], mock_args.s_maxs[dim[1]])
                plt.colorbar()
                plt.savefig("%s/new_roa_dim_%d_%d.png" % (args.viz_dir, cfg_i, dim_i), bbox_inches='tight', pad_inches=0.1)
                plt.close()

            mock_args.num_samples = args.num_samples


            ax = plt.subplot(1, 2, 1)
            plt_mask_scatter(nn_trs[:,0,dim[0]], nn_trs[:,0,dim[1]], nn_marked.float(),
                             ["red", "blue"], [1.0, 1.0], ["Unstable", "Stable"])
            plt.axis("scaled")
            plt.xlabel(str_list[dim[0]])
            plt.ylabel(str_list[dim[1]])

            if nn_c_in_points.shape[0]>4:
                nn_hull = ConvexHull(points=to_np(nn_c_in_points))
                nn_patch = Polygon(to_np(nn_c_in_points)[nn_hull.vertices], color="purple", label="RoA", alpha=0.5)
                ax.add_patch(nn_patch)
            ax.legend()
            ax.set_title("NN (Stable:%d%%, ROA:%d%%)"%(torch.mean(nn_marked.float()).item()*100, 100*nn_c_in_idx.shape[0]/N))

            ax = plt.subplot(1, 2, 2)


            plt_mask_scatter(lqr_trs[:, 0, dim[0]], lqr_trs[:, 0, dim[1]], lqr_marked.float(),
                             ["red", "blue"], [1.0, 1.0], ["Unstable", "Stable"])
            plt.axis("scaled")

            if lqr_c_in_points.shape[0] > 4:
                lqr_hull = ConvexHull(points=to_np(lqr_c_in_points))
                lqr_patch = Polygon(to_np(lqr_c_in_points)[lqr_hull.vertices], color="purple", label="RoA", alpha=0.5)
                ax.add_patch(lqr_patch)
            ax.legend()
            ax.set_title("LQR(Stable:%d%%, ROA:%d%%)" % (torch.mean(lqr_marked.float()).item() * 100, 100 * lqr_c_in_idx.shape[0] / N))
            plt.savefig("%s/roa_dim_%d_%d.png" % (args.viz_dir, cfg_i, dim_i), bbox_inches='tight', pad_inches=0.1)
            plt.close()


def quad_clf(lqr_trs, lqr_P_cuda):
    lqr_vs = []
    for i in range(lqr_trs.shape[1]):
        lqr_v = utils.quadratic_multi(lqr_trs[:, i, :args.X_DIM], lqr_P_cuda) / 2
        lqr_vs.append(lqr_v)
    return torch.stack(lqr_vs, dim=1)


def main():
    utils.set_random_seed(args.random_seed)
    utils.setup_data_exp_and_logger(args, offset=1)
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    init_s, params, lqr_P_cuda, lqr_K_cuda = init_samp_with_P_K(car_params, args)
    if args.plot_roa_v:
        compute_and_plot_roa_vref(init_s)  # inspect RoA wrt v_ref
        exit()

    if args.test_lqr_roa:
        test_lqr_roa(init_s, params, lqr_K_cuda)
        exit()

    clf = roa_nn.CLF(None, args)
    actor = roa_nn.Actor(None, args)
    utils.safe_load_nn(clf, args.clf_pretrained_path, load_last=args.load_last, key="clf_")
    utils.safe_load_nn(actor, args.actor_pretrained_path, load_last=args.load_last, key="actor_")
    if args.gpus is not None:
        clf, actor = to_cuda(clf, actor)

    if args.actor_2:
        args.single_clf = True
        clf_2 = roa_nn.CLF(None, args)
        actor_2 = roa_nn.Actor(None, args)
        args.single_clf = False
        args.clf_2_pretrained_path = args.actor_2_pretrained_path
        utils.safe_load_nn(clf_2, args.clf_2_pretrained_path, load_last=args.load_last, key="clf_")
        utils.safe_load_nn(actor_2, args.actor_2_pretrained_path, load_last=args.load_last, key="actor_")
        if args.gpus is not None:
            clf_2, actor_2 = to_cuda(clf_2, actor_2)
    else:
        actor_2 = None
        clf_2 = None

    if args.actor_path_list:
        actor_list = []
        iter_list = []
        clf_list = []
        for path in args.actor_path_list:
            clf_i = roa_nn.CLF(None, args)
            actor_i = roa_nn.Actor(None, args)
            utils.safe_load_nn(clf_i, "%s/models/clf_model_e%03d_000.ckpt"%(args.actor_pretrained_path, path), load_last=False, key="clf_")
            utils.safe_load_nn(actor_i, "%s/models/actor_model_e%03d_000.ckpt"%(args.actor_pretrained_path, path), load_last=False, key="actor_")
            if args.gpus is not None:
                clf_i, actor_i = to_cuda(clf_i, actor_i)
            iter_list.append(path)
            actor_list.append(actor_i)
            clf_list.append(clf_i)
    else:
        iter_list = None
        actor_list = None
        clf_list = None

    # Collect LQR trajectories
    lqr_trs, lqr_us = get_traj_lqr(init_s, lqr_K_cuda, car_params, params, args)
    marked, roa_idx = analyze_traj_u(lqr_trs, lqr_us, "LQR controller", header="s0_hist_lqr", skip_print=False,
                                     skip_viz=False, args=args, trs_v=quad_clf(lqr_trs, lqr_P_cuda))

    # Training
    if not args.skip_clf_pretraining:
        clf = pretrain_clf_multi(clf, init_s, lqr_trs, params, lqr_P_cuda)
        torch.save(clf.state_dict(), "%s/clf_model_pre.ckpt" % (args.model_dir))
    if not args.skip_actor_pretraining:
        actor = pretrain_actor_multi(actor, init_s, lqr_trs, params, lqr_K_cuda)
        torch.save(actor.state_dict(), "%s/actor_model_pre.ckpt" % (args.model_dir))

    for out_i in range(args.roa_iter):
        if args.plot_roa_act_lqr:
            plot_roa_act_lqr(clf, actor, actor_2, clf_2, iter_list, actor_list, clf_list, car_params, args)

        if args.always_new_sampling:
            init_s, params, lqr_P_cuda, lqr_K_cuda = init_samp_with_P_K(car_params, args)
            lqr_trs, lqr_us = get_traj_lqr(init_s, lqr_K_cuda, car_params, params, args)
            marked, roa_idx = analyze_traj_u(lqr_trs, lqr_us, "LQR Baseline",
                                             header="s%02d_o%02d_hist_policy" % (out_i, 0), skip_print=False,
                                             skip_viz=True, args=args, trs_v=quad_clf(lqr_trs, lqr_P_cuda))

        print("OUT_I=%04d" % (out_i))
        dbg_init_s = init_s.detach().clone()
        dbg_params = params.detach().clone()
        po_trs, po_us = get_traj_nn(init_s, actor, car_params, params, args)
        marked, roa_idx = analyze_traj_u(po_trs, po_us, "Before CLF, NN controller (epoch=%02d, mid=%02d)" % (out_i, 0),
                                         header="s%02d_o%02d_hist_policy" % (out_i, 0), skip_print=False, skip_viz=True,
                                         args=args, trs_v=clf(po_trs))
        if args.skip_training:
            return
        if args.skip_clf_training == False:
            for ii in range(args.clf_outer_iters):
                clf = train_clf(clf, marked, roa_idx, init_s, po_trs, params, lqr_P_cuda, out_i)
                if args.clf_outer_iters>1:
                    analyze_traj_u(po_trs, po_us,
                                                     "CLF Update, NN controller (epoch=%02d, mid=%02d)" % (out_i, 0),
                                                     header="s%02d_o%02d_hist_policy" % (out_i, 0), skip_print=False,
                                                     skip_viz=True,
                                                     args=args, trs_v=clf(po_trs))

        torch.save(clf.state_dict(), "%s/clf_model_e%03d_%03d.ckpt" % (args.model_dir, out_i, 0))

        init_v = clf(init_s)
        c_in, c_out = levelset_linesearch(init_v, marked, roa_idx, init_s, po_trs, out_i, 0, args)
        c_in_mask = (init_v < c_in)
        c_out_mask = (torch.logical_and(init_v >= c_in, init_v < c_out))

        if args.train_actor_resampling:
            new_s, new_params, new_v, c_in, c_out, c_in_mask, c_out_mask, is_covered = \
                roa_sampling(init_s, init_v, c_in, c_out, params, args, clf)
            new_P, new_K = solve_for_P_K(new_params, car_params, args)
            if args.gpus is not None:
                new_s, new_params, new_P_cuda, new_K_cuda = to_cuda(new_s, new_params, new_P, new_K)
            new_trs, new_us = get_traj_nn(new_s, actor, car_params, new_params, args)
            new_marked, new_roa_idx = analyze_traj_u(new_trs, new_us,
                                                     "NEW SAMPLE FOR NN controller (epoch=%02d, mid=%02d)" % (out_i, 0),
                                                     header="s%02d_o%02d_hist_policy" % (out_i, 0), skip_print=True,
                                                     skip_viz=True, args=args, trs_v=clf(new_trs))
            new_set = new_v, new_params, new_marked, new_roa_idx, new_s, new_trs, new_K
            # TODO
            actor = train_actor(actor, clf, init_v, params, marked, roa_idx, init_s, po_trs, c_in, c_out,
                                c_in_mask, c_out_mask, out_i, 0, lqr_K_cuda, new_set=new_set, dbg_init_s=dbg_init_s, dbg_params=dbg_params)

            torch.save(actor.state_dict(), "%s/actor_model_e%03d_%03d.ckpt" % (args.model_dir, out_i, 0))

        else:
            actor = train_actor(actor, clf, init_v, params, marked, roa_idx, init_s, po_trs, c_in, c_out,
                                c_in_mask, c_out_mask, out_i, 0, lqr_K_cuda)
            torch.save(actor.state_dict(), "%s/actor_model_e%03d_%03d.ckpt" % (args.model_dir, out_i, 0))


def hyperparameters():
    parser = argparse.ArgumentParser("ROA Training")
    parser.add_argument('--exp_name', type=str, default="ROA_")
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--viz_freq', type=int, default=500)
    parser.add_argument('--save_freq', type=int, default=2000)
    # SIM CONFIGS
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--nt', type=int, default=500)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--s_mins', type=float, nargs="+", default=[-1, -1, -1, -1, -1, -1, -1.0])
    parser.add_argument('--s_maxs', type=float, nargs="+", default=[1, 1, 1, 1, 1, 1, 1.0])

    # TRAINING CONFIGS
    parser.add_argument('--clf_pretrained_path', type=str, default=None)
    parser.add_argument('--actor_pretrained_path', type=str, default=None)

    # VIZ CONFIGS
    parser.add_argument('--num_viz_samples', type=int, default=20)
    parser.add_argument('--stable_norm', type=float, default=0.01)

    # NEW SIM CONFIGS
    parser.add_argument('--mu_choices', type=float, nargs="+", default=[1.0, 0.01])
    parser.add_argument('--roa_iter', type=int, default=20)
    parser.add_argument('--policy_iter', type=int, default=1)
    parser.add_argument('--lqr_factor', type=float, default=1)
    parser.add_argument('--controller_dt', type=float, default=0.01)
    parser.add_argument('--clip_angle', action='store_true', default=False)
    parser.add_argument('--plot_roa_v', action='store_true', default=False)
    parser.add_argument('--cutoff_norm', type=float, default=0.1)
    parser.add_argument('--v_ref', type=float, default=5.0)

    # CLF
    parser.add_argument('--clf_mode', type=str, choices=["ellipsoid", "nn", "diag", "var_ell"], default="ellipsoid")
    parser.add_argument('--clf_init_type', type=str, choices=["eye", "lqr", "rand"], default="lqr")
    parser.add_argument('--clf_hiddens', type=int, nargs="+", default=[64, 64])
    parser.add_argument('--check_lqr_sim', action='store_true', default=False)
    parser.add_argument('--pretrain_with_init_states', action='store_true', default=False)
    parser.add_argument('--skip_clf_pretraining', action='store_true', default=False)

    # Actor
    parser.add_argument('--actor_mode', type=str, choices=["mat", "nn", "mat_nn"], default="mat_nn")
    parser.add_argument('--actor_init_type', type=str, choices=["lqr", "rand"], default="lqr")
    parser.add_argument('--actor_hiddens', type=int, nargs="+", default=[64, 64])
    parser.add_argument('--actor_clip_u', action='store_true', default=False)
    parser.add_argument('--tanh_w_gain', type=float, default=None)
    parser.add_argument('--tanh_a_gain', type=float, default=None)
    parser.add_argument('--u_limit', type=float, nargs="+", default=None)
    parser.add_argument('--actor_nn_res_ratio', type=float, default=0.1)
    parser.add_argument('--skip_actor_pretraining', action='store_true', default=False)

    parser.add_argument('--c_multiplier', type=float, default=1.1)

    parser.add_argument('--alpha_v', type=float, default=0.1)
    parser.add_argument('--weight_cls', type=float, default=1.0)
    parser.add_argument('--weight_roa', type=float, default=1.0)
    parser.add_argument('--weight_monot', type=float, default=1.0)
    parser.add_argument('--weight_cin', type=float, default=1.0)
    parser.add_argument('--weight_cout', type=float, default=1.0)

    # CONFIGS for multiple mu, vref, wref
    parser.add_argument('--multi', action='store_true', default=False)
    parser.add_argument('--clf_e2e', action='store_true', default=False)
    parser.add_argument('--actor_e2e', action='store_true', default=False)

    parser.add_argument('--ref_mins', type=float, nargs="+", default=None)
    parser.add_argument('--ref_maxs', type=float, nargs="+", default=None)
    parser.add_argument('--pre_actor_u_weight', type=float, default=1.0)
    parser.add_argument('--pre_actor_k_weight', type=float, default=1.0)

    parser.add_argument('--batch_op', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--num_workers', type=int, default=32)

    parser.add_argument('--clf_ell_scale', action='store_true', default=False)
    parser.add_argument('--clf_pretrain_traj', action='store_true', default=False)
    parser.add_argument('--clf_pretrain_mask', action='store_true', default=False)
    parser.add_argument('--load_last', action='store_true', default=False)
    parser.add_argument('--actor_pretrain_rel_loss', action='store_true', default=False)
    parser.add_argument('--actor_pretrain_traj', action='store_true', default=False)
    parser.add_argument('--actor_pretrain_mask', action='store_true', default=False)

    parser.add_argument('--no_viz', action='store_true', default=False)

    parser.add_argument('--train_clf_batch', action='store_true', default=False)
    parser.add_argument('--new_sampling', action='store_true', default=False)
    parser.add_argument('--train_actor_resampling', action='store_true', default=False)
    parser.add_argument('--train_actor_use_all', action='store_true', default=False)
    parser.add_argument('--train_actor_print_every', action='store_true', default=False)
    parser.add_argument('--skip_clf_training', action='store_true', default=False)
    parser.add_argument('--test_lqr_roa', action='store_true', default=False)
    parser.add_argument('--roa_multiplier', type=float, default=1.0)

    # TODO lr, epochs, print_freq, viz_freq
    parser.add_argument('--pretrain_clf_setups', type=float, nargs="+", default=[0.00002, 20000, 1000, 5000])
    parser.add_argument('--pretrain_actor_setups', type=float, nargs="+", default=[0.00005, 20000, 100, 1000])
    parser.add_argument('--clf_setups', type=float, nargs="+", default=[0.00001, 1000, 100, 1000])
    parser.add_argument('--actor_setups', type=float, nargs="+", default=[0.00001, 10, 10, 50])

    parser.add_argument('--skip_training', action='store_true', default=False)
    parser.add_argument('--always_new_sampling', action='store_true', default=False)
    parser.add_argument('--test_setups', type=int, nargs="+", default=[2, 17, 17])
    parser.add_argument('--test_just_plot', type=str, nargs="+", default=None)

    parser.add_argument('--debug_viz', action='store_true', default=False)
    parser.add_argument('--clf_no_exp_scale', action="store_true", default=False)
    parser.add_argument('--clf_ell_extra', action="store_true", default=False)
    parser.add_argument('--clf_ell_extra_tanh', action="store_true", default=False)
    parser.add_argument('--clf_ell_extra_xnorm', action="store_true", default=False)
    parser.add_argument('--test_use_lqr', action='store_true', default=False)

    # TODO train ROA level set estimator
    parser.add_argument('--joint_pretrained_path', type=str, default=None)
    parser.add_argument('--load_data_path', type=str, default=None)
    parser.add_argument('--net_hiddens', type=int, nargs="+", default=[64, 64])
    parser.add_argument('--net_pretrained_path', type=str, default=None)
    parser.add_argument('--num_configs', type=int, default=None)
    parser.add_argument('--net_setups', type=float, nargs="+", default=[1e-4, 10000, 100, 1000, 500])
    parser.add_argument('--grid_params', action='store_true', default=False)
    parser.add_argument('--net_exp', action='store_true', default=False)
    parser.add_argument('--net_rel_loss', action='store_true', default=False)
    parser.add_argument('--allow_factor', type=float, default=1.0)

    # TODO Phase II, planning
    parser.add_argument('--plan_setups', type=float, nargs="+", default=[1e-4, 10000, 100, 1000, 500])

    parser.add_argument('--relative_roa_scale', type=float, default=None)
    parser.add_argument('--complex', action='store_true', default=False)
    parser.add_argument('--cap_seg', type=int, default=2)
    parser.add_argument('--skip_viz', type=int, default=1)
    parser.add_argument('--simple_plan', action='store_true', default=False)
    parser.add_argument('--use_real', action='store_true', default=False)

    parser.add_argument('--cover_c', action='store_true', default=False)
    parser.add_argument('--cover_least_ratio', type=float, default=0.1)

    # TODO Planning
    parser.add_argument('--no_actor', action='store_true', default=False)
    parser.add_argument('--accumulate_traj', action='store_true', default=False)
    parser.add_argument('--use_actor', action='store_true', default=False)
    parser.add_argument('--no_lqr', action='store_true', default=False)
    parser.add_argument('--use_sample', action='store_true', default=False)
    parser.add_argument('--clf_nt', type=int, default=None)
    parser.add_argument('--lqr_nt', type=int, default=None)
    parser.add_argument('--use_middle', action='store_true', default=False)



    # TODO nn CLF training
    parser.add_argument('--new_clf_pretraining', action='store_true', default=False)
    parser.add_argument('--v_margin', type=float, default=0.01)
    parser.add_argument('--pre_clf_zero_weight', type=float, default=1.0)
    parser.add_argument('--pre_clf_dec_weight', type=float, default=1.0)
    parser.add_argument('--pre_clf_cls_weight', type=float, default=1.0)
    parser.add_argument('--new_sampling_freq', type=int, default=100)

    parser.add_argument('--plot_roa_act_lqr', action='store_true', default=False)
    parser.add_argument('--use_v_for_stable', action='store_true', default=False)

    # TODO actor training
    parser.add_argument('--statewise_actor_training', action='store_true', default=False)
    parser.add_argument('--config', type=int, default=0)
    parser.add_argument('--joint_v_margin', type=float, default=0.0)
    parser.add_argument('--graduate_growing', action='store_true', default=False)

    parser.add_argument('--max_v_thres', type=float, default=-1.0)

    # TODO again for planner
    parser.add_argument('--multi_nns', action='store_true', default=False)
    parser.add_argument('--joint_pretrained_path0', type=str, default=None)
    parser.add_argument('--joint_pretrained_path1', type=str, default=None)
    parser.add_argument('--clf_pretrained_path0', type=str, default=None)
    parser.add_argument('--clf_pretrained_path1', type=str, default=None)
    parser.add_argument('--actor_pretrained_path0', type=str, default=None)
    parser.add_argument('--actor_pretrained_path1', type=str, default=None)

    parser.add_argument('--traj_roa_alpha', type=float, default=None)
    parser.add_argument('--clf_outer_iters', type=int, default=1.0)

    parser.add_argument('--new_ring_seg', type=int, default=5)
    parser.add_argument('--always_same', action='store_true', default=False)
    parser.add_argument('--all_ice', action='store_true', default=False)
    parser.add_argument('--race', action='store_true', default=False)
    parser.add_argument('--race_angle', type=float, default=50)
    parser.add_argument('--wild', action='store_true', default=False)

    parser.add_argument('--clf_ell_eye', action='store_true', default=False)
    parser.add_argument('--normalize', action="store_true", default=False)

    parser.add_argument('--methods', type=str, nargs="+", default=["ours", "clf"])  # ["rl", "mpc", "clf", "ours"]
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--deviation_thres', type=float, default=3.0)
    parser.add_argument('--rlp_paths', type=str, nargs="+", default=None)
    parser.add_argument('--auto_rl', action='store_true', default=False)
    parser.add_argument('--load_from', type=str, default=None)

    # TODO for single clf baseline
    parser.add_argument('--single_clf', action="store_true", default=False)
    parser.add_argument('--mono_actor_pretrained_path', type=str, default=None)

    parser.add_argument('--actor_2', action='store_true', default=False)
    parser.add_argument('--actor_2_pretrained_path', type=str, default=None)
    parser.add_argument('--actor_path_list', type=int, nargs="+", default=None)

    parser.add_argument('--animation', action='store_true', default=False)
    parser.add_argument('--max_keep_viz_trials', type=int, default=100)
    parser.add_argument('--use_d', action='store_true', default=False)

    # TODO for multi iter training
    parser.add_argument('--model_iter', type=int, nargs="+", default=None)

    # TODO for RL & ours multi iter comparison
    parser.add_argument('--from_file', type=str, default=None)
    parser.add_argument('--pret', action='store_true', default=False)
    parser.add_argument('--plot_map_only', action='store_true', default=False)


    parser.add_argument('--multi_initials', action="store_true", default=False)
    parser.add_argument('--multi_ani', action='store_true', default=False)
    parser.add_argument('--select_me', type=int, nargs="+", default=None)

    parser.add_argument("--mbpo_paths", type=str, nargs="+", default=None)
    parser.add_argument("--pets_paths", type=str, nargs="+", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = hyperparameters()

    if args.config == 1:
        args.nt = 50
        args.ref_mins = [3.0, 0.0]
        args.ref_maxs = [30.0, 0.0]
        args.s_mins = [-1.5] * 7
        args.s_maxs = [1.5] * 7

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

    if len(args.s_mins)==1:
        args.s_mins = args.s_mins * 7
    if len(args.s_maxs)==1:
        args.s_maxs = args.s_maxs * 7

    t1 = time.time()
    car_params = sscar_utils.VehicleParameters()
    main()
    t2 = time.time()
    print("ROA training finished in %.4fs" % (t2 - t1))
