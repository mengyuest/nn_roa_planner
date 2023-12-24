import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import time

import cgw_sim_choi as choi
import torch
import roa_utils as utils
import scipy.io
import torch.nn as nn
import cgw_utils as misc
from roa_nn import CGW_Actor, CGW_CLF


def generate_init_s(args):
    # TODO use gait idx
    gait_all = args.gait_data
    gait_th = args.gait_th
    x_list = []
    th_list = []
    for i in range(args.n_theta):
        x_ref = gait_all[i][0].reshape((1, 4))
        dx = utils.uniform_sample_tsr(args.n_trials, args.dx_min, args.dx_max).type_as(gait_all[0])
        x = x_ref + dx
        th = torch.full_like(x[:, 0:1], gait_th[i])
        x_list.append(x)
        th_list.append(th)

    # TODO check whether th and th_real match
    x_list = torch.cat(x_list, dim=0)
    th_list = torch.cat(th_list, dim=0)
    if args.gpus is not None:
        x_list = x_list.cuda()
        th_list = th_list.cuda()
    return x_list, th_list


def build_net(key, args):
    d = {
        "actor": {"prefix": "actor_", "class": CGW_Actor, "path": args.actor_pretrained_path},
        "clf": {"prefix": "clf_", "class": CGW_CLF, "path": args.clf_pretrained_path},
    }
    assert key in d
    net = d[key]["class"](args)
    utils.safe_load_nn(net, d[key]["path"], load_last=args.load_last, key=d[key]["prefix"])
    if args.gpus is not None:
        net = net.cuda()
    return net

def build_net_full(key, pretrained_path, args):
    print(key, pretrained_path)
    d = {
        "actor": {"prefix": "actor_", "class": CGW_Actor, "path": pretrained_path},
        "clf": {"prefix": "clf_", "class": CGW_CLF, "path": pretrained_path},
    }
    assert key in d
    net = d[key]["class"](args)
    utils.safe_load_nn(net, d[key]["path"], load_last=args.load_last, key=d[key]["prefix"])
    if args.gpus is not None:
        net = net.cuda()
    return net


def compute_loss(x, th, xf, mask, val, clf, actor, epi, args):
    cache_d = {}
    N, T, _ = x.shape
    x = torch.cat((x, th), dim=-1)
    x3d = x
    x = x.reshape(N * T, 5)
    ref = clf.get_ref(x, num_ths=args.n_theta)
    mask3d = mask
    cum3d = torch.cumsum(mask3d, dim=1)
    ref3d = ref.reshape(N, T, 4)
    mask = mask.reshape(N * T, 1)

    val3d = val
    val = val.reshape(N*T, 1)

    first_mask = torch.where(torch.logical_and(cum3d[:,:,0] == 1, mask3d[:,:,0] == 1))

    x_0 = x3d[first_mask[0], 0, :-1]
    x_1 = x3d[first_mask[0], first_mask[1], :-1]
    ref_0 = ref3d[first_mask[0], 0]
    ref_1 = ref3d[first_mask[0], first_mask[1]]

    # stable-based (only terminal states)
    pos_dbg = torch.norm(x_1 - ref_1, dim=-1) < 0.1
    pos_idx = torch.where(pos_dbg)

    if args.test_r18:
        pos_mask = first_mask[0][pos_idx[0]]
        full_idx = torch.tensor(list(range(N))).type_as(x)
        neg_mask = full_idx[np.setdiff1d(range(full_idx.shape[0]), utils.to_np(pos_mask))].long()

        sec_mask = torch.where(torch.logical_and(cum3d[:, :, 0] == 1, mask3d[:, :, 0] == 1))
        thd_mask = torch.where(torch.logical_and(cum3d[:, :, 0] == 2, mask3d[:, :, 0] == 1))
        frth_mask = torch.where(torch.logical_and(cum3d[:, :, 0] == 3, mask3d[:, :, 0] == 1))
        fifth_mask = torch.where(torch.logical_and(cum3d[:, :, 0] == 4, mask3d[:, :, 0] == 1))

        ind_list = [(0, 1), (0, 2), (0, 3), (2, 3)]
        labels = ["q1 (rad)", "q2 (rad)", "q1d (rad/s)", "q2d (rad/s)"]
        xmin = [-0.25, -0.5, -1.25, -1.5]
        xmax = [0.25, 0.5, 0.25, 2.5]
        gait_th = args.gait_th

        assert args.n_theta==1
        for m_i, mask in enumerate([None, first_mask, sec_mask, thd_mask, frth_mask, fifth_mask]):
            plt.figure(figsize=(8, 8))
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                ind1, ind2 = ind_list[i]
                plt.plot(utils.to_np(args.gait_data[0][:, ind1]),
                         utils.to_np(args.gait_data[0][:, ind2]),
                         color="black", linestyle="--", label="gait")
                if m_i == 0:
                    plt.scatter(utils.to_np(x3d[pos_mask, 0, ind1]),
                                utils.to_np(x3d[pos_mask, 0, ind2]), color="blue", label='stable', s=4)

                    plt.scatter(utils.to_np(x3d[neg_mask, 0, ind1]),
                                utils.to_np(x3d[neg_mask, 0, ind2]), color="red", label='unstable', s=4)
                else:
                    plt.scatter(utils.to_np(x3d[mask[0], mask[1], ind1]),
                                utils.to_np(x3d[mask[0], mask[1], ind2]), color="blue", label='stable', s=4)
                plt.xlim(xmin[ind1], xmax[ind1])
                plt.ylim(xmin[ind2], xmax[ind2])
                plt.xlabel(labels[ind1])
                plt.ylabel(labels[ind2])
                plt.tight_layout()
            plt.savefig("%s/e%05d_debug_r_%.3f_m%02d.png" % (args.viz_dir, 0, gait_th[0], m_i),
                        bbox_inches='tight', pad_inches=0.1)
            plt.close()
        exit()

    pos_mask = first_mask[0][pos_idx[0]]
    full_idx = torch.tensor(list(range(N))).type_as(x)
    neg_mask = full_idx[np.setdiff1d(range(full_idx.shape[0]), utils.to_np(pos_mask))].long()

    # TODO (debug on roa VIZ)
    if epi==0:
        plot_roa(x3d, x3d[first_mask[0][pos_idx], first_mask[1][pos_idx]], pos_mask, neg_mask, args)

    v_zero = clf(torch.cat((ref, x[:, -1:]), dim=-1), ref=ref, num_ths=args.n_theta)
    v = clf(x, ref=ref, num_ths=args.n_theta)
    v3d = v.reshape((N, T, 1))
    loss_zero = v_zero ** 2
    loss_dec = torch.relu(v3d[:, 1:, :] - (1-args.alpha_v) * v3d[:, :-1, :] + args.v_thres)
    loss_bdry = torch.mean(torch.relu(v3d[pos_mask.long(), 0, :] - args.pos_thres)) + \
                torch.mean(torch.relu(args.neg_thres - v3d[neg_mask, 0, :]))

    loss_dq1 = torch.relu(x[:, 2:3] - args.dq1_thres)

    # only consider the first switch, and also the valid
    anti_cum3d = torch.logical_and(cum3d < 1, val3d).float()
    anti_cum = anti_cum3d.reshape((N * T, 1))
    anti_cum3d_dec = anti_cum3d[:, :-1]

    loss_zero = mask_mean(loss_zero, anti_cum) * args.w_zero
    loss_dec = mask_mean(loss_dec, anti_cum3d_dec) * args.w_dec
    loss_bdry = loss_bdry * args.w_bdry
    loss_dq1 = mask_mean(loss_dq1, anti_cum) * args.w_dq1

    loss = loss_zero + loss_dec + loss_bdry + loss_dq1

    cache_d["x3d"] = x3d
    cache_d["v3d"] = v3d
    cache_d["v"] = v

    return loss, loss_zero, loss_dec, loss_bdry, loss_dq1, cache_d


def mask_mean(x, mask):
    assert len(x.shape) == len(mask.shape)
    ndim = len(x.shape) - 1
    for i in range(ndim):
        assert x.shape[i] == mask.shape[i]
    return torch.mean(x * mask) / torch.clamp(torch.mean(mask), min=0.0001)


def get_gait_data(args):
    # ori_gait_data = scipy.io.loadmat("walker/result_ral_main_gait.mat")['gait']
    tmp_data = np.load("walker/gait_data_all.npz", allow_pickle=True)
    if args.gait_idx_indices is not None:
        all_gait_data = tmp_data['data'][args.gait_idx_indices]
        all_gait_theta = tmp_data["thetas"][args.gait_idx_indices]
    else:
        all_gait_data = tmp_data['data'][args.gait_idx_begin: args.gait_idx_end]
        all_gait_theta = tmp_data["thetas"][args.gait_idx_begin: args.gait_idx_end]

    all_gait_data = [torch.from_numpy(each_gait) for each_gait in all_gait_data]
    all_gait_theta = torch.from_numpy(all_gait_theta).float()
    if args.gpus is not None:
        all_gait_data = [each_gait.cuda() for each_gait in all_gait_data]
        all_gait_theta = all_gait_theta.cuda()

    return all_gait_data, all_gait_theta


def create_optimizer(actor, clf, args):
    if args.actor_only:
        params_list = actor.parameters()
    elif args.clf_only:
        params_list = clf.parameters()
    else:
        params_list = list(actor.parameters()) + list(clf.parameters())
    optimizer = torch.optim.RMSprop(params_list, args.lr)
    return optimizer


def generate_traj(train_x, train_th, actor, args):
    mask = zeros_as(train_x[:, 0:1])
    valid = ones_as(train_x[:, 0:1])
    xf = zeros_as(train_x[:, 0:1])
    x_list = [train_x]
    th_list = [train_th]
    xf_list = [xf]
    mask_list = [mask]
    val_list = [valid]

    x = train_x
    for ti in range(args.nt):
        prev_x = x_list[-1]
        u = actor(torch.cat((x.detach(), train_th), dim=-1))
        x, xf, mask, valid = get_next_x(prev_x, x, xf, u, valid, args)
        x_list.append(x)
        th_list.append(train_th)
        xf_list.append(xf)
        mask_list.append(mask)
        val_list.append(valid)

    x_list = torch.stack(x_list, dim=1)
    th_list = torch.stack(th_list, dim=1)
    xf_list = torch.stack(xf_list, dim=1)
    mask_list = torch.stack(mask_list, dim=1)  # (N, T+1, 1)
    val_list = torch.stack(val_list, dim=1)  # (N, T+1, 1)

    return x_list, th_list, xf_list, mask_list, val_list


def main():
    utils.set_seed_and_exp(args)

    args.params = choi.create_params()
    args.gait_data, args.gait_th = get_gait_data(args)
    args.gait_data_np = [utils.to_np(each_gait_data) for each_gait_data in args.gait_data]
    args.gait_th_np = utils.to_np(args.gait_th)

    # TODO initial conditions (x, target_th), shape (N, 5)
    train_x, train_th = generate_init_s(args)
    train_xf = torch.zeros_like(train_x[:, 0:1])

    clf = build_net("clf", args)
    actor = build_net("actor", args)

    losses, zero_losses, dec_losses, bdry_losses, dq1_losses, mask_meter, valid_meter, vtr_meter, xf_meter \
        = utils.get_n_meters(9)
    optimizer = create_optimizer(actor, clf, args)

    for epi in range(args.num_epochs):
        if args.n_theta == 29:
            the_viz_th_list = [0, 7, 14, 21, 28]
        else:
            the_viz_th_list = range(args.n_theta)
        if args.skip_training:
            for r_i in the_viz_th_list:
                for viz_ti in [0]:
                    plot_heat_map_new(args.gait_th_np[r_i], args.gait_data, r_i, viz_ti, clf,
                                  title="e%05d_heat_r%02d_t%03d.png" % (epi, r_i, viz_ti), args=args)
            continue

        dbg_t1=time.time()
        # TODO (trajectory sampling!)
        new_data=False
        if args.regen_init and epi % args.regen_freq == 0 and epi != 0:
            train_x, train_th = generate_init_s(args)
            new_data=True
        dbg_t2 = time.time()
        # TODO collect samples along trajectory
        if args.clf_only==False or (epi==0 or new_data):
            x_list, th_list, xf_list, mask_list, val_list = generate_traj(train_x, train_th, actor, args)
        else:
            x_list = x_list.detach()
            th_list = th_list.detach()
            xf_list = xf_list.detach()
            mask_list = mask_list.detach()
            val_list = val_list.detach()
        dbg_t3 = time.time()
        # TODO compute the loss
        loss, loss_zero, loss_dec, loss_bdry, loss_dq1, cache_d = \
            compute_loss(x_list, th_list, xf_list, mask_list, val_list, clf, actor, epi, args)
        dbg_t4 = time.time()
        utils.update(losses, loss)
        utils.update(zero_losses, loss_zero)
        utils.update(dec_losses, loss_dec)
        utils.update(bdry_losses, loss_bdry)
        utils.update(dq1_losses, loss_dq1)
        utils.update(mask_meter, torch.mean(mask_list))
        utils.update(valid_meter, torch.mean(val_list))
        utils.update(vtr_meter, torch.mean(
            (torch.logical_and(val_list[:, -1], torch.sum(mask_list, dim=1)>0)).float()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dbg_t5 = time.time()
        # TODO evaluate on testing
        if epi % args.print_freq == 0 or epi == args.num_epochs - 1:
            print("%05d/%05d L:%.3f(%.3f) z:%.3f(%.3f) d:%.3f(%.3f) b:%.3f(%.3f) dq1:%.3f(%.3f) | m:%.3f(%.3f) v:%.3f(%.3f) r:%.3f(%.3f)" %(
                epi, args.num_epochs,
                losses.val, losses.avg, zero_losses.val, zero_losses.avg,
                dec_losses.val, dec_losses.avg, bdry_losses.val, bdry_losses.avg,
                dq1_losses.val, dq1_losses.avg, mask_meter.val, mask_meter.avg,
                valid_meter.val, valid_meter.avg, vtr_meter.val, vtr_meter.avg,
            ))

        if epi % args.save_freq == 0 or epi == args.num_epochs - 1:
            torch.save(clf.state_dict(), "%s/clf_e%05d.ckpt" % (args.model_dir, epi))
            torch.save(actor.state_dict(), "%s/actor_e%05d.ckpt" % (args.model_dir, epi))

        if (not args.skip_viz) and epi % args.viz_freq == 0 or epi == args.num_epochs - 1:
            x_list_np = utils.to_np(cache_d["x3d"])
            v_list_np = utils.to_np(cache_d["v3d"])

            # TODO do visualization
            # TODO lyapunov curves

            for r_i in the_viz_th_list: #[0,1,2]: #[0, 15, 24]:
                batch0 = r_i * args.n_trials
                batch1 = (r_i+1) * args.n_trials
                if r_i == 15:
                    if args.not_plot_phase==False:
                        misc.plot_phase_full(x_list_np[batch0:batch1],
                                None, utils.to_np(args.gait_data[r_i]), epi, args=args,
                                new_title="%02d_%.3f_"%(r_i, args.gait_data[r_i][0, 0]))

                for viz_ti in [0]: #[0, 25, 50]:
                    plot_heat_map_new(args.gait_th_np[r_i], args.gait_data, r_i, viz_ti, clf,
                                  title="e%05d_heat_r%02d_t%03d.png"%(epi, r_i, viz_ti), args=args)


def plot_heat_map(v, x, th, all_gait_data, epi, r_i, viz_ti, clf, title, args):
    ind_list = [(0, 1), (1, 2), (2, 3), (1, 3)]
    labels = ["q1 (rad)", "q2 (rad)", "q1d (rad/s)", "q2d (rad/s)"]
    x_mins = [-0.05, -0.1, -0.25, -0.5]
    x_maxs = [0.05, 0.1, 0.25, 0.5]
    plt.figure(figsize=(8, 8))

    nx = args.viz_nx
    ny = args.viz_ny

    # ref point x,
    for i in range(4):
        id1, id2 = ind_list[i]
        plt.subplot(2, 2, i + 1)

        err1 = torch.linspace(x_mins[id1], x_maxs[id1], nx).type_as(all_gait_data[r_i])
        err2 = torch.linspace(x_mins[id2], x_maxs[id2], ny).type_as(all_gait_data[r_i])
        err12_x, err12_y = torch.meshgrid(err1, err2)
        err12_x = err12_x.reshape((nx * ny, ))
        err12_y = err12_y.reshape((nx * ny, ))

        x_ref = all_gait_data[r_i][viz_ti:viz_ti+1].tile([nx * ny, 1])
        x_ref = torch.cat([x_ref, th * torch.ones_like(x_ref[:, 0:1])], dim=-1)
        x_ref[:, id1] = x_ref[:, id1] + err12_x
        x_ref[:, id2] = x_ref[:, id2] + err12_y

        if args.gpus is not None:
            x_ref = x_ref.cuda()

        v_ref = clf(x_ref, num_ths=1)

        v_ref = utils.to_np(v_ref).reshape((nx, ny))

        plt.imshow(v_ref.T, origin="lower")
        plt.xlabel(labels[id1])
        plt.ylabel(labels[id2])

    plt.savefig("%s/%s" % (args.viz_dir,title), bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_heat_map_new(th, all_gait_data, r_i, viz_ti, clf, title, args):
    ind_list = [(0, 1), (1, 2), (2, 3), (1, 3)]
    labels = [
        r"$q_1$ (rad)",
        r"$q_2$ (rad)",
        r"$\dot{q}_1$ (rad/s)",
        r"$\dot{q}_2$ (rad/s)"
        ]
    x_mins = [-0.05, -0.1, -0.25, -0.5]
    x_maxs = [0.05, 0.1, 0.25, 0.5]
    plt.figure(figsize=(8, 8))

    nx = 100
    ny = 100
    SMALL_SIZE = 20
    trun_cmap = utils.truncate_colormap(plt.get_cmap('magma'), 0.0, 1.0)
    # ref point x,
    for i in range(4):
        plt.figure(figsize=(8,8))
        id1, id2 = ind_list[i]
        # plt.subplot(2, 2, i + 1)
        err1 = torch.linspace(x_mins[id1], x_maxs[id1], nx).type_as(all_gait_data[r_i])
        err2 = torch.linspace(x_mins[id2], x_maxs[id2], ny).type_as(all_gait_data[r_i])
        err12_x, err12_y = torch.meshgrid(err1, err2)
        err12_x = err12_x.reshape((nx * ny, ))
        err12_y = err12_y.reshape((nx * ny, ))
        x_ref = all_gait_data[r_i][viz_ti:viz_ti+1].tile([nx * ny, 1])
        x_ref = torch.cat([x_ref, th * torch.ones_like(x_ref[:, 0:1])], dim=-1)
        x_ref[:, id1] = x_ref[:, id1] + err12_x
        x_ref[:, id2] = x_ref[:, id2] + err12_y
        if args.gpus is not None:
            x_ref = x_ref.cuda()
        v_ref = clf(x_ref, num_ths=1)

        plt.contourf(to_np(x_ref[:, id1]).reshape((nx, ny)),
                     to_np(x_ref[:, id2]).reshape((nx, ny)),
                     to_np(v_ref[:, 0]).reshape((nx, ny)),
                     levels=30, cmap=trun_cmap, alpha=0.7,  # locator=ticker.LogLocator(base=3, numticks=20),
                     )

        nn_bdry = get_bdry_points(x_ref.detach().cpu(), v_ref.detach().cpu(), 0.9, [id1, id2])
        ax = plt.gca()
        ax.add_patch(Polygon(nn_bdry, antialiased=True, fill=False, facecolor="blue",
                             edgecolor="blue", linestyle="--", linewidth=2.5, label="Ours", alpha=1.0))

        plt.legend(fontsize=int(SMALL_SIZE * 0.75))
        plt.xlabel(labels[id1], fontsize=SMALL_SIZE)
        plt.ylabel(labels[id2], fontsize=SMALL_SIZE)
        plt.xticks(fontsize=SMALL_SIZE)
        plt.yticks(fontsize=SMALL_SIZE)
        xmins = np.min(to_np(x_ref), axis=0)
        xmaxs = np.max(to_np(x_ref), axis=0)
        plt.xlim(xmins[id1], xmaxs[id1])
        plt.ylim(xmins[id2], xmaxs[id2])
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=int(SMALL_SIZE * 0.75))
        plt.locator_params(axis='y', nbins=6)
        plt.locator_params(axis='x', nbins=4)
        plt.savefig("%s/%s_%d.png" % (args.viz_dir, title.split(".png")[0], i), bbox_inches='tight', pad_inches=0.1)
        plt.close()

from scipy.spatial import ConvexHull
def get_bdry_points(points, v, c, dim):
    bdry = points[torch.where((v - c) * (v) < 0)[0]]
    bdry = bdry[:, [dim[0], dim[1]]]
    # bdry = to_np(bdry)
    hull = ConvexHull(points=bdry)
    return bdry[hull.vertices]


def to_np(tensor):
    return tensor.detach().cpu().numpy()



def get_next_x(prev_x, x, xf, u, valid, args):
    l = 1.0
    for tti in range(args.num_sim_steps):
        xdot = choi.compute_xdot(x, u, use_torch=True, args=args)
        x = x + xdot * (args.dt / args.num_sim_steps)
    x_mid = choi.compute_fine(x, prev_x, args)
    x_plus = choi.compute_impact(x_mid, use_torch=True)
    xf_plus = xf + l * torch.sin(x_mid[:, 0:1] + x_mid[:, 1:2]) - l * torch.sin(x_mid[:, 0:1])
    mask = choi.detect_switch(x, prev_x, args)
    x_next = x * (1 - mask) + x_plus * mask
    xf_next = xf * (1 - mask) + xf_plus * mask
    curr_valid = torch.logical_and(torch.logical_and(x_next[:, 0:1] <= 0.5, x_next[:, 0:1] >= -0.5),
                                   torch.logical_and(x_next[:, 1:2] <= 1.0, x_next[:, 1:2] >= -1.0))
    valid_next = torch.logical_and(valid, curr_valid)

    return x_next, xf_next, mask, valid_next


def plot_roa(x3d, x_1, pos_mask, neg_mask, args):
    ind_list = [(0, 1), (0, 2), (0, 3), (2, 3)]
    labels = ["q1 (rad)", "q2 (rad)", "q1d (rad/s)", "q2d (rad/s)"]
    xmin = [-0.2, -0.4, -1.25, -1.5]
    xmax = [0.2, 0.4, 0.25, 2.5]
    gait_th = args.gait_th
    for r_i in range(args.n_theta): #[0, 1, 2]: #[3, 6, 10, 12]:
        for ti in range(args.nt):
            # if ti % 3 == 0:
            if ti == 0:
                plt.figure(figsize=(14, 7))
                for i in range(4):
                    plt.subplot(2, 4, i * 2 + 1)
                    ind1, ind2 = ind_list[i]
                    plt.plot(utils.to_np(args.gait_data[r_i][:, ind1]),
                             utils.to_np(args.gait_data[r_i][:, ind2]),
                             color="black", linestyle="--", label="gait")

                    pos_sub_mask = torch.where(x3d[pos_mask, ti, -1] == gait_th[r_i])[0]
                    neg_sub_mask = torch.where(x3d[neg_mask, ti, -1] == gait_th[r_i])[0]

                    plt.scatter(utils.to_np(x3d[pos_mask[pos_sub_mask], ti, ind1]),
                                utils.to_np(x3d[pos_mask[pos_sub_mask], ti, ind2]), color="blue", label='stable', s=8)

                    plt.scatter(utils.to_np(x3d[neg_mask[neg_sub_mask], ti, ind1]),
                                utils.to_np(x3d[neg_mask[neg_sub_mask], ti, ind2]), color="red", label='unstable', s=8)

                    plt.xlim(xmin[ind1], xmax[ind1])
                    plt.ylim(xmin[ind2], xmax[ind2])
                    plt.xlabel(labels[ind1])
                    plt.ylabel(labels[ind2])
                    plt.tight_layout()


                for i in range(4):
                    plt.subplot(2, 4, i * 2 + 2)
                    ind1, ind2 = ind_list[i]
                    plt.plot(utils.to_np(args.gait_data[r_i][:, ind1]),
                             utils.to_np(args.gait_data[r_i][:, ind2]),
                             color="black", linestyle="--", label="gait")
                    sub_mask = torch.where(x_1[:, -1] == gait_th[r_i])[0]
                    plt.scatter(utils.to_np(x_1[sub_mask, ind1]),
                                utils.to_np(x_1[sub_mask, ind2]), color="blue", label='stable', s=8)
                    plt.xlim(xmin[ind1], xmax[ind1])
                    plt.ylim(xmin[ind2], xmax[ind2])
                    plt.xlabel(labels[ind1])
                    plt.ylabel(labels[ind2])
                    plt.tight_layout()


                plt.suptitle("Phase (t=%04d/%04d)" % (ti, args.nt))
                plt.savefig("%s/e%05d_debug_r_%.3f_t%04d.png" % (args.viz_dir, 0, gait_th[r_i], ti), bbox_inches='tight', pad_inches=0.1)
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="cgw_train")
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=50000)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--nt', type=int, default=200)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--num_sim_steps', type=int, default=2)
    parser.add_argument('--qp_bound', type=float, default=4.0)
    parser.add_argument('--u_gain', type=float, default=4.0)

    parser.add_argument('--n_theta', type=int, default=29)
    parser.add_argument('--gait_idx_begin', type=int, default=0)
    parser.add_argument('--gait_idx_end', type=int, default=29)
    parser.add_argument('--gait_idx_indices', nargs="+", type=int, default=None)
    parser.add_argument('--th_min', type=float, default=0.1305)
    parser.add_argument('--th_max', type=float, default=0.1305)
    parser.add_argument('--dx_min', type=float, nargs="+", default=None)
    parser.add_argument('--dx_max', type=float, nargs="+", default=None)

    parser.add_argument('--n_trials', type=int, default=100)

    parser.add_argument('--skip_viz', action='store_true', default=False)

    parser.add_argument('--print_sim_freq', type=int, default=3)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--viz_freq', type=int, default=10)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--clf_hiddens', type=int, nargs="+", default=[64, 64])
    parser.add_argument('--actor_hiddens', type=int, nargs="+", default=[64, 64])
    parser.add_argument('--clf_pretrained_path', type=str, default=None)
    parser.add_argument('--actor_pretrained_path', type=str, default=None)

    parser.add_argument('--nn_ratio', type=float, default=0.3)
    parser.add_argument('--clf_nn_weight', type=float, default=0.0)

    parser.add_argument('--alpha_v', type=float, default=0.5)
    parser.add_argument('--v_thres', type=float, default=0.1)

    parser.add_argument('--pos_thres', type=float, default=1.0)
    parser.add_argument('--neg_thres', type=float, default=1.0)
    parser.add_argument('--dq1_thres', type=float, default=-0.01)

    parser.add_argument('--w_zero', type=float, default=1.0)
    parser.add_argument('--w_dec', type=float, default=1.0)
    parser.add_argument('--w_bdry', type=float, default=1.0)
    parser.add_argument('--w_dq1', type=float, default=1.0)

    parser.add_argument('--actor_only', action='store_true', default=False)
    parser.add_argument('--clf_only', action='store_true', default=False)
    parser.add_argument('--regen_init', action='store_true', default=False)
    parser.add_argument('--regen_freq', type=int, default=10)

    parser.add_argument('--traj_sampling', action='store_true', default=False)
    parser.add_argument('--not_plot_phase', action='store_true', default=False)

    parser.add_argument('--c_level', type=float, default=1.0)

    parser.add_argument('--test_r18', action='store_true', default=False)

    parser.add_argument('--viz_nx', type=int, default=25)
    parser.add_argument('--viz_ny', type=int, default=25)

    parser.add_argument('--skip_training', action='store_true', default=False)

    args = parser.parse_args()
    args.load_last = True
    args.fine_switch = True
    args.constant_g = False
    args.changed_dynamics = False
    args.hjb_u_bound = 4
    args.reset_q1_threshold = -0.03

    if args.gait_idx_indices is not None:
        assert len(args.gait_idx_indices) == args.n_theta
    else:
        assert args.gait_idx_end - args.gait_idx_begin == args.n_theta
    args.X_DIM = 4
    args.N_REF = 1
    args.U_DIM = 1

    l = 1.0
    zeros_as = torch.zeros_like
    ones_as = torch.ones_like

    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.4f seconds" % (t2 - t1))