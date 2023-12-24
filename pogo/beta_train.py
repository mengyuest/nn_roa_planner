import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon

import beta_pogo_sim
from beta_fit import Dyna, preproc_data

sys.path.append("../")
import roa_utils as utils
from roa_nn import CLF, Actor, Net

def cal_loss(pred, y):
    return torch.mean(torch.square(pred-y))


def to_np(tensor):
    return tensor.detach().cpu().numpy()


def to_item(tensor):
    return tensor.detach().cpu().item()


def plt_mask_scatter(data_x, data_y, mask, color_list, scale_list, label_list=None):
    for i in range(2):
        idx = torch.where(mask==i)[0]
        if label_list is not None:
            plt.scatter(to_np(data_x[idx]), to_np(data_y[idx]), color=color_list[i], s=scale_list[i], label=label_list[i])
        else:
            plt.scatter(to_np(data_x[idx]), to_np(data_y[idx]), color=color_list[i], s=scale_list[i])

from scipy.spatial import ConvexHull


def comp_x(x_r, net, clf, actor, args):
    v = clf(x_r)
    u = actor(x_r)
    new_x_u = torch.cat([x_r[:, :2], u], dim=-1)
    # next_x = net(new_x_u)[:, 1:3]
    next_x = dynamic(new_x_u, net, args)
    next_x_r = torch.cat([next_x, x_r[:, 2:]], dim=-1)
    next_v = clf(next_x_r)

    cls_mask = torch.logical_or(
        next_v <= (1 - args.alpha_v) * v,
        torch.norm(x_r[:, :2] - x_r[:, 2:], dim=-1, keepdim=True) < args.norm_thres
    )
    mask_ratio = torch.mean(cls_mask.float())

    roa_c, roa_mask = find_level_c(v, cls_mask, args)
    roa_ratio = torch.mean(roa_mask.float())

    return next_x_r, v, cls_mask, roa_mask, mask_ratio, roa_ratio


def find_level_c(v, cls_mask, args):
    cand_c = v[torch.where(cls_mask.float() > 0)]
    if cand_c.shape[0] > 0:
        roa_max_c = torch.max(cand_c)
    else:
        roa_max_c = 0.0

    cand_c = v[torch.where(cls_mask.float() < 1)]
    if cand_c.shape[0] > args.vio_thres:
        if args.vio_thres == 0:
            roa_min_c = torch.min(cand_c)
        else:
            roa_min_c = torch.sort(cand_c)[0][args.vio_thres]
    else:
        roa_min_c = 65535
    roa_c = min(roa_max_c, roa_min_c)
    roa_mask = v < roa_c
    return roa_c, roa_mask


def mock_net(x_u, args):
    x = x_u[:, :2]
    u = x_u[:, 2:]
    if args.normalize:
        x = (x - args.in_means[:2]) / args.in_stds[:2]
        u = (u - args.in_means[2:4]) / args.in_stds[2:4]
    new_x = x + u
    if args.normalize:
        new_x = new_x * args.in_stds[:2] + args.in_means[:2]
    return new_x

def dynamic(x_u, net, args):
    if args.mock_net:
        return mock_net(x_u, args)
    else:
        return net(x_u)[:, 1:3]


def next_x_full(x, actor, net, args):
    x_u = torch.cat([x[:, 1:3], actor(x[:, 1:5])], dim=-1)
    next_x_self = net(x_u)  # (N, 3)
    next_x = torch.cat([
        x[:, 0:1] + next_x_self[:, 0:1],
        next_x_self[:, 1:3],
        x[:, 3:5]
    ], dim=-1)
    return next_x


def main():
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    utils.setup_data_exp_and_logger(args, offset=1)

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    data_path = utils.smart_path(args.data_path) + "/data.npz"
    train_x_u, test_x_u, x_u_mean, x_u_std, train_y, test_y, y_mean, y_std = preproc_data(data_path, split_ratio=args.split_ratio)

    args.in_means = x_u_mean
    args.in_stds = x_u_std
    args.out_means = y_mean
    args.out_stds = y_std

    net = Dyna(args)
    clf = CLF(None, args)
    actor = Actor(None, args)

    utils.safe_load_nn(net, args.dyna_pretrained_path, load_last=args.load_last, key="model_")
    utils.safe_load_nn(actor, args.actor_pretrained_path, load_last=args.load_last, key="actor_", model_iter=args.model_iter, mode="pogo")
    utils.safe_load_nn(clf, args.clf_pretrained_path, load_last=args.load_last, key="clf_", model_iter=args.model_iter, mode="pogo")

    X_MIN=np.min(train_x_u.detach().cpu().numpy()[:,0])
    X_MAX=np.max(train_x_u.detach().cpu().numpy()[:,0])
    Y_MIN=np.min(train_x_u[:,1].detach().cpu().numpy())
    Y_MAX=np.max(train_x_u[:,1].detach().cpu().numpy())
    GRID_X=(X_MAX-X_MIN)/(args.nx-1)
    GRID_Y=(Y_MAX-Y_MIN)/(args.ny-1)

    vis_x_xd_norm = np.linspace(X_MIN, X_MAX, args.nx)
    vis_x_y_norm = np.linspace(Y_MIN, Y_MAX, args.ny)
    vis_x_2d_norm_0, vis_x_2d_norm_1 = np.meshgrid(vis_x_xd_norm, vis_x_y_norm, indexing="xy")
    vis_x_2d = np.stack((vis_x_2d_norm_0.flatten(), vis_x_2d_norm_1.flatten()), axis=-1)  # (N * 2)
    vis_x_2d = torch.from_numpy(vis_x_2d).float()

    vl_x_xd_norm=np.linspace(X_MIN, X_MAX, args.sparse_nx)
    vl_x_y_norm = np.linspace(Y_MIN, Y_MAX, args.sparse_ny)
    vl_x_2d_norm_0, vl_x_2d_norm_1 = np.meshgrid(vl_x_xd_norm, vl_x_y_norm, indexing="xy")
    vl_x_2d = np.stack((vl_x_2d_norm_0.flatten(), vl_x_2d_norm_1.flatten()), axis=-1)  # (N * 2)
    vl_x_2d = torch.from_numpy(vl_x_2d).float()

    # rand_idx_vis = torch.randperm(vis_x_2d.shape[0])[:args.num_vis_pts]

    if args.gpus is not None:
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        train_x_u = train_x_u.cuda()
        test_x_u = test_x_u.cuda()
        net = net.cuda()
        clf = clf.cuda()
        actor = actor.cuda()
        net.update_device()
        clf.update_device()
        actor.update_device()
        vis_x_2d = vis_x_2d.cuda()
        vl_x_2d = vl_x_2d.cuda()

        x_u_mean_gpu = x_u_mean.cuda()
        x_u_std_gpu = x_u_std.cuda()
        y_mean_gpu = y_mean.cuda()
        y_std_gpu = y_std.cuda()

        if args.mock_net:
            args.in_means = args.in_means.cuda()
            args.in_stds = args.in_stds.cuda()
            args.out_means = args.out_means.cuda()
            args.out_stds = args.out_stds.cuda()

    np.savez("%s/data_mean_std.npz" % args.model_dir, x_u_mean=x_u_mean, x_u_std=x_u_std, y_mean=y_mean, y_std=y_std)

    if args.test_for_real:
        tt1=time.time()
        num_steps = args.nt
        dt = args.dt1
        dt2 = args.dt2
        l0 = args.l0
        g = args.g
        k = args.k
        M = args.M

        global_vars = num_steps, dt, dt2, l0, g, k, M

        # test_x_u, x_u_mean, x_u_std, y_mean, y_std
        initial_state = test_x_u[:args.test_batch_size]

        nn_x = initial_state[:,:2] * 1.0  # (n, 4) normalized xd, y, th, k
        gt_x = initial_state * 1.0  # (n, 4) normalized xd, y, th, k
        hy_x = initial_state * 1.0

        nn_ori_x_list=[]
        gt_ori_x_list=[]
        num_loops=args.test_num_loops
        markersize = args.test_markersize
        num_vis = args.test_num_vis

        ani_x_list=[]
        visible = [True for _ in range(args.test_num_vis)]
        for loop_i in range(num_loops):
            ani_x_list.append([])
            # est
            nn_score=clf(nn_x[:, :2])
            nn_u = actor(nn_x[:, :2])  # (n, 2) normalized xd, y => (n, 2) th, k
            nn_new_x_u = torch.cat([nn_x[:, :2], nn_u], dim=-1)  # (n, 4) normalized xd, y, th, k
            #nn_next_x_sub = net(nn_new_x_u)[:, 1:3]  # (n, 3) normalized x, xd, y
            nn_next_x_sub = dynamic(nn_new_x_u, net, args)

            # print(nn_x.shape, x_u_std.shape, x_u_mean.shape)
            nn_ori_x = nn_x * x_u_std_gpu[0:2] + x_u_mean_gpu[0:2]  # (n, 2) original xd, y
            nn_ori_next_x = nn_next_x_sub * x_u_std_gpu[0:2] + x_u_mean_gpu[0:2]  # (n, 3) original x, xd, y
            nn_ori_u = nn_u * x_u_std_gpu[2:] + x_u_mean_gpu[2:]
            nn_ori_x_list.append(nn_ori_x)

            # real
            gt_score = clf(gt_x[:, :2])
            gt_modes = torch.zeros_like(initial_state)  # (n, 4) 0, 0, 0, 0
            gt_u = actor(gt_x[:, :2])  # (n, 2) normalized xd, y => (n, 2) th, k
            gt_ori_x = torch.zeros(initial_state.shape[0], 6).to(initial_state.device)  # (n, 4) 0, 0, 0, 0
            gt_ori_x[:, 1:3] = gt_x[:, :2] * x_u_std_gpu[:2] + x_u_mean_gpu[:2]
            gt_ori_x[:, 5] = gt_ori_x[:, 2] - 1
            gt_ori_x_list.append(gt_ori_x)
            gt_sim_x = gt_ori_x * 1.0
            gt_ori_u = gt_u * x_u_std_gpu[2:] + x_u_mean_gpu[2:]
            gt_x_list=[]
            gt_m_list=[]
            for ti in range(args.nt):
                gt_next_x, gt_next_modes = beta_pogo_sim.pytorch_sim(gt_sim_x, gt_modes, gt_ori_u, global_vars, check_invalid=True)
                gt_x_list.append(gt_sim_x.detach().cpu().numpy())
                gt_m_list.append(gt_modes.detach().cpu().numpy())
                gt_sim_x, gt_modes = gt_next_x, gt_next_modes
            gt_ori_next_x = np.zeros((args.test_batch_size, 4))
            # backtrace to find valid xs
            # and find first return apex state
            gt_x_list = np.stack(gt_x_list, axis=0)
            gt_m_list = np.stack(gt_m_list, axis=0)
            for ni in range(args.test_batch_size):
                m6 = np.nonzero(gt_m_list[:, ni] == 7)
                if m6[0].shape[0] > 0:
                    idx = m6[0][0]
                    gt_ori_next_x[ni, :] = gt_x_list[idx, ni, :4]
                    ani_x_list[-1].append(gt_x_list[:idx, ni, :6])
                else:
                    ani_x_list[-1].append(None)
                    visible[ni] = False
                # else:
                #     modes[ni] = -1.0

            # TODO comparison
            print("%d nns:%.4f gts:%.4f nn_x:%.4f,%.4f gt_x:%.4f,%.4f nn_u:%.4f,%.4f gt_u:%.4f,%.4f| nn_next_x:%.4f,%.4f  gt_next_x:%.4f,%.4f"%(
                loop_i, nn_score[0,0], gt_score[0,0], nn_ori_x[0, 0], nn_ori_x[0, 1], gt_ori_x[0, 1], gt_ori_x[0, 2],
                nn_ori_u[0, 0], nn_ori_u[0, 1], gt_ori_u[0, 0], gt_ori_u[0, 1],
                nn_ori_next_x[0, 0], nn_ori_next_x[0, 1],
                gt_ori_next_x[0, 1], gt_ori_next_x[0, 2]
            ))

            # est
            nn_x = nn_next_x_sub

            # real
            gt_x = (gt_ori_next_x[:, 1:3] - x_u_mean[0:2].detach().numpy()) / x_u_std[0:2].detach().numpy()
            gt_x = torch.from_numpy(gt_x).float().to(initial_state.device)
        tt2=time.time()

        sync_x_list=[[] for _ in range(num_vis)]
        init_x = [0 for _ in range(num_vis)]
        for n_i in range(num_vis):
            for loop_i in range(num_loops):
                for tti in range(len(ani_x_list[loop_i][n_i])):
                    s=ani_x_list[loop_i][n_i][tti]
                    new_s=[s[0]+init_x[n_i], s[1], s[2], s[3], s[4]+init_x[n_i], s[5]]
                    if new_s[5]>0:
                        NUM_REPEATS=10
                    else:
                        NUM_REPEATS=1
                    for xxx in range(NUM_REPEATS):
                        sync_x_list[n_i].append(new_s)
                init_x[n_i] = init_x[n_i] + s[0]

        min_time_len = np.min([len(xx) for xx in sync_x_list])
        viz_freq = 200
        for tt_i in range(min_time_len):
            if tt_i % viz_freq == 0:
                plt.figure(figsize=(8, 3))
                plt.rcParams.update({'font.size': 16})
                for n_i in range(num_vis):
                    s = sync_x_list[n_i][tt_i]
                    plt.plot([s[0], s[4]], [s[2], s[5]], linewidth=2)
                    plt.plot(s[0], s[2], 'ro', markersize=4)
                    plt.plot(s[4], s[5], 'ro', markersize=4)
                plt.xlim(-1, 15.0)
                plt.ylim(-0.5, 2.5)
                plt.xlabel("x (m)")
                plt.ylabel("y (m)")
                plt.tight_layout()
                plt.savefig("%s/t_%05d.png" % (args.viz_dir, tt_i), pad_inches=0.5)
                plt.close()

        # first plot first one from ani_x_list
        viz_freq=20
        flight_viz_freq = 5
        stance_viz_freq = 50
        init_x = [0 for _ in range(num_vis)]

        for loop_i in range(num_loops):
            # TODO plot heatmap with coordinates
            plt.subplot(1, 2, 1)
            vis_x_val = clf(vis_x_2d)
            im = plt.imshow(vis_x_val.detach().cpu().numpy().reshape((args.ny, args.nx)), origin='lower', cmap=cm.inferno)
            plt.colorbar(im,fraction=0.046, pad=0.04, format="%.3f")
            ax = plt.gca()
            n_xticks = np.linspace(X_MIN * x_u_std[0] + x_u_mean[0], X_MAX * x_u_std[0] + x_u_mean[0], 4)
            n_yticks = np.linspace(Y_MIN * x_u_std[1] + x_u_mean[1], Y_MAX * x_u_std[1] + x_u_mean[1], 7)

            n_x_start = n_xticks[0]
            n_x_delta = (n_xticks[1] - n_xticks[0])*(n_xticks.shape[0]-1)/(args.nx-1)
            n_y_start = n_yticks[0]
            n_y_delta = (n_yticks[1] - n_yticks[0])*(n_yticks.shape[0]-1)/(args.ny-1)

            ax.set_xticks(np.linspace(0, args.nx, n_xticks.shape[0]))
            ax.set_xticklabels(["%.3f" % xx for xx in n_xticks])
            ax.set_yticks(np.linspace(0, args.ny, n_yticks.shape[0]))
            ax.set_yticklabels(["%.3f" % xx for xx in n_yticks])
            plt.xlabel("x velocity")
            plt.ylabel("y height")

            # TODO plot nn line
            for line_i in range(num_vis):
                plt.plot([(nn_ori_x_list[step_i][line_i, 0].detach().cpu().numpy()-n_x_start)/n_x_delta for step_i in range(num_loops)],
                        [(nn_ori_x_list[step_i][line_i, 1].detach().cpu().numpy()-n_y_start)/n_y_delta for step_i in range(num_loops)],
                        color="blue")
            # TODO plot nn scatter varied time
            plt.scatter((nn_ori_x_list[loop_i][:num_vis, 0].detach().cpu().numpy()-n_x_start)/n_x_delta,
                        (nn_ori_x_list[loop_i][:num_vis, 1].detach().cpu().numpy()-n_y_start)/n_y_delta,
                        color="red", zorder=1000005, s=markersize * 10)
            plt.gca().set_title('estimated dynamics')

            # TODO plot heatmap with coordinates
            plt.subplot(1, 2, 2)
            vis_x_val = clf(vis_x_2d)
            im = plt.imshow(vis_x_val.detach().cpu().numpy().reshape((args.ny, args.nx)), origin='lower', cmap=cm.inferno)
            plt.colorbar(im,fraction=0.046, pad=0.04, format="%.3f")
            ax = plt.gca()
            n_xticks = np.linspace(X_MIN * x_u_std[0] + x_u_mean[0], X_MAX * x_u_std[0] + x_u_mean[0], 4)
            n_yticks = np.linspace(Y_MIN * x_u_std[1] + x_u_mean[1], Y_MAX * x_u_std[1] + x_u_mean[1], 7)

            ax.set_xticks(np.linspace(0, args.nx, n_xticks.shape[0]))
            ax.set_xticklabels(["%.3f" % xx for xx in n_xticks])
            ax.set_yticks(np.linspace(0, args.ny, n_yticks.shape[0]))
            ax.set_yticklabels(["%.3f" % yy for yy in n_yticks])
            plt.xlabel("x velocity")
            plt.ylabel("y height")

            # TODO plot gt line
            for line_i in range(num_vis):
                plt.plot([(gt_ori_x_list[step_i][line_i, 1].detach().cpu().numpy()-n_x_start)/n_x_delta for step_i in range(num_loops)],
                        [(gt_ori_x_list[step_i][line_i, 2].detach().cpu().numpy()-n_y_start)/n_y_delta for step_i in range(num_loops)],
                        color="blue")
            # TODO plot gt scatter varied time
            plt.scatter((gt_ori_x_list[loop_i][:num_vis, 1].detach().cpu().numpy()-n_x_start)/n_x_delta,
                        (gt_ori_x_list[loop_i][:num_vis, 2].detach().cpu().numpy()-n_y_start)/n_y_delta,
                        color="red", zorder=1000005, s=markersize * 10)
            plt.gca().set_title('real dynamics')
            plt.tight_layout()
            plt.savefig("%s/fig_nn_gt_%02d.png" % (args.viz_dir, loop_i), bbox_inches='tight', pad_inches=0.1)
            plt.close()

        print("finished in %.4f seconds"%(tt2-tt1))
        exit()

    if args.estimate_roa:
        estimate_roa(net, clf, actor)
    else:
        train_clf_actor(train_x_u, test_x_u, net, clf, actor)

def get_bdry_points(points, v, c, dim):
    bdry = points[torch.where((v - c) * (v) < 0)[0]]
    bdry = bdry[:, [dim[0], dim[1]]]
    bdry = to_np(bdry)
    hull = ConvexHull(points=bdry)
    return bdry[hull.vertices]



def estimate_roa(net, clf, actor):
    import tqdm
    # RoA is defined by multi-steps, decreasing V or fall into range
    # collect fitting data
    tlen = args.net_tlen  # 3
    num_tests = args.net_num_tests  # 1000, 400
    num_samples = args.net_num_configs  # 10000, 10
    batch_size = 200

    data_list = []
    dbg_t0 = time.time()
    q_err = utils.uniform_sample_tsr(num_tests, args.err_mins, args.err_maxs)
    q_err = (1 + args.roa_bloat) * q_err
    q_err = q_err.unsqueeze(0).repeat(batch_size, 1, 1)  # (bs, num_tests)
    if args.multi_initials is False:
        for ni in tqdm.tqdm(range(num_samples // batch_size)):
            q_ref = utils.uniform_sample_tsr(batch_size, args.ref_mins, args.ref_maxs)
            q_ref = q_ref.unsqueeze(1).repeat(1, num_tests, 1)
            q_abs = q_ref + q_err
            q_abs = q_abs.reshape(batch_size * num_tests, -1)
            q_ref = q_ref.reshape(batch_size * num_tests, -1)
            q_x_r = torch.cat([q_abs, q_ref], dim=-1)
            if args.gpus is not None:
                q_x_r = q_x_r.cuda()

            # get a seq of states and v
            x_r_list = [q_x_r]
            v_list = []
            u_list = []

            for i in range(tlen):
                the_x_r = x_r_list[-1]
                the_v = clf(the_x_r).detach()
                the_u = actor(the_x_r).detach()
                new_x_u = torch.cat([the_x_r[:, :2], the_u], dim=-1)
                next_x = dynamic(new_x_u, net, args).detach()
                next_x_r = torch.cat([next_x, q_x_r[:, 2:]], dim=-1)

                x_r_list.append(next_x_r)
                v_list.append(the_v)
                u_list.append(the_u)

            x_r_list = torch.stack(x_r_list[:-1], dim=1)
            v_list = torch.stack(v_list, dim=1)

            for bi in range(batch_size):
                x_r_b = x_r_list[bi*num_tests:(bi+1)*num_tests]
                v_b = v_list[bi*num_tests:(bi+1)*num_tests]

                # train roa_estimator
                mask_b = torch.logical_or(torch.all(v_b[:, 1:] <= (1 - args.alpha_v) * v_b[:, :-1], dim=1),
                    torch.norm(x_r_b[:, -1, :2] - x_r_b[:, -1, 2:], keepdim=True, dim=-1) < args.norm_thres)

                # TODO compute the level set c value
                roa_c_b, roa_mask_b = find_level_c(v_b[:, 0], mask_b, args)
                # TODO visualization
                collected_sample = torch.cat([x_r_b[0, 0, 2:], torch.tensor([roa_c_b]).type_as(x_r_b),
                                              torch.mean(mask_b.float()).type_as(x_r_b).unsqueeze(0),
                                              torch.mean(roa_mask_b.float()).type_as(x_r_b).unsqueeze(0)], dim=0)
                data_list.append(collected_sample)

        data_list = torch.stack(data_list, dim=0)
        plot_heatmap(data_list, data_list[:, 2], "%s/heat0_cval.png" % (args.viz_dir))
        plot_heatmap(data_list, data_list[:, 3], "%s/heat1_mask.png" % (args.viz_dir))
        plot_heatmap(data_list, data_list[:, 4], "%s/heat2_roa_mask.png" % (args.viz_dir))
        dbg_t1 = time.time()
        print("Finished the data preparation(%s) in %.4f seconds" % (data_list.shape, dbg_t1 - dbg_t0))

    # TODO visualize the heatmaps
    if args.multi_initials:
        ctr_nx = 100
        ctr_ny = 100
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
        NUM_TRIALS = 100
        q_v_refs = np.linspace(1.0, 4.0, 10)
        q_h_refs = np.linspace(1.5, 4.0, 10)
        for trial_i in range(NUM_TRIALS):
            # (x, xdot, y, ydot)
            lin0 = torch.linspace(args.err_mins[0], args.err_maxs[0], ctr_nx)
            lin1 = torch.linspace(args.err_mins[1], args.err_maxs[1], ctr_ny)
            lin0 = lin0 * (1 + args.roa_bloat)
            lin1 = lin1 * (1 + args.roa_bloat)

            q_ref = utils.uniform_sample_tsr(1, args.ref_mins, args.ref_maxs)
            q_ref[0, 0] = q_v_refs[trial_i // 10]
            q_ref[0, 1] = q_h_refs[trial_i % 10]
            q_rel = torch.stack(torch.meshgrid(lin0, lin1), dim=-1).reshape((ctr_nx * ctr_ny, 2))
            q_abs = q_rel + q_ref

            q_ref = q_ref.repeat([q_abs.shape[0], 1])
            print(q_ref, q_abs.shape, q_ref.shape)
            q_x_r = torch.cat([q_abs, q_ref], dim=-1)

            if args.gpus is not None:
                q_x_r = q_x_r.cuda()

            x_r_list = [q_x_r]
            v_list = []
            u_list = []

            for i in range(tlen):
                the_x_r = x_r_list[-1]
                the_v = clf(the_x_r).detach()
                the_u = actor(the_x_r).detach()
                new_x_u = torch.cat([the_x_r[:, :2], the_u], dim=-1)
                next_x = dynamic(new_x_u, net, args).detach()
                next_x_r = torch.cat([next_x, q_x_r[:, 2:]], dim=-1)

                x_r_list.append(next_x_r)
                v_list.append(the_v)
                u_list.append(the_u)

            x_r_list = torch.stack(x_r_list[:-1], dim=1)
            v_list = torch.stack(v_list, dim=1)

            mask_v = torch.logical_or(torch.all(v_list[:, 1:] <= (1 - args.alpha_v) * v_list[:, :-1], dim=1),
                                      torch.norm(x_r_list[:, -1, :2] - x_r_list[:, -1, 2:], keepdim=True,
                                                 dim=-1) < args.norm_thres)
            roa_c_v, roa_mask_v = find_level_c(v_list[:, 0], mask_v, args)

            x_init = x_r_list[:, 0, :]
            v_init = v_list[:, 0, :]

            plt.contourf(to_np(x_init[:, 0]).reshape((ctr_nx, ctr_ny)),
                         to_np(x_init[:, 1]).reshape((ctr_nx, ctr_ny)),
                         to_np(v_init[:, 0]).reshape((ctr_nx, ctr_ny)),
                         levels=30, cmap=trun_cmap, alpha=0.7,  # locator=ticker.LogLocator(base=3, numticks=20),
                         )

            nn_bdry = get_bdry_points(x_init, v_init, roa_c_v, [0, 1])
            ax = plt.gca()
            ax.add_patch(Polygon(nn_bdry, antialiased=True, fill=False, facecolor="blue",
                                 edgecolor="blue", linestyle="--", linewidth=2.5, label="Ours", alpha=1.0))

            plt.axis("scaled")
            plt.legend(fontsize=int(SMALL_SIZE * 0.75))
            plt.xlabel(r"$\dot{x}$ (m/s)", fontsize=SMALL_SIZE)
            plt.ylabel(r"$y$ (m)", fontsize=SMALL_SIZE)
            plt.xticks(fontsize=SMALL_SIZE)
            plt.yticks(fontsize=SMALL_SIZE)
            xmins=np.min(to_np(x_init), axis=0)
            xmaxs=np.max(to_np(x_init), axis=0)
            plt.xlim(xmins[0], xmaxs[0])
            plt.ylim(xmins[1], xmaxs[1])
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=int(SMALL_SIZE * 0.75))
            plt.savefig("%s/pogo_roa_dim_%02d_%.3f_%.3f.png" % (args.viz_dir, trial_i, q_ref[0,0], q_ref[0,1]), bbox_inches='tight', pad_inches=0.1)
            plt.close()


    # TODO visualize the heatmaps
    if args.multi_initials:
        dbg_t1 = time.time()
        print("Finished the data preparation in %.4f seconds" % (dbg_t1 - dbg_t0))
        return

    # ROA ESTIMATOR TRAINING

    net = Net(args)
    if args.gpus is not None:
        net = net.cuda()
    args.net_lr = args.net_setups[0]
    args.net_epochs = int(args.net_setups[1])
    args.net_print_freq = int(args.net_setups[2])
    args.net_viz_freq = int(args.net_setups[3])
    args.net_save_freq = int(args.net_setups[4])

    levelset_data = data_list
    split = 0.8
    levelset_input = levelset_data[:, 0:2]
    levelset_gt = levelset_data[:, 2:3]
    train_input, test_input = utils.split_tensor(levelset_input, int(split * num_samples))
    train_gt, test_gt = utils.split_tensor(levelset_gt, int(split * num_samples))
    criterion1 = lambda x, y: torch.mean((x - y) ** 2 / torch.clamp(y, min=1e-4))
    criterion2 = nn.MSELoss()

    train_losses, test_losses, train_losses1, test_losses1, train_losses2, test_losses2 = utils.get_n_meters(6)
    optimizer = torch.optim.RMSprop(net.parameters(), args.net_lr)
    train_len = train_input.shape[0] // 10

    for epi in range(args.net_epochs):
        if epi % 500 == 0:
            train_indices = torch.randperm(train_input.shape[0])[:train_len]
        train_est = net(train_input[train_indices])
        test_est = net(test_input)

        train_loss1 = criterion1(train_est, train_gt[train_indices])
        test_loss1 = criterion1(test_est, test_gt)
        train_loss2 = criterion2(train_est, train_gt[train_indices])
        test_loss2 = criterion2(test_est, test_gt)
        if args.net_rel_loss:
            train_loss = train_loss1
            test_loss = test_loss1
        else:
            train_loss = train_loss2
            test_loss = test_loss2

        utils.update(train_losses, train_loss)
        utils.update(test_losses, test_loss)
        utils.update(train_losses1, train_loss1)
        utils.update(test_losses1, test_loss1)
        utils.update(train_losses2, train_loss2)
        utils.update(test_losses2, test_loss2)

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epi % args.net_print_freq == 0:
            print("%05d/%05d train: %.6f (%.6f) %.6f %.6f  test: %.6f (%.6f) %.6f %.6f" % (
                epi, args.net_epochs, train_losses.val, train_losses.avg, train_losses1.val, train_losses2.val,
                test_losses.val, test_losses.avg, test_losses1.val, test_losses2.val))

        if epi % args.net_viz_freq == 0:
            val_est = net(data_list[:, :2])
            plot_heatmap(data_list, val_est, "%s/est_%05d_cval.png" % (args.viz_dir, epi))

        if epi == 0 or epi == args.net_epochs -1 or epi % args.net_save_freq == 0:
            torch.save(net.state_dict(), "%s/net_e%05d.ckpt" % (args.model_dir, epi))


def plot_heatmap(data_list, val_list, img_name):
    cmap = cm.get_cmap('inferno')
    plt.scatter(to_np(data_list[:, 0]), to_np(data_list[:, 1]), c=to_np(val_list), s=10, cmap=cmap)
    plt.xlabel("xdot (m/s)")
    plt.ylabel("y (m)")
    plt.colorbar()
    plt.savefig(img_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()



def plot_roa_points(x_r_list, mask, roa_mask, ni):
    xmin = torch.min(x_r_list[:, 0, 0]).detach().cpu().item()
    xmax = torch.max(x_r_list[:, 0, 0]).detach().cpu().item()
    ymin = torch.min(x_r_list[:, 0, 1]).detach().cpu().item()
    ymax = torch.max(x_r_list[:, 0, 1]).detach().cpu().item()
    for ti in range(x_r_list.shape[1]):
        plt_mask_scatter(x_r_list[:, ti, 0], x_r_list[:, ti, 1], mask.float(), color_list=["red", "blue"],
                         scale_list=[0.2, 0.2])
        nn_c_in_points = x_r_list[:, ti, :2][torch.where(roa_mask)[0]]
        ax = plt.gca()
        if nn_c_in_points.shape[0] > 2:
            nn_hull = ConvexHull(points=to_np(nn_c_in_points))
            nn_patch = Polygon(to_np(nn_c_in_points)[nn_hull.vertices], color="cyan", label="RoA", alpha=0.5)
            ax.add_patch(nn_patch)

        plt.legend()

        plt.xlabel("x_dot(m/s)")
        plt.ylabel("y (m)")
        plt.axis("scaled")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig("%s/roa_epi%04d_t%04d.png" % (args.viz_dir, ni, ti), bbox_inches='tight', pad_inches=0.1)
        plt.close()


def train_clf_actor(train_x_u, test_x_u, net, clf, actor):
    train_losses, train_cls_losses, train_grad_losses, test_losses, test_cls_losses, test_grad_losses \
        = utils.get_n_meters(6)

    if args.train_actor_only:
        optimizer = torch.optim.SGD(list(actor.parameters()), args.lr)
    else:
        if args.train_intervals is not None:
            clf_lr = args.clf_lr if args.clf_lr is not None else args.lr
            actor_lr = args.actor_lr if args.actor_lr is not None else args.lr
            optim_clf = torch.optim.SGD(list(clf.parameters()), clf_lr)
            optim_actor = torch.optim.SGD(list(actor.parameters()), actor_lr)
        else:
            optimizer = torch.optim.SGD(list(clf.parameters())+list(actor.parameters()), args.lr)

    num_tests = test_x_u.shape[0]
    q_err = utils.uniform_sample_tsr(num_tests, args.err_mins, args.err_maxs)
    q_err = (1 + args.roa_bloat) * q_err
    q_ref = utils.uniform_sample_tsr(num_tests, args.ref_mins, args.ref_maxs)
    q_ref[:, 0] = q_ref[0, 0]
    q_ref[:, 1] = q_ref[0, 1]
    q_abs = q_err + q_ref
    q_x_r = torch.cat([q_abs, q_ref], dim=-1)

    if args.gpus is not None:
        q_err = q_err.cuda()
        q_abs = q_abs.cuda()
        q_ref = q_ref.cuda()
        q_x_r = q_x_r.cuda()

    for epi in range(args.epochs):
        # TODO sample for ref
        if epi % args.print_freq == 0:
            num_samples = train_x_u.shape[0] + test_x_u.shape[0]
            rep = 100
            s_ratio = min(max((epi // rep) / (args.epochs // rep), 0.01), 1)
            x_err = utils.uniform_sample_tsr(num_samples, args.err_mins, args.err_maxs)
            if args.train_ratio:
                x_err = x_err * s_ratio
            x_ref = utils.uniform_sample_tsr(num_samples, args.ref_mins, args.ref_maxs)
            x_abs = x_err + x_ref
            x_abs[:, 0] = torch.clamp(x_abs[:, 0], min=args.abs_mins[0], max=args.abs_maxs[0])
            x_abs[:, 1] = torch.clamp(x_abs[:, 1], min=args.abs_mins[1], max=args.abs_maxs[1])
            n_train = train_x_u.shape[0]
            train_x_err, test_x_err = utils.split_tensor(x_err, n_train)
            train_x_ref, test_x_ref = utils.split_tensor(x_ref, n_train)
            train_x_abs, test_x_abs = utils.split_tensor(x_abs, n_train)
            if args.gpus is not None:
                train_x_ref = train_x_ref.cuda()
                test_x_ref = test_x_ref.cuda()
                train_x_abs = train_x_abs.cuda()
                test_x_abs = test_x_abs.cuda()

            # train_x_r = torch.cat([train_x_u[:, :2], train_x_ref], dim=-1)
            # test_x_r = torch.cat([test_x_u[:, :2], test_x_ref], dim=-1)
            train_x_r = torch.cat([train_x_abs, train_x_ref], dim=-1)
            test_x_r = torch.cat([test_x_abs, test_x_ref], dim=-1)

        if args.batch_size is not None:
            if epi % args.print_freq == 0:
                rand_idx = torch.randperm(train_x_u.shape[0])
                sub_train_x_r = train_x_r[rand_idx[:args.batch_size]]
        else:
            sub_train_x_r = train_x_r

        # TODO TRAIN
        sub_train_v = clf(sub_train_x_r)
        sub_train_u = actor(sub_train_x_r)
        sub_train_new_x_u = torch.cat([sub_train_x_r[:, :2], sub_train_u], dim=-1)
        # sub_train_next_x = net(sub_train_new_x_u)[:, 1:3]
        sub_train_next_x = dynamic(sub_train_new_x_u, net, args)
        sub_train_next_x_r = torch.cat([sub_train_next_x, sub_train_x_r[:, 2:]], dim=-1)
        sub_train_next_v = clf(sub_train_next_x_r)

        train_cls_mask = torch.logical_or(
            sub_train_next_v <= (1 - args.alpha_v) * sub_train_v,
            torch.norm(sub_train_x_r[:, :2] - sub_train_x_r[:, 2:], dim=-1, keepdim=True) < args.norm_thres
        )

        train_cls_loss = utils.mask_mean(torch.relu(sub_train_v - 1.0), train_cls_mask.float()) + \
                         utils.mask_mean(torch.relu(1 - sub_train_v), 1 - train_cls_mask.float())
        if args.train_alpha_v is None:
            train_grad_loss = torch.mean(
                torch.relu(sub_train_next_v - (1 - args.alpha_v) * sub_train_v + args.v_margin))
        else:
            train_grad_loss = torch.mean(
                torch.relu(sub_train_next_v - (1 - args.train_alpha_v) * sub_train_v + args.v_margin))
        # train_grad_loss = torch.mean(torch.norm(sub_train_next_x - sub_train_x_r[:, 2:], dim=-1))

        train_cls_loss = train_cls_loss * args.cls_weight
        train_grad_loss = train_grad_loss * args.grad_weight
        train_loss = train_cls_loss + train_grad_loss

        # TODO TEST
        test_v = clf(test_x_r)
        test_u = actor(test_x_r)
        test_new_x_u = torch.cat([test_x_r[:, :2], test_u], dim=-1)
        # test_next_x = net(test_new_x_u)[:, 1:3]
        test_next_x = dynamic(test_new_x_u, net, args)
        test_next_x_r = torch.cat([test_next_x, test_x_r[:, 2:]], dim=-1)
        test_next_v = clf(test_next_x_r)
        test_cls_mask = torch.logical_or(
            test_next_v < (1 - args.alpha_v) * test_v - args.v_margin,
            torch.norm(test_x_r[:, :2] - test_x_r[:, 2:], dim=-1, keepdim=True) < args.norm_thres
        )
        test_cls_loss = utils.mask_mean(torch.relu(test_v - 1.0), test_cls_mask.float()) + \
                        utils.mask_mean(torch.relu(1 - test_v), 1 - test_cls_mask.float())
        if args.train_alpha_v is None:
            test_grad_loss = torch.mean(torch.relu(test_next_v - (1 - args.alpha_v) * test_v + args.v_margin))
        else:
            test_grad_loss = torch.mean(torch.relu(test_next_v - (1 - args.train_alpha_v) * test_v + args.v_margin))
        # test_grad_loss = torch.mean(torch.norm(test_next_x - test_x_r[:, 2:], dim=-1))
        test_cls_loss = test_cls_loss * args.cls_weight
        test_grad_loss = test_grad_loss * args.grad_weight
        test_loss = test_cls_loss + test_grad_loss

        train_losses.update(train_loss.detach().cpu().item())
        train_cls_losses.update(train_cls_loss.detach().cpu().item())
        train_grad_losses.update(train_grad_loss.detach().cpu().item())

        test_losses.update(test_loss.detach().cpu().item())
        test_cls_losses.update(test_cls_loss.detach().cpu().item())
        test_grad_losses.update(test_grad_loss.detach().cpu().item())

        mask_ratio = torch.mean(train_cls_mask.float())

        roa_c, roa_mask = find_level_c(sub_train_v, train_cls_mask, args)
        roa_ratio = torch.mean(roa_mask.float())

        if args.no_viz == False and epi % (10 * args.print_freq) == 0:
            # TODO normal case
            next_q_x_r, cls_v, cls_mask, roa_mask, mask_ratio, roa_ratio = comp_x(q_x_r, net, clf, actor, args)
            q_rel = q_x_r[:, :2] - q_x_r[:, 2:]
            next_q_rel = next_q_x_r[:, :2] - next_q_x_r[:, 2:]

            plt_mask_scatter(q_rel[:, 0], q_rel[:, 1], cls_mask.float(), color_list=["red", "blue"],
                             scale_list=[0.2, 0.2])
            nn_c_in_points = q_rel[torch.where(roa_mask)[0]]
            ax = plt.gca()
            if nn_c_in_points.shape[0] > 2:
                nn_hull = ConvexHull(points=to_np(nn_c_in_points))
                nn_patch = Polygon(to_np(nn_c_in_points)[nn_hull.vertices], color="cyan", label="RoA", alpha=0.5)
                ax.add_patch(nn_patch)

            plt.legend()
            plt.xlabel("x_dot(m/s)")
            plt.ylabel("y (m)")
            plt.axis("scaled")
            plt.savefig("%s/debug_%04d.png" % (args.viz_dir, epi),
                        bbox_inches='tight', pad_inches=0.1)
            plt.close()

            cmap = cm.get_cmap('inferno')
            plt.scatter(to_np(q_rel[:, 0]), to_np(q_rel[:, 1]), c=to_np(cls_v)[:, 0], s=0.2, cmap=cmap)
            plt.colorbar()
            plt.xlabel("x_dot(m/s)")
            plt.ylabel("y (m)")
            plt.axis("scaled")
            plt.savefig("%s/heat_%04d.png" % (args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
            plt.close()

            q_rel_np = to_np(q_rel)
            next_q_rel_np = to_np(next_q_rel)
            for i in range(min(next_q_rel.shape[0], 100)):
                arr0 = q_rel_np[i]
                arr1 = next_q_rel_np[i]
                # TODO width = 0.02
                plt.arrow(arr0[0], arr0[1], arr1[0] - arr0[0], arr1[1] - arr0[1], width=0.005, color="blue")
            plt.xlabel("x_dot(m/s)")
            plt.ylabel("y (m)")
            plt.axis("scaled")
            plt.savefig("%s/arr_%04d.png" % (args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
            plt.close()

        train_loss.backward()
        if args.train_intervals is not None:
            interval = sum(args.train_intervals)
            if epi % interval < args.train_intervals[0]:
                the_optim = optim_actor
            else:
                the_optim = optim_clf
        else:
            the_optim = optimizer

        the_optim.step()
        the_optim.zero_grad()

        if epi % args.print_freq == 0:
            print("[%05d/%05d] %.2f%% %.2f%% train:%.4f(%.4f) cls:%.4f(%.4f) grad:%.4f(%.4f) "
                  "test:%.4f(%.4f) cls:%.4f(%.4f) grad:%.4f(%.4f)" %
                  (epi, args.epochs,
                   mask_ratio * 100, roa_ratio * 100,
                   train_losses.val, train_losses.avg, train_cls_losses.val, train_cls_losses.avg,
                   train_grad_losses.val, train_grad_losses.avg,
                   test_losses.val, test_losses.avg, test_cls_losses.val, test_cls_losses.avg,
                   test_grad_losses.val, test_grad_losses.avg
                   ))

        if epi % args.save_freq == 0:
            torch.save(actor.state_dict(), "%s/actor_e%05d.ckpt" % (args.model_dir, epi))
            torch.save(clf.state_dict(), "%s/clf_e%05d.ckpt" % (args.model_dir, epi))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="beta_fit_dbg")
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--hiddens', type=int, nargs="+", default=[16, 16])

    parser.add_argument('--dyna_pretrained_path', type=str, default=None)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--data_paths', type=str, default=None)
    parser.add_argument('--ranges', type=int, nargs="+", default=None)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--viz_freq', type=int, default=500)
    parser.add_argument('--save_freq', type=int, default=2000)

    parser.add_argument('--batch_size', type=int, default=None)

    parser.add_argument('--zero_weight', type=float, default=1.0)
    parser.add_argument('--pos_weight', type=float, default=1.0)
    parser.add_argument('--grad_weight', type=float, default=1.0)

    parser.add_argument('--pos_thres', type=float, default=0.1)
    parser.add_argument('--grad_thres', type=float, default=0.1)

    parser.add_argument('--nx', type=int, default=51)
    parser.add_argument('--ny', type=int, default=51)
    parser.add_argument('--sparse_nx', type=int, default=7)
    parser.add_argument('--sparse_ny', type=int, default=7)
    parser.add_argument('--num_vis_pts', type=int, default=25)
    parser.add_argument('--actor_gain', type=float, default=1.0)
    parser.add_argument('--prior_gain', type=float, default=0.1)
    parser.add_argument('--nn_clf_gain', type=float, default=0.2)

    parser.add_argument('--test_for_real', action='store_true', default=False)
    parser.add_argument('--actor_pretrained_path', type=str, default=None)
    parser.add_argument('--clf_pretrained_path', type=str, default=None)
    parser.add_argument('--nt', type=int, default=3000)
    parser.add_argument('--dt1', type=float, default=0.001)
    parser.add_argument('--dt2', type=float, default=0.0001)
    parser.add_argument('--l0', type=float, default=1.0)
    parser.add_argument('--g', type=float, default=10.0)
    parser.add_argument('--k', type=float, default=32000)
    parser.add_argument('--M', type=float, default=80)
    parser.add_argument('--ratio_grad_loss', action='store_true', default=False)
    parser.add_argument('--ratio_thres', type=float, default=0.01)


    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--test_num_loops', type=int, default=10)
    parser.add_argument('--test_markersize', type=float, default=1.0)
    parser.add_argument('--test_num_vis', type=int, default=10)

    # TODO new nn
    parser.add_argument('--multi', action='store_true', default=False)
    parser.add_argument('--clf_mode', type=str, default="ellipsoid", choices=["ellipsoid", "nn", "diag"])
    parser.add_argument('--clf_hiddens', type=int, nargs="+", default=[16, 16])
    parser.add_argument('--clf_ell_scale', action='store_true', default=False)
    parser.add_argument('--clf_ell_extra', action='store_true', default=False)
    parser.add_argument('--clf_e2e', action='store_true', default=False)

    parser.add_argument('--actor_mode', type=str, default="nn", choices=["nn"])
    parser.add_argument('--actor_hiddens', type=int, nargs="+", default=[16, 16])
    parser.add_argument('--actor_e2e', action='store_true', default=False)
    parser.add_argument('--tanh_w_gain', type=float, default=2)  # theta
    parser.add_argument('--tanh_a_gain', type=float, default=2000) # Forth
    parser.add_argument('--ref_mins', type=float, nargs="+", default=None)
    parser.add_argument('--ref_maxs', type=float, nargs="+", default=None)
    parser.add_argument('--alpha_v', type=float, default=0.1)
    parser.add_argument('--v_margin', type=float, default=0.1)
    parser.add_argument('--cls_weight', type=float, default=1.0)
    parser.add_argument('--norm_thres', type=float, default=0.1)
    parser.add_argument('--err_mins', type=float, nargs="+", default=[-2, -2])
    parser.add_argument('--err_maxs', type=float, nargs="+", default=[2, 2])
    parser.add_argument('--abs_mins', type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument('--abs_maxs', type=float, nargs="+", default=[5.0, 5.0])
    parser.add_argument('--clf_ell_eye', action='store_true', default=False)

    parser.add_argument('--normalize', action="store_true", default=False)
    parser.add_argument('--load_last', action='store_true', default=False)
    parser.add_argument('--no_viz', action='store_true', default=False)
    parser.add_argument('--train_actor_only', action='store_true', default=False)
    parser.add_argument('--train_intervals', type=int, nargs="+", default=None)  # first actor, then clf
    parser.add_argument('--clf_lr', type=float, default=None)
    parser.add_argument('--actor_lr', type=float, default=None)

    parser.add_argument('--clf_eye_scale', type=float, default=1.0)

    parser.add_argument('--train_ratio', action='store_true', default=False)
    parser.add_argument('--mock_net', action='store_true', default=False)
    parser.add_argument('--u_norm', action='store_true', default=False)
    parser.add_argument('--split_ratio', type=float, default=0.8)

    parser.add_argument('--vio_thres', type=int, default=0)
    parser.add_argument('--roa_bloat', type=float, default=0.0)
    parser.add_argument('--train_alpha_v', type=float, default=None)

    parser.add_argument('--estimate_roa', action='store_true', default=False)
    parser.add_argument('--net_tlen', type=int, default=3)
    parser.add_argument('--net_num_tests', type=int, default=1000)
    parser.add_argument('--net_num_configs', type=int, default=50000)
    parser.add_argument('--net_exp', action='store_true', default=False)
    parser.add_argument('--net_hiddens', type=int, nargs="+", default=[64, 64])
    parser.add_argument('--net_setups', type=float, nargs="+", default=None)
    parser.add_argument('--net_rel_loss', action='store_true', default=False)

    parser.add_argument('--model_iter', type=int, nargs="+", default=None)

    parser.add_argument('--multi_initials', action='store_true', default=False)

    args = parser.parse_args()

    args.X_DIM = 2  # (xdot, y)
    args.N_REF = 2  # (xdot', y')
    args.U_DIM = 2  # (theta, P)
    args.load_last = True

    if hasattr(args, "model_iter") == False:
        args.model_iter = None

    if args.model_iter is not None:
        bak_list = args.model_iter
        exp_name = args.exp_name
        total_t1=time.time()
        for bak_val in bak_list:
            args.model_iter = bak_val
            args.exp_name = exp_name + "_%05d"%(args.model_iter)
            t1 = time.time()
            main()
            t2 = time.time()
            print("ROA training finished in %.4fs" % (t2 - t1))
        print("RoA Training total finished in %.4fs"%(time.time() - total_t1))
    else:
        tt1 = time.time()
        main()
        tt2 = time.time()

        print("Finished in %.4f seconds" % (tt2 - tt1))
