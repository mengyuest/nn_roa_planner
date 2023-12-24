import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import roa_utils as utils
import cgw_sim_choi as choi

def to_np(x):
    return x.detach().cpu().numpy()

def uniform(num_samples, xmin_xmax):
    x = np.random.rand(num_samples)
    xmin, xmax = xmin_xmax
    return x * (xmax - xmin) + xmin
    
def dq_uniform(dqs, n):
    d = len(dqs)
    x = np.random.rand(n, d)
    r = np.array(dqs)
    x = (x - 0.5) * 2 * r
    return x

def generate_traj_s(args):
    # gait data
    gait_data = args.gait_data
    N = args.num_sampling
    if args.evenly_q1:
        q1 = uniform(N, (args.th1d, -args.th1d))
        ref_q1 = choi.get_closest(q1, gait_data, dim=0)
        ref_q2 = choi.get_closest(q1, gait_data, dim=1)
        ref_q1d = choi.get_closest(q1, gait_data, dim=2)
        ref_q2d = choi.get_closest(q1, gait_data, dim=3)
    else:
        idx = np.random.randint(low=0, high=gait_data.shape[0], size=(N,))
        ref_q1 = gait_data[idx, 0]
        ref_q2 = gait_data[idx, 1]
        ref_q1d = gait_data[idx, 2]
        ref_q2d = gait_data[idx, 3]

    # add perturbation around
    ref_s = np.stack((ref_q1, ref_q2, ref_q1d, ref_q2d), axis=-1)
    noise_s = dq_uniform(args.radius_init, N)
    init_s = ref_s + noise_s
    return torch.from_numpy(init_s).float()


def pretrain_clf_from_uref(clf, actor, uref_func, data, args):
    num_epochs, lr, print_freq, save_freq, viz_freq = args.pretrain_setups
    num_epochs = int(num_epochs)
    print_freq = int(print_freq)
    save_freq = int(save_freq)
    viz_freq = int(viz_freq)

    train_zero_losses, train_dec_losses, train_losses = utils.get_n_meters(3)

    optimizer = torch.optim.RMSprop(clf.parameters(), lr)
    for epi in range(num_epochs):
        #if epi == 0 or (args.regen_init and epi % args.regen_freq == 0 and epi != 0):
        if epi % args.regen_freq == 0:
            if args.traj_sampling:
                sa_x = generate_traj_s(args)
            else:
                raise NotImplementedError
            if args.gpus is not None:
                sa_x = sa_x.cuda()

        sa_u = uref_func(sa_x)
        sa_x_curr = sa_x
        sa_x_next = sa_x
        for tti in range(args.num_sim_steps):
            xdot = choi.compute_xdot(sa_x_next, sa_u, use_torch=True, args=args)
            sa_x_next = sa_x_next + xdot * (args.dt / args.num_sim_steps)

        sa_ref_zero = clf.get_ref(sa_x)
        sa_loss_zero = clf(sa_ref_zero, sa_ref_zero)  # (N, 1)

        sa_v_curr = clf(sa_x_curr)
        sa_v_next = clf(sa_x_next)
        sa_loss_dec = torch.relu(sa_v_next - (1 - args.alpha_v) * sa_v_curr + args.v_thres)  # (N, 1)

        sa_loss_zero = args.zero_weight * torch.mean(sa_loss_zero)
        sa_loss_dec = args.dec_weight * torch.mean(sa_loss_dec)
        sa_loss = sa_loss_zero + sa_loss_dec

        utils.update(train_zero_losses, sa_loss_zero)
        utils.update(train_dec_losses, sa_loss_dec)
        utils.update(train_losses, sa_loss)

        sa_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epi % print_freq == 0 or epi == num_epochs - 1:
            print("PRET-CLF %05d/%05d L:%.3f (%.3f) Z:%.3f (%.3f) D:%.3f (%.3f)" % (
                    epi, num_epochs,
                    train_losses.val, train_losses.avg, train_zero_losses.val, train_zero_losses.avg,
                    train_dec_losses.val, train_dec_losses.avg))
        if (not args.skip_viz) and epi % viz_freq == 0 or epi == num_epochs - 1:
            plot_clf_heat(clf, epi, args, header="aprior_")

    return clf


def pretrain_actor_from_uref(clf, actor, uref_func, data, args):
    num_epochs, lr, print_freq, save_freq, viz_freq = args.pretrain_setups
    num_epochs = int(num_epochs)
    print_freq = int(print_freq)
    save_freq = int(save_freq)
    viz_freq = int(viz_freq)

    train_losses, test_losses = utils.get_n_meters(2)

    optimizer = torch.optim.RMSprop(actor.parameters(), lr)
    for epi in range(num_epochs):
        if epi % args.regen_freq == 0:
            if args.traj_sampling:
                sa_x = generate_traj_s(args)
            else:
                raise NotImplementedError
            if args.gpus is not None:
                sa_x = sa_x.cuda()

        sa_u_ref = uref_func(sa_x)
        sa_u_nn = actor(sa_x)

        square_loss = torch.square(sa_u_ref - sa_u_nn)
        num_train = int(square_loss.shape[0] * 0.8)
        train_loss = torch.mean(square_loss[:num_train])
        test_loss = torch.mean(square_loss[num_train:])

        utils.update(train_losses, train_loss)
        utils.update(test_losses, test_loss)

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epi % print_freq == 0 or epi == num_epochs - 1:
            print("PRET-U %05d/%05d train:%.3f (%.3f) test:%.3f (%.3f)" % (
                    epi, num_epochs, train_losses.val, train_losses.avg, test_losses.val, test_losses.avg))

    return actor


def plot_sim(xs, xfs, ti, header, title, args, indices=None):
    l = 1.0
    offset = 0
    doff = 3
    left_end = -2

    if args.test:
        L = 5
        ynL = 1
    else:
        L = 10
        ynL = 1

    for ii, idx in enumerate(indices):
        off = offset + ii * doff
        # slope
        s_x0 = left_end
        s_x1 = L * np.cos(args.alpha)
        s_y0 = L * np.sin(args.alpha) + off
        s_y1 = off

        # bipedal
        q1 = xs[idx, 0]
        q2 = xs[idx, 1]
        xf = xfs[idx]
        q1x = xf * np.cos(args.alpha)
        q1y = (L - xf) * np.sin(args.alpha)
        cx = q1x - l * np.sin(q1 - args.alpha)
        cy = q1y + l * np.cos(q1 - args.alpha)
        q2x = cx + l * np.sin(q1 + q2 - args.alpha)
        q2y = cy - l * np.cos(q1 + q2 - args.alpha)

        plt.plot([s_x0, s_x1], [s_y0, s_y1], color="brown", label="slope" if ii == 0 else None)
        plt.plot([q1x, cx], [q1y + off, cy + off], color="blue", label="leg-stance" if ii == 0 else None)
        plt.plot([q2x, cx], [q2y + off, cy + off], color="red", label="leg-swing" if ii == 0 else None)
        plt.scatter(q1x, q1y + off, color="red", label="Foot" if ii == 0 else None)
        plt.scatter(cx, cy + off, color="black", label="CoM" if ii == 0 else None)

    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("scaled")
    plt.xlim(left_end, L)
    plt.ylim(0, ynL * L)
    plt.title("SIM (t=%04d/%04d) %s"%(ti, args.nt, title))
    plt.savefig("%s/%s_%04d.png"%(args.viz_dir, header, ti), bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_examine(x_list_np, the_mask_np, s, gait_data, actor, args, epi):
    f_list=[]
    g_list=[]
    x_list=[s]
    u_list=[]
    m_list=[]
    x = s

    for ti in range(args.nt):
        prev_x = x
        u = actor(x.detach())
        for tti in range(args.num_sim_steps):
            fvec = choi.get_fvec(x, use_torch=True)
            gvec = choi.get_gvec(x, use_torch=True)
            f_list.append(fvec)
            g_list.append(gvec)
            xdot = fvec + gvec * u
            x = x + xdot * (args.dt / args.num_sim_steps)
        if args.fine_switch:
            x_mid = choi.compute_fine(x, prev_x, args)
            x_plus = choi.compute_impact(x_mid, use_torch=True)
        else:
            x_plus = choi.compute_impact(x, use_torch=True)
        mask = choi.detect_switch(x, prev_x, args)
        x = x * (1 - mask) + x_plus * mask
        x_list.append(x)
        u_list.append(u)
        m_list.append(mask)

    f_list = torch.stack(f_list, dim=1)
    g_list = torch.stack(g_list, dim=1)
    x_list = torch.stack(x_list, dim=1)
    u_list = torch.stack(u_list, dim=1)
    m_list = torch.stack(m_list, dim=1)

    f_np = to_np(f_list)
    g_np = to_np(g_list)
    x_np = to_np(x_list)
    u_np = to_np(u_list)
    m_np = to_np(m_list)

    ind_list = [(0, 1), (1, 2), (2, 3), (1, 3)]
    labels = ["q1 (rad)", "q2 (rad)", "q1d (rad/s)", "q2d (rad/s)"]
    mins = args.exam_mins  # [-0.5, -2.0, -4, -8.0]
    maxs = args.exam_maxs  # [0.5, 2.0, 4, 8.0]
    os.makedirs("%s/debug/" % (args.viz_dir), exist_ok=True)

    if args.colorful:
        rand_val = np.random.random(args.num_samples)

    # plot mag-increase plot
    err_list=[]
    for ti in range(args.nt):
        q1_query_ful = x_np[:, ti, 0].reshape((-1, 1))
        q1 = gait_data[:, 0].reshape((1, -1))
        indices = np.argmin(np.abs(q1 - q1_query_ful), axis=1)
        err_np = x_np[:, ti, :] - gait_data[indices, :]
        err_list.append(err_np)
    err_list = np.stack(err_list, axis=1)
    is_inc_list=[]
    for ti in range(1, args.nt):
        is_inc = np.all(np.abs(err_list[:,ti,:]) > np.abs(err_list[:,ti-1,:]), axis=-1)
        is_inc = np.logical_and(is_inc, m_np[:, ti, 0]==False)
        is_inc_list.append(is_inc)
    is_inc_list = np.stack(is_inc_list, axis=1)
    is_inc_list = is_inc_list.astype(np.float32)


    plt.figure(figsize=(8, 8))
    for i in range(args.num_all_inc):
        plt.subplot(args.num_all_inc + 1, 1, i+1)
        plt.plot(range(args.nt-1), is_inc_list[i, :])
        plt.xlabel("time step")
        plt.ylabel("sample %02d - all inc"%(i))
    plt.subplot(args.num_all_inc + 1, 1, args.num_all_inc + 1)
    plt.plot(range(args.nt-1), np.mean(is_inc_list, axis=0))
    plt.xlabel("time step")
    plt.ylabel("increase rate")
    plt.tight_layout()
    plt.savefig("%s/debug/a_curve_e%05d.png" % (args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
    plt.close()

    for ti in range(args.nt):
        q1_query_ful = x_np[:, ti, 0].reshape((-1, 1))
        q1 = gait_data[:, 0].reshape((1, -1))
        indices = np.argmin(np.abs(q1 - q1_query_ful), axis=1)
        err_np = x_np[:, ti, :] - gait_data[indices, :]

        plt.figure(figsize=(10, 10))

        fontsize = 20
        markersize = 8
        num_ticks = 5
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)

            idx0, idx1 = ind_list[i]
            if args.elapse:
                for exi in range(args.num_samples):
                    plt.plot(err_list[exi, :ti+1, idx0], err_list[exi, :ti+1, idx1], c="royalblue", alpha=0.5)
            if args.colorful:
                plt.scatter(err_np[:, idx0], err_np[:, idx1], s=markersize, c=rand_val)
            else:
                plt.scatter(err_np[:, idx0], err_np[:, idx1], s=markersize)
            plt.scatter(0, 0, color="red", s=markersize * 3)
            plt.xlim(mins[idx0], maxs[idx0])
            plt.ylim(mins[idx1], maxs[idx1])

            ax = plt.gca()
            xmin, xmax = ax.get_xlim()
            ax.set_xticks(np.round(np.linspace(xmin, xmax, num_ticks), 2))
            ymin, ymax = ax.get_ylim()
            ax.set_yticks(np.round(np.linspace(ymin, ymax, num_ticks), 2))

            plt.xlabel(labels[idx0], fontsize=fontsize)
            plt.ylabel(labels[idx1], fontsize=fontsize)
        plt.suptitle("t=%03d/%03d" % (ti, args.nt), fontsize=fontsize)
        plt.tight_layout()
        plt.savefig("%s/debug/err_e%05d_t%03d.png" % (args.viz_dir, epi, ti), bbox_inches='tight', pad_inches=0.1)
        plt.close()

    os.system("convert -delay 10 -loop 0 %s/debug/err*0.png %s/debug/ani_e%05d.gif" % (args.viz_dir, args.viz_dir, epi))

    for idx in range(args.num_samples):
        if idx >= 20:
            break
        # idx = 0
        print("idx=",idx)
        q1_query_ful = x_np[idx,:-1,0].reshape((-1, 1))
        q1 = gait_data[:, 0].reshape((1, -1))
        indices = np.argmin(np.abs(q1 - q1_query_ful), axis=1)

        plt.figure(figsize=(10, 10))
        plt.subplot(12, 1, 1)
        plt.plot(range(args.nt), f_np[idx, ::2, 0], label="f(x)_0", color="blue")
        plt.legend()

        plt.subplot(12, 1, 2)
        plt.plot(range(args.nt), f_np[idx, ::2, 1], label="f(x)_1", color="blue")
        plt.legend()

        plt.subplot(12, 1, 3)
        plt.plot(range(args.nt), f_np[idx, ::2, 2], label="f(x)_2", color="blue")
        plt.legend()

        plt.subplot(12, 1, 4)
        plt.plot(range(args.nt), f_np[idx, ::2, 3], label="f(x)_3", color="blue")
        plt.legend()

        plt.subplot(12, 1, 5)
        plt.plot(range(args.nt), g_np[idx, ::2, 2], label="g(x)_2", color="blue")
        plt.legend()

        plt.subplot(12, 1, 6)
        plt.plot(range(args.nt), g_np[idx, ::2, 3], label="g(x)_3", color="blue")
        plt.legend()

        plt.subplot(12, 1, 7)
        plt.plot(range(args.nt), g_np[idx, ::2, 2] * u_np[idx, :, 0], label="g(x)_2 u", color="blue")
        plt.legend()

        plt.subplot(12, 1, 8)
        plt.plot(range(args.nt), g_np[idx, ::2, 3] * u_np[idx, :, 0], label="g(x)_3 u", color="blue")
        plt.legend()

        plt.subplot(12, 1, 9)
        plt.plot(range(args.nt), x_np[idx, :-1, 0], label="x_0", color="blue")
        plt.plot(range(args.nt), gait_data[indices, 0], label="gait x_0", color="black", linestyle="--")
        plt.legend()

        plt.subplot(12, 1, 10)
        plt.plot(range(args.nt), x_np[idx, :-1, 1], label="x_1", color="blue")
        plt.plot(range(args.nt), gait_data[indices, 1], label="gait x_1", color="black", linestyle="--")
        plt.legend()

        plt.subplot(12, 1, 11)
        plt.plot(range(args.nt), x_np[idx, :-1, 2], label="x_2", color="blue")
        plt.plot(range(args.nt), gait_data[indices, 2], label="gait x_2", color="black", linestyle="--")
        plt.legend()

        plt.subplot(12, 1, 12)
        plt.plot(range(args.nt), x_np[idx, :-1, 3], label="x_3", color="blue")
        plt.plot(range(args.nt), gait_data[indices, 3], label="gait x_3", color="black", linestyle="--")
        plt.legend()

        plt.tight_layout()
        plt.savefig("%s/debug/e%05d_%04d.png" % (args.viz_dir, epi, idx), bbox_inches='tight', pad_inches=0.1)
        plt.close()



def plot_q1_others_curve(xs_np, gait_data, est_y_np, valids_np, ratio_val_np, args, epi):
    os.makedirs("%s/q1" % (args.viz_dir), exist_ok=True)
    xmin = [-0.2, -0.4, -1.25, -1.5]
    xmax = [0.2, 0.4, 0.25, 2.5]
    labels = ["q1 (rad)", "q2 (rad)", "q1d (rad/s)", "q2d (rad/s)"]
    ind = [(0, 1), (0, 2), (0, 3), (2, 3)]
    selected = ratio_val_np[:, 0]==0
    for ti in range(args.nt):
        if ti % args.print_sim_freq == 0:
            s = 8
            alpha=0.5
            plt.figure(figsize=(8, 8))
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                idx0, idx1 = ind[i]
                plt.plot(gait_data[:, idx0], gait_data[:, idx1], linestyle="--", color="black", alpha=0.5, label="gait")

                if i == 3:
                    plt.plot(est_y_np[:, idx0 - 1], est_y_np[:, idx1 - 1], linestyle="-.", color="red", label="nn-est")
                else:
                    plt.plot(gait_data[:, idx0], est_y_np[:, idx1 - 1], linestyle="-.", color="red", label="nn-est")
                if args.elapse:
                    for exi in range(args.num_samples):
                        plt.plot(xs_np[exi, :ti+1, idx0], xs_np[exi, :ti+1, idx1], c="royalblue", alpha=0.5)
                plt.scatter(xs_np[:, ti, idx0], xs_np[:, ti, idx1], color="blue", s=s, alpha=alpha)

                if args.plot_valid_color:
                    plt.scatter(xs_np[selected, ti, idx0], xs_np[selected, ti, idx1], color="red", s=s, alpha=alpha)

                plt.xlabel(labels[idx0])
                plt.ylabel(labels[idx1])
                plt.xlim(xmin[idx0], xmax[idx0])
                plt.ylim(xmin[idx1], xmax[idx1])
            plt.tight_layout()
            plt.savefig("%s/q1/fit_e%05d_t%04d.png" % (args.viz_dir, epi, ti), bbox_inches='tight', pad_inches=0.1)
            plt.close()


def plot_traj_roa(x_curr, x_next, v_curr, v_next, args, epi):
    mask = (v_next - v_curr) <= 0

    gait_data = args.gait_data
    ms = 8
    # mask = mask
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.scatter(x_curr[np.where(mask==1), 0], x_curr[np.where(mask==1), 1], label="stable", color="blue", s=ms)
    plt.scatter(x_curr[np.where(mask==0), 0], x_curr[np.where(mask==0), 1], label="unstable", color="red", s=ms)
    plt.plot(gait_data[:, 0], gait_data[:, 1], color="black", linestyle="--", label="gait")
    plt.legend()
    plt.xlabel("q1 (rad)")
    plt.ylabel("q2 (rad)")

    plt.subplot(2, 2, 2)
    plt.scatter(x_curr[np.where(mask == 1), 2], x_curr[np.where(mask == 1), 3], label="stable", color="blue", s=ms)
    plt.scatter(x_curr[np.where(mask == 0), 2], x_curr[np.where(mask == 0), 3], label="unstable", color="red", s=ms)
    plt.plot(gait_data[:, 2], gait_data[:, 3], color="black", linestyle="--", label="gait")
    plt.legend()
    plt.xlabel("q1d (rad/s)")
    plt.ylabel("q2d (rad/s)")

    plt.subplot(2, 2, 3)
    max_plot=100
    for i in range(min(x_curr.shape[0], 100)):
        plt.plot([x_curr[i, 0], x_next[i, 0]], [x_curr[i, 1], x_next[i, 1]], color="blue")
    plt.scatter(x_next[:max_plot, 0], x_next[:max_plot, 1], color="red", s=ms, marker="v")
    plt.plot(gait_data[:, 0], gait_data[:, 1], color="black", linestyle="--", label="gait")
    plt.xlabel("q1 (rad)")
    plt.ylabel("q2 (rad)")

    plt.subplot(2, 2, 4)
    for i in range(min(x_curr.shape[0], 100)):
        plt.plot([x_curr[i, 2], x_next[i, 2]], [x_curr[i, 3], x_next[i, 3]], color="blue")
    plt.scatter(x_next[:max_plot, 2], x_next[:max_plot, 3], color="red", s=ms, marker="v")
    plt.plot(gait_data[:, 2], gait_data[:, 3], color="black", linestyle="--", label="gait")
    plt.xlabel("q1d (rad/s)")
    plt.ylabel("q2d (rad/s)")

    plt.tight_layout()
    plt.savefig("%s/traj_roa_e%05d.png" % (args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_lya_curve(v_list_np, the_mask_np, epi, args):
    assert the_mask_np.shape[0] == args.num_samples
    for ni in range(args.num_samples):
        if args.test or the_mask_np[ni, -1, 0] == 1:
            if args.test:
                NT = 300 + 1
            else:
                NT = args.nt + 1
            plt.plot(range(NT), v_list_np[ni, :NT, 0], alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("CLF")
    plt.savefig("%s/e%05d_curve.png" % (args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_phase(x_list_np, val_list_np, gait_data=None, epi=None, args=None, new_title=""):
    xmin = -0.5
    xmax = 0.5
    ymin = -1.0
    ymax = 1.0
    draw_x = np.clip(x_list_np[:,:,0], xmin, xmax)
    draw_y = np.clip(x_list_np[:,:,1], ymin, ymax)

    xdmin = -1.0
    xdmax = 1.0
    ydmin = -2.0
    ydmax = 2.0
    draw_xd = np.clip(x_list_np[:, :, 2], xdmin, xdmax)
    draw_yd = np.clip(x_list_np[:, :, 3], ydmin, ydmax)

    for ti in range(args.nt):
        if ti % args.print_sim_freq == 0:
            plt.subplot(1, 2, 1)
            if gait_data is not None:
                plt.plot(gait_data[:,0], gait_data[:,1], color="black", linestyle="--", label="gait")
                plt.plot(np.linspace(-0.5, 0, 100), np.linspace(1.0, 0.0, 100), color="pink", label="switching")

            plt.scatter(draw_x[:, ti], draw_y[:, ti], s=1.0, alpha=0.3, color="blue")
            plt.legend()
            plt.axis("scaled")
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.xlabel("q1 (rad)")
            plt.ylabel("q2 (rad)")
            plt.title("Phase (t=%04d/%04d)" % (ti, args.nt))

            plt.subplot(1, 2, 2)
            if gait_data is not None:
                plt.plot(gait_data[:, 2], gait_data[:, 3], color="black", linestyle="--", label="gait")
            plt.scatter(draw_xd[:, ti], draw_yd[:, ti], s=1.0, alpha=0.3, color="blue")
            plt.legend()
            plt.axis("scaled")
            plt.xlim(xdmin, xdmax)
            plt.ylim(ydmin, ydmax)
            plt.xlabel("q1d (rad/s)")
            plt.ylabel("q2d (rad/s)")
            # plt.title("Phase (t=%04d/%04d)" % (ti, args.nt))

            plt.savefig("%s/e%05d_phase_%s%04d.png" % (args.viz_dir, epi, new_title, ti), bbox_inches='tight', pad_inches=0.1)
            plt.close()


def plot_phase_full(x_list_np, val_list_np, gait_data=None, epi=None, args=None, new_title=""):
    x_mins = []
    x_maxs = []
    ind_list = [(0, 1), (0, 2), (0,3), (2,3)]
    labels = ["q1 (rad)", "q2 (rad)", "q1d (rad/s)", "q2d (rad/s)"]
    xmin = [-0.2, -0.4, -1.25, -1.5]
    xmax = [0.2, 0.4, 0.25, 2.5]

    for ti in range(args.nt):
        if ti % 3 == 0:
            plt.figure(figsize=(6, 6))
            for i in range(4):
                plt.subplot(2, 2, i+1)
                ind1, ind2 = ind_list[i]
                plt.plot(gait_data[:, ind1], gait_data[:, ind2], color="black", linestyle="--", label="gait")
                plt.scatter(x_list_np[:, ti, ind1], x_list_np[:, ti, ind2], s=1.0, alpha=0.3, color="blue")
                plt.legend()
                # plt.axis("scaled")
                plt.xlim(xmin[ind1], xmax[ind1])
                plt.ylim(xmin[ind2], xmax[ind2])
                plt.xlabel(labels[ind1])
                plt.ylabel(labels[ind2])
                plt.tight_layout()
            plt.title("Phase (t=%04d/%04d)" % (ti, args.nt))
            plt.savefig("%s/e%05d_phful_%s%04d.png" % (args.viz_dir, epi, new_title, ti), bbox_inches='tight', pad_inches=0.1)
            plt.close()


def plot_error_curve(x_list_np, epi, args):
    NT = 300 + 1
    NTS = range(NT)

    q1 = x_list_np[0, :NT, 0]
    q2 = x_list_np[0, :NT, 1]
    q1d = x_list_np[0, :NT, 2]
    q2d = x_list_np[0, :NT, 3]

    q1_ref = q1
    q2_ref = choi.poly(q1, args.params)
    q1d_ref = q1d
    q2d_ref = q2d

    plt.subplot(4, 1, 1)
    plt.plot(NTS, q1_ref, color="blue", label="q1_ref")  # q1ref
    plt.plot(NTS, q1, color="red", label="q1")  # q1
    plt.xlabel("timestep")
    plt.ylabel("value")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(NTS, q2_ref, color="blue", label="q2_ref")  # q1ref
    plt.plot(NTS, q2, color="red", label="q2")  # q1
    plt.xlabel("timestep")
    plt.ylabel("value")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(NTS, q1d_ref, color="blue", label="q1d_ref")  # q1d_ref
    plt.plot(NTS, q1d, color="red", label="q1d")  # q1d
    plt.xlabel("timestep")
    plt.ylabel("value")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(NTS, q2d_ref, color="blue", label="q2d_ref")  # q2d_ref
    plt.plot(NTS, q2d, color="red", label="q2d")  # q2d
    plt.xlabel("timestep")
    plt.ylabel("value")
    plt.legend()

    plt.savefig("%s/e%05d_err.png" % (args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_roa(x_list, the_mask, val_list, epi, args):
    valids = np.all(val_list, axis=1).flatten()
    plt.scatter(x_list[np.where(valids), 0, 0], x_list[np.where(valids), 0, 1], color="blue", s=1)
    plt.scatter(x_list[np.where(valids == False), 0, 0], x_list[np.where(valids == False), 0, 1], color="red", s=1)
    plt.xlabel("q1 (rad)")
    plt.ylabel("q2 (rad)")
    plt.savefig("%s/e%05d_roa.png" % (args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_real_roa(clf, actor, args, epi, oracle=None):
    # dim (0, 1) | (1, 2) | (1, 3) | (2, 3)
    # initialize (q1, q2, q1d, q2d)
    # when choosing the rest 2 coords, the rest 2 will be the same equilibrium
    N = args.num_samples
    T = args.nt
    ref_q = args.gait_data[0]

    # generate the random
    dq = np.random.rand(N, 4)
    dq_vec = np.array(args.dq_init)
    dq = (dq - 0.5) * 2 * dq_vec
    s_rand = ref_q + dq

    # (q1, q2) case
    def task(ind, ind_keep, sub_i, num_t):
        s = np.array(s_rand)
        s[:, ind_keep[0]] = ref_q[ind_keep[0]]
        s[:, ind_keep[1]] = ref_q[ind_keep[1]]

        # simulate the trajectory
        x = torch.from_numpy(s).float().cuda()
        x_list = [x]
        for ti in range(T):
            prev_x = x
            if args.opt_ctr:
                u = choi.get_hjb_u(x, oracle)
            else:
                u = actor(x.detach())

            # dynamics
            for tti in range(args.num_sim_steps):
                xdot = choi.compute_xdot(x, u, use_torch=True, args=args)
                x = x + xdot * (args.dt / args.num_sim_steps)

            # jump/reset
            x_plus = choi.compute_impact(x, use_torch=True)
            mask = choi.detect_switch(x, prev_x)
            x = x * (1 - mask) + x_plus * mask
            x_list.append(x)
        x_list = torch.stack(x_list, dim=1)

        # compute the V decreasing cases
        v_list = clf(x_list.reshape(N*(T+1), -1)).reshape(N, T+1, 1)

        x_list = x_list.detach().cpu()

        # TODO(temp)
        v_list = v_list[:, :num_t]

        roa_mask = torch.all(v_list[:, 1:] < (1 - args.alpha) * v_list[:, :-1], dim=1)
        roa_mask = roa_mask.float()
        plt.subplot(2, 2, sub_i)
        plt.scatter(x_list[torch.where(roa_mask!=1)[0], 0, ind[0]], x_list[torch.where(roa_mask!=1)[0], 0, ind[1]], color="red", label="unstable", s=1.0, alpha=0.5)
        plt.scatter(x_list[torch.where(roa_mask)[0], 0, ind[0]], x_list[torch.where(roa_mask)[0], 0, ind[1]], color="blue", label="stable", s=1.0, alpha=0.5)
        plt.xlabel(strs[ind[0]])
        plt.ylabel(strs[ind[1]])
        plt.legend()
        plt.tight_layout()
    strs = ["q1 (rad)", "q2 (rad)", "q1d (rad/s)", "q2d (rad/s)"]

    for num_t in [2, 3, 5]:
        task([0, 1], [2, 3], 1, num_t)
        task([1, 2], [0, 3], 2, num_t)
        task([1, 3], [0, 2], 3, num_t)
        task([2, 3], [0, 1], 4, num_t)
        plt.savefig("%s/e%05d_real_roa_T%d.png" % (args.viz_dir, epi, num_t), bbox_inches='tight', pad_inches=0.1)
        plt.close()


def plot_curve(x_list_np, xf_list_np, u_list_np, v_list_np, gait_data, epi, args):
    indices=[0, 1, 2, 3]
    for idx in indices:
        gait_q1 = choi.get_closest(x_list_np[idx, :, 0], gait_data, dim=0)
        gait_q2 = choi.get_closest(x_list_np[idx, :, 0], gait_data, dim=1)
        gait_q1d = choi.get_closest(x_list_np[idx, :, 0], gait_data, dim=2)
        gait_q2d = choi.get_closest(x_list_np[idx, :, 0], gait_data, dim=3)

        ref_q1 = args.params.th1d * np.ones_like(gait_q1)
        ref_q2 = choi.poly(x_list_np[idx, :, 0], args.params)
        ref_q1d = args.q1d_ref * np.ones_like(gait_q1d)
        ref_q2d = choi.dpoly(x_list_np[idx, :, 0], x_list_np[idx, :, 2], args.params)
        for ti in range(x_list_np.shape[1]):
            if ti % (args.print_sim_freq) == 0:
                fig = plt.figure(figsize=(10, 10))
                gs = GridSpec(8, 1)
                ts = range(ti)
                alpha = 0.5
                # q1
                fig.add_subplot(gs[0, 0])
                plt.plot(ts, x_list_np[idx, :ti, 0], color="red", label="sim")
                plt.plot(ts, ref_q1[:ti], linestyle="-.", color="purple", label="track", alpha=alpha)
                plt.plot(ts, gait_q1[:ti], linestyle="--", color="black", label="gait", alpha=alpha)
                plt.xlim(0, args.nt+1)
                plt.legend()

                fig.add_subplot(gs[1, 0])
                plt.plot(ts, x_list_np[idx, :ti, 1], color="red", label="sim")
                plt.plot(ts, ref_q2[:ti], linestyle="-.", color="purple", label="track", alpha=alpha)
                plt.plot(ts, gait_q2[:ti], linestyle="--", color="black", label="gait", alpha=alpha)
                plt.xlim(0, args.nt + 1)
                plt.legend()

                fig.add_subplot(gs[2, 0])
                plt.plot(ts, x_list_np[idx, :ti, 2], color="red", label="sim")
                plt.plot(ts, ref_q1d[:ti], linestyle="-.", color="purple", label="track", alpha=alpha)
                plt.plot(ts, gait_q1d[:ti], linestyle="--", color="black", label="gait", alpha=alpha)
                plt.xlim(0, args.nt + 1)
                plt.legend()

                fig.add_subplot(gs[3, 0])
                plt.plot(ts, x_list_np[idx, :ti, 3], color="red", label="sim")
                plt.plot(ts, ref_q2d[:ti], linestyle="-.", color="purple", label="track", alpha=alpha)
                plt.plot(ts, gait_q2d[:ti], linestyle="--", color="black", label="gait", alpha=alpha)
                plt.xlim(0, args.nt + 1)
                plt.legend()

                fig.add_subplot(gs[4, 0])
                if args.changed_dynamics:
                    plt.plot(ts, u_list_np[idx, :ti, 2], color="red", label="control")
                else:
                    plt.plot(ts, u_list_np[idx, :ti, 0], color="red", label="control")
                plt.xlim(0, args.nt + 1)
                plt.legend()

                fig.add_subplot(gs[5, 0])
                plt.plot(ts, v_list_np[idx, :ti, 0], color="purple", label="lyapunov")
                plt.xlim(0, args.nt + 1)
                plt.legend()

                fig.add_subplot(gs[6:, 0])
                left_end = -2
                off = 0
                L = 5
                l = 1
                ynL = 1
                # slope
                s_x0 = left_end
                s_x1 = L * np.cos(args.alpha)
                s_y0 = L * np.sin(args.alpha) + off
                s_y1 = off

                # bipedal
                q1 = x_list_np[idx, ti, 0]
                q2 = x_list_np[idx, ti, 1]
                xf = xf_list_np[idx, ti]
                q1x = xf * np.cos(args.alpha)
                q1y = (L - xf) * np.sin(args.alpha)
                cx = q1x - l * np.sin(q1 - args.alpha)
                cy = q1y + l * np.cos(q1 - args.alpha)
                q2x = cx + l * np.sin(q1 + q2 - args.alpha)
                q2y = cy - l * np.cos(q1 + q2 - args.alpha)

                plt.plot([s_x0, s_x1], [s_y0, s_y1], color="brown", label="slope")
                plt.plot([q1x, cx], [q1y + off, cy + off], color="blue", label="leg-stance")
                plt.plot([q2x, cx], [q2y + off, cy + off], color="red", label="leg-swing")
                plt.scatter(q1x, q1y + off, color="red", label="Foot")
                plt.scatter(cx, cy + off, color="black", label="CoM")

                plt.legend()
                plt.xlabel("x (m)")
                plt.ylabel("y (m)")
                plt.axis("scaled")
                plt.xlim(left_end, L)
                plt.ylim(0, ynL * L)

                plt.tight_layout()

                plt.title("SIM (t=%04d/%04d)" % (ti, args.nt))
                plt.savefig("%s/dbg_e%05d_i%04d_t%04d.png" % (args.viz_dir, epi, idx, ti),
                            bbox_inches='tight', pad_inches=0.1)
                plt.close()


def plot_clf_heat(clf, epi, args, header=""):
    for qi, q1 in enumerate([-0.13, 0, 0.13]):
        plt.figure(figsize=(8, 8))
        M = 100

        err_d1, err_d2, err_d3 = args.clf_viz_err

        err2 = torch.linspace(-err_d1, err_d1, M).cuda()
        err3 = torch.linspace(-err_d2, err_d2, M).cuda()
        err4 = torch.linspace(-err_d3, err_d3, M).cuda()

        # err2 = torch.linspace(-0.5, 0.5, M).cuda()
        # err3 = torch.linspace(-1, 1, M).cuda()
        # err4 = torch.linspace(-1, 1, M).cuda()

        err_23_x, err_23_y = torch.meshgrid(err2, err3)
        err_24_x, err_24_y = torch.meshgrid(err2, err4)
        err_34_x, err_34_y = torch.meshgrid(err3, err4)

        err_23_x = err_23_x.T.reshape([-1, 1])
        err_23_y = err_23_y.T.reshape([-1, 1])
        err_24_x = err_24_x.T.reshape([-1, 1])
        err_24_y = err_24_y.T.reshape([-1, 1])
        err_34_x = err_34_x.T.reshape([-1, 1])
        err_34_y = err_34_y.T.reshape([-1, 1])


        zeros = torch.zeros_like(err_23_x)
        q1_t = torch.ones_like(err_23_x) * q1
        err1 = zeros

        v_23 = clf.compute_v(q1_t, err1, err_23_x, err_23_y, zeros)
        v_24 = clf.compute_v(q1_t, err1, err_24_x, zeros, err_24_y)
        v_34 = clf.compute_v(q1_t, err1, zeros, err_34_x, err_34_y)

        v_23 = to_np(v_23)
        v_24 = to_np(v_24)
        v_34 = to_np(v_34)

        plt.subplot(2, 2, 1)
        plt.imshow(v_23.reshape([M, M]), origin="lower")
        plt.xlabel("q2_err")
        plt.ylabel("q1d_err")
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.imshow(v_24.reshape([M, M]), origin="lower")
        plt.xlabel("q2_err")
        plt.ylabel("q2d_err")
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.imshow(v_34.reshape([M, M]), origin="lower")
        plt.xlabel("q1d_err")
        plt.ylabel("q2d_err")
        plt.colorbar()

        plt.savefig("%s/%sheat_e%05d_%02d_q1_%.4f.png" % (
            args.viz_dir, header, epi, qi, q1), bbox_inches='tight', pad_inches=0.1)
        plt.close()

def get_gait_data():
    gait_str = """
            [[ 0.1954769  -0.39095381 -0.75098276  0.84715623]
             [ 0.18701912 -0.37403822 -0.75194013  0.73399985]
             [ 0.17828843 -0.35657686 -0.73985833  0.61868256]
             [ 0.16904408 -0.33808815 -0.73270839  0.53684717]
             [ 0.15945166 -0.31890333 -0.72131479  0.46414152]
             [ 0.14933339 -0.29866678 -0.70566869  0.40447736]
             [ 0.14032325 -0.2806465  -0.68168187  0.35481673]
             [ 0.13121651 -0.26243302 -0.66095102  0.30043977]
             [ 0.12171151 -0.24342301 -0.63801199  0.24864183]
             [ 0.11198891 -0.2239778  -0.61313689  0.19832826]
             [ 0.1021308  -0.2042616  -0.58539337  0.15049179]
             [ 0.09220324 -0.18440647 -0.55220789  0.10623652]
             [ 0.08216057 -0.16432115 -0.51310581  0.06603827]
             [ 0.07203654 -0.14407308 -0.46698153  0.03131433]
             [ 0.06186418 -0.12372836 -0.41152471  0.00328764]
             [ 0.05160491 -0.10320982 -0.34765154 -0.01696834]
             [ 0.04122806 -0.08245611 -0.27609453 -0.02837917]]
             """
    return utils.load_np_from_str(gait_str)


def get_around_init_gait_switch(args, perturbed=False):
    gait_x = get_gait_data()
    gait_x = torch.from_numpy(gait_x).float()
    dx = utils.uniform_sample_tsr(args.num_samples, args.dx_mins, args.dx_maxs)
    idx = np.random.choice(gait_x.shape[0], args.num_samples)
    idx = torch.from_numpy(idx)
    cfg = (torch.linspace(-0.20, -0.04, 17) * -1).reshape((-1, 1))[idx]
    x = gait_x[idx] + dx
    x[:, 1] = -2 * x[:, 0]
    if perturbed:
        cfg = cfg + utils.uniform_sample_tsr(args.num_samples, args.dcfg_min, args.dcfg_max)
        cfg = torch.clamp(cfg, 0.04, 0.20)
    return x, cfg


def get_initial_x(args):
    if args.x_init_type == "whole":
        x = utils.uniform_sample_tsr(args.num_samples, args.x_mins, args.x_maxs)
    elif args.x_init_type == "switch":
        x = utils.uniform_sample_tsr(args.num_samples, args.x_mins, args.x_maxs)
        x[:, 1] = -2 * x[:, 0]
    elif args.x_init_type == "around_init_gait":
        gait_x = np.zeros((args.num_samples, 4))
        dx = utils.uniform_sample_tsr(args.num_samples, args.dx_mins, args.dx_maxs)
        x = gait_x + dx
    elif args.x_init_type == "around_init_gait_switch":
        x, cfg = get_around_init_gait_switch(args)
    return x


def get_initial_u(args):
    u = utils.uniform_sample_tsr(args.num_samples, args.u_mins, args.u_maxs)
    return u


def get_u(x, u_init, args):
    # u format (alpha, beta, theta, F1, F2)
    # u = alpha * QP (theta) + beta * HJB + F1 * 1(if q2<0) + F2 * 1(if q2>=0)
    # alpha, between (0, 1)
    # beta, between (0, 1)
    # theta, between (0.04, 0.20)
    # u_f1, between (-1, 1)
    # u_f2, between (-1, 1)
    alpha, beta, theta, f1, f2 = torch.split(u_init, split_size_or_sections=1, dim=-1)

    u_qp = choi.get_qp_u(x, args)
    u_hjb = choi.get_hjb_u(x, args.oracle)

    u_f1 = f1 * (x[:, 1:2] < 0).float()
    u_f2 = f2 * (x[:, 1:2] >= 0).float()
    u = (1 - beta) * alpha * u_qp + beta * u_hjb + u_f1 + u_f2
    return u


def get_u_comp(x, alpha, beta, theta, f1, f2, args):
    u_qp = choi.get_qp_u(x, args)
    u_hjb = choi.get_hjb_u(x, args.oracle)
    u_f1 = f1 * (x[:, 1:2] < 0).float()
    u_f2 = f2 * (x[:, 1:2] >= 0).float()
    u = (1 - beta) * alpha * u_qp + beta * u_hjb + u_f1 + u_f2
    return u


def bipedal_step(x, u, xf, args):
    l = 1
    prev_x = x
    for tti in range(args.num_sim_steps):
        xdot = choi.compute_xdot(x, u, use_torch=False, args=args)
        x = x + xdot * (args.dt / args.num_sim_steps)
    if args.fine_switch:
        x_mid = choi.compute_fine(x, prev_x, args, use_torch=False)
        x_plus = choi.compute_impact(x_mid, use_torch=False)
    mask = choi.detect_switch(x, prev_x, args, use_torch=False)
    x = x * (1 - mask) + x_plus * mask
    xf_plus = xf + l * np.sin(x_mid[:, 0:1] + x_mid[:, 1:2]) - l * np.sin(x_mid[:, 0:1])
    xf = xf * (1 - mask) + xf_plus * mask
    return x, xf, mask


def bipedal_next(x_init, u_init, args, obtain_full=False, xf=None,
                 use_rl=False, ppo_agent=None, agent=None, running=None, direct_u=None,
                 target_th=None, use_mbpo=False, use_pets=False):
    l = 1
    x = x_init.detach()
    x_list = [x]
    if obtain_full:
        xf_list = [xf]
        val_list = [torch.ones_like(x[:, 0:1])]
    mask_list = [torch.zeros_like(x[:, 0:1])]
    is_swi = torch.zeros_like(x[:, 0:1])
    if any([use_rl, (direct_u is not None), use_mbpo, use_pets]):
        args.params = choi.create_params()
    else:
        args.params = choi.create_params(th1d=-u_init[:, 2])

    n_samples = x_init.shape[0]
    flag_list = [None for _ in range(n_samples)]
    x_seg=[[] for _ in range(n_samples)]
    xf_seg=[[] for _ in range(n_samples)]
    val_seg=[[] for _ in range(n_samples)]

    # alpha, beta, theta, f1, f2 = torch.split(u_init, split_size_or_sections=1, dim=-1)

    dbg_t1=time.time()
    t_sim=0

    if direct_u is not None:
        nt = direct_u.shape[1]  # (N, T, 1)
    else:
        nt = args.nt

    for ti in range(nt):
        prev_x = x
        dbg_tmp = time.time()
        if any([use_rl, use_mbpo, use_pets]):
            if ppo_agent is not None:
                rl_state = torch.cat((x, target_th), dim=-1)
                u = ppo_agent.select_action(rl_state, use_torch=True, batch=True)
            else:
                the_x_th = torch.cat((x, target_th), dim=-1)
                with torch.no_grad():
                    if use_rl:
                        rl_state = running(the_x_th, update=False, use_torch=True).float()
                        if hasattr(agent, "action"):
                            u = agent.action(rl_state)
                        elif hasattr(agent, "common"):
                            rl_embed = agent.common(rl_state)
                            u = agent.policy(rl_embed)
                        elif hasattr(agent, "actor"):
                            rl_embed = agent.actor.common(rl_state)
                            u = agent.actor.policy(rl_embed)
                        else:
                            raise NotImplementedError
                    else:
                        rl_state = the_x_th.cuda().float()
                        u = utils.rl_u(rl_state, None, agent, mbpo=True)
                    u = torch.clamp(u, min=-4.0, max=4.0)
        elif direct_u is not None:
            u = direct_u[:, ti]
        else:
            u = get_u(x, u_init, args)
        dbg_ttt2=time.time()
        # print("inside %.4f" % (dbg_ttt2 - dbg_tmp))
        # u = get_u_comp(x, alpha, beta, theta, f1, f2, args)

        for tti in range(args.num_sim_steps):
            xdot = choi.compute_xdot(x, u, use_torch=True, args=args)
            x = x + xdot * (args.dt / args.num_sim_steps)

        if args.fine_switch:
            x_mid = choi.compute_fine(x, prev_x, args)
            x_plus = choi.compute_impact(x_mid, use_torch=True)

        mask = choi.detect_switch(x, prev_x, args)
        x = x * (1 - mask) + x_plus * mask
        t_sim += time.time() - dbg_tmp
        if obtain_full:
            xf_plus = xf + l * torch.sin(x_mid[:, 0:1] + x_mid[:, 1:2]) - l * torch.sin(x_mid[:, 0:1])
            xf = xf * (1 - mask) + xf_plus * mask
            xf_list.append(xf)
            val = torch.logical_and(torch.logical_and(torch.abs(x[:, 0:1])<=0.5, torch.abs(x[:, 1:2])<=1),
                                    val_list[-1])
            val_list.append(val)

        x = prev_x * is_swi + x * (1 - is_swi)
        if obtain_full:
            for i in range(n_samples):
                if flag_list[i] is None and torch.norm(prev_x[i] - x[i])==0:
                    flag_list[i] = ti

        is_swi = (torch.logical_or(is_swi, mask)).float()
        x_list.append(x)
        mask_list.append(mask)

    dbg_t2 = time.time()
    x_list_old = x_list
    x_list = torch.stack(x_list, dim=1)
    # find the first return x (if no, return last idx of x_list)
    if obtain_full:
        xf_list_old = xf_list
        val_list_old = val_list
        # for i in range(args.nt):
        #     print(i, flag, xf_list_old[i])

        # should return a list of n_samples trajs
        # each traj has n_i length of x state (N)

        xf_list = torch.stack(xf_list, dim=1)
        val_list = torch.stack(val_list, dim=1)

        for i in range(n_samples):
            x_seg[i] = x_list[i, :flag_list[i]+1, :] if flag_list[i] is not None else x_list[i]
            xf_seg[i] = xf_list[i, :flag_list[i]+1, :] if flag_list[i] is not None else xf_list[i]
            val_seg[i] = val_list[i, :flag_list[i]+1, :] if flag_list[i] is not None else val_list[i]

        return x_list[:, -1, :], is_swi, x_seg, xf_seg, val_seg
    else:
        return x_list[:, -1, :], is_swi