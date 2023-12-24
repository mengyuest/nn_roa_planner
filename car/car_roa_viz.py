import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("../")
import roa_utils as utils


def plot_v_heatmap_whole(ind_list, f_list, img_path, args, viz_config=None, viz_P=None, title=None):
    str_list = ["x (m)", "y (m)", "delta (rad)", "v (m/s)", "psi (rad)", "psi_dot (rad/s)", "beta (rad)"]
    plt.figure(figsize=(8, 8))
    num_cols = len(f_list)
    num_rows = len(ind_list)
    for row_i in range(num_rows):
        for col_j in range(num_cols):
            plt.subplot(num_rows, num_cols, num_cols * row_i + col_j + 1)
            if args.multi:
                plot_v_heatmap(lambda x, pp: f_list[col_j](x,pp), indices=ind_list[row_i], str_list=str_list, args=args,
                               viz_config=viz_config, viz_P=viz_P)
            else:
                plot_v_heatmap(lambda x: f_list[col_j](x), indices=ind_list[row_i], str_list=str_list, args=args)
    if title is not None:
        plt.title(title)
    plt.tight_layout()  # (pad=0.1)  # (pad=1.08)
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_v_heatmap(func, indices, str_list, args, viz_config=None, viz_P=None):
    nx = 41
    ny = 41
    ratio = 1.2
    idx0, idx1 = indices

    xmin, xmax = utils.rescale_min_max(args.s_mins[idx0], args.s_maxs[idx0], ratio)
    ymin, ymax = utils.rescale_min_max(args.s_mins[idx1], args.s_maxs[idx1], ratio)

    viz_x = np.linspace(xmin, xmax, nx)
    viz_y = np.linspace(ymin, ymax, ny)
    viz_xy0, viz_xy1 = np.meshgrid(viz_x, viz_y, indexing='xy')
    viz_xy = np.stack((viz_xy0, viz_xy1), axis=-1)
    viz_xy = viz_xy.reshape((nx * ny, 2))
    N = viz_xy.shape[0]
    viz_s = np.zeros((N, args.X_DIM))
    viz_s[:, [idx0, idx1]] = viz_xy
    viz_s = torch.from_numpy(viz_s).float().cuda()
    if args.multi:
        viz_s = torch.cat([viz_s, viz_config.unsqueeze(0).repeat(viz_s.shape[0], 1)], dim=-1)
        v = func(viz_s, viz_P)
    else:
        v = func(viz_s)
    plt.imshow(v.detach().cpu().numpy().reshape(nx, ny).T, origin="lower")

    ax = plt.gca()
    n_xticks = 3
    n_yticks = 5
    ax.set_xticks(np.linspace(0 - 0.5, nx - 0.5, n_xticks))
    ax.set_xticklabels(["%.3f" % xx for xx in np.linspace(xmin, xmax, n_xticks)])
    ax.set_yticks(np.linspace(0 - 0.5, ny - 0.5, n_yticks))
    ax.set_yticklabels(["%.3f" % xx for xx in np.linspace(ymin, ymax, n_yticks)])
    plt.xlabel(str_list[idx0])
    plt.ylabel(str_list[idx1])
    # https://www.geeksforgeeks.org/set-matplotlib-colorbar-size-to-match-graph/
    im_ratio = 1.0 * nx / ny
    plt.colorbar(fraction=0.047*im_ratio)


def plot_sim_whole(ind_list, trs, marked, img_path, args):
    str_list = ["x (m)", "y (m)", "delta (rad)", "v (m/s)", "psi (rad)", "psi_dot (rad/s)", "beta (rad)"]
    size = 1.0
    color0 = "red"
    color1 = "blue"
    idx = marked.float().nonzero().detach().cpu().numpy()
    num_subplots = len(ind_list)
    num_cols = 2
    num_rows = num_subplots // num_cols + num_subplots % num_cols
    for ti in range(args.nt):
        if ti % 5 == 0:
            for i in range(len(ind_list)):
                idx0, idx1 = ind_list[i]
                s = trs[:, ti, :].detach().cpu().numpy()
                plt.subplot(num_rows, num_cols, i+1)
                plt.scatter(s[:, idx0], s[:, idx1], s=size, c=color0)
                plt.scatter(s[idx, idx0], s[idx, idx1], s=size * 2, c=color1)
                plt.axis("scaled")
                plt.xlim(args.s_mins[idx0] * 1.5, args.s_maxs[idx0] * 1.5)
                plt.ylim(args.s_mins[idx1] * 1.5, args.s_maxs[idx1] * 1.5)
                plt.xlabel(str_list[idx0])
                plt.ylabel(str_list[idx1])
            plt.tight_layout()
            plt.savefig(img_path%(ti), bbox_inches='tight', pad_inches=0.1)
            plt.close()


def visualization_for_lqr(lqr_trs, marked, args):
    size = 1.0
    color0 = "red"
    color1 = "blue"
    idx = marked.float().nonzero().detach().cpu().numpy()
    for ti in range(args.nt):
        if ti % 1 ==0:
            print("vis sim ti=",ti)
            s = lqr_trs[:,ti,:].detach().cpu().numpy()
            plt.subplot(2,2,1)
            plt.scatter(s[:,0], s[:,1], s=size, c=color0)
            plt.scatter(s[idx, 0], s[idx, 1], s=size*2, c=color1)
            plt.axis("scaled")
            plt.xlim(args.s_mins[0] * 1.5, args.s_maxs[0] * 1.5)
            plt.ylim(args.s_mins[1] * 1.5, args.s_maxs[1] * 1.5)

            plt.subplot(2, 2, 2)
            plt.scatter(s[:, 2], s[:, 3], s=size, c=color0)
            plt.scatter(s[idx, 2], s[idx, 3], s=size * 2, c=color1)

            plt.axis("scaled")
            plt.xlim(args.s_mins[2] * 1.5, args.s_maxs[3] * 1.5)
            plt.ylim(args.s_mins[2] * 1.5, args.s_maxs[3] * 1.5)

            plt.subplot(2, 2, 3)
            plt.scatter(s[:, 4], s[:, 5], s=size, c=color0)
            plt.scatter(s[idx, 4], s[idx, 5], s=size * 2, c=color1)
            plt.axis("scaled")
            plt.xlim(args.s_mins[4] * 1.5, args.s_maxs[4] * 1.5)
            plt.ylim(args.s_mins[5] * 1.5, args.s_maxs[5] * 1.5)

            plt.subplot(2, 2, 4)
            plt.scatter(s[:, 5], s[:, 6], s=size, c=color0)
            plt.scatter(s[idx, 5], s[idx, 6], s=size * 2, c=color1)
            plt.axis("scaled")
            plt.xlim(args.s_mins[5] * 1.5, args.s_maxs[5] * 1.5)
            plt.ylim(args.s_mins[6] * 1.5, args.s_maxs[6] * 1.5)

            plt.savefig("%s/sim_lqr_%04d.png"%(args.viz_dir, ti), bbox_inches='tight', pad_inches=0.1)
            plt.close()


def plot_policy_hist(data, title, img_path):
    data = data.detach().cpu().numpy()
    plt.hist(data, bins=30)
    plt.title(title)
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()