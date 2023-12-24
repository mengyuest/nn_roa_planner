import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import cgw_sim_choi as choi
import torch
import roa_utils as utils
import scipy.io


def poly_eq(th):
    c = np.array([[-2.09140172e+01, 1.01779147e+01, -2.19792131e+00, 1.19786279e+00, -3.89372615e-03],
                  [4.18284214e+01, -2.03560264e+01,  4.39587748e+00, -2.39572807e+00, 7.78751451e-03],
                  [9.78807593e+02, -5.51510121e+02,  1.27498470e+02, -1.59978385e+01, 1.94235590e-01],
                  [2.77177316e+03, -1.23266758e+03,  2.12218175e+02, -1.19956292e+01, 1.86437853e-01]])
    est_list = []
    for i in range(4):
        est_list.append(c[i][0] * th ** 4 + c[i][1] * th ** 3 + c[i][2] * th ** 2 + c[i][3]* th + c[i][4])
    return torch.cat(est_list, dim=-1)

def main():
    utils.set_seed_and_exp(args)

    # sample theta from
    thetas = torch.linspace(args.theta_min, args.theta_max, args.n_theta)
    init_x = poly_eq(thetas.unsqueeze(-1))
    init_x[:, 0] = thetas
    init_x[:, 1] = -2 * thetas

    # perturbation in dq1, dq2
    x_dq1 = utils.uniform_sample_tsr(args.n_trials, [args.dq1_min], [args.dq1_max])
    x_dq2 = utils.uniform_sample_tsr(args.n_trials, [args.dq2_min], [args.dq2_max])

    # extra n_trials dimension
    # print(x_dq1.shape)
    x_dq1 = x_dq1.unsqueeze(0).tile([args.n_theta, 1, 1])
    # print(x_dq1.shape)
    x_dq2 = x_dq2.unsqueeze(0).tile([args.n_theta, 1, 1])

    thetas = thetas.unsqueeze(1).tile([1, args.n_trials])
    init_x = init_x.unsqueeze(1).tile([1, args.n_trials, 1])

    # print(x_dq1.shape, init_x.shape)

    init_x[:, :, 2] = init_x[:, :, 2] + x_dq1[:, :, 0]
    init_x[:, :, 3] = init_x[:, :, 3] + x_dq2[:, :, 0]

    thetas = thetas.reshape((-1, 1))  # (n_theta * n_trials, 1)
    init_x = init_x.reshape((-1, 4))  # (n_theta * n_trials, 4)

    l = 1
    x = init_x
    xf = torch.zeros_like(init_x[:, 0:1])
    mask = torch.zeros_like(init_x[:, 0:1])
    valid = torch.zeros_like(init_x[:, 0:1])

    x_list = [x]
    xf_list = [xf]
    u_list = []
    mask_list = [mask]
    args.th1d = thetas[:, 0]
    args.params = choi.create_params(th1d=args.th1d)
    for ti in range(args.nt):
        if ti % 100 == 0:
            print("sim",ti)
        prev_x = x_list[-1]
        u = choi.get_qp_u(x, args)
        for tti in range(args.num_sim_steps):
            # print(x.shape, u.shape)
            xdot = choi.compute_xdot(x, u, use_torch=True, args=args)
            x = x + xdot * (args.dt / args.num_sim_steps)
        x_mid = choi.compute_fine(x, prev_x, args)
        x_plus = choi.compute_impact(x_mid, use_torch=True)
        xf_plus = xf + l * torch.sin(x_mid[:, 0:1] + x_mid[:, 1:2]) - l * torch.sin(x_mid[:, 0:1])

        mask = choi.detect_switch(x, prev_x, args)

        x = x * (1 - mask) + x_plus * mask
        xf = xf * (1 - mask) + xf_plus * mask

        x_list.append(x)
        xf_list.append(xf)
        u_list.append(u)
        mask_list.append(mask)

    x_list = torch.stack(x_list, dim=1)
    xf_list = torch.stack(xf_list, dim=1)
    u_list = torch.stack(u_list, dim=1)
    m_list = torch.stack(mask_list, dim=1)
    # find the last switch-to-switch for each traj

    x_list = x_list.reshape((args.n_theta, args.n_trials, args.nt+1, 4))
    m_list = m_list.reshape((args.n_theta, args.n_trials, args.nt+1, 1))
    cum_list = torch.cumsum(m_list, dim=-2)

    gait_data_list = []
    theta_list = []

    for i in range(args.n_theta):
        # find from the n_trials the least discrepancy one
        val_idx = torch.where(torch.sum(m_list[i], 1)>=2)[0]
        val_x = x_list[i, val_idx]  # (n_vals, nt+1, 4)
        val_m = m_list[i, val_idx]  # (n_vals, nt+1, 1)
        val_cum = cum_list[i, val_idx]  # (n_vals, nt+1, 1)
        val_last1 = torch.where(torch.logical_and(val_cum[:, :, 0] == val_cum[:, -1:, 0], val_m[:, :, 0]==1))
        val_last2 = torch.where(torch.logical_and(val_cum[:, :, 0] == val_cum[:, -1:, 0]-1, val_m[:, :, 0]==1))
        x_last1 = val_x[val_last1]
        x_last2 = val_x[val_last2]
        assert x_last1.shape[0] == x_last2.shape[0] == val_idx.shape[0]
        dev_x = torch.norm(x_last2-x_last1, dim=-1)
        smallest_idx = torch.where(dev_x == torch.min(dev_x))[0][0]
        gait_data_list.append(
            utils.to_np(val_x[smallest_idx, val_last2[1][smallest_idx]:val_last1[1][smallest_idx]]))
        theta_list.append(thetas[i * args.n_trials])
        print("%02d  th:%.4f  val:%04d  sm:%03d  T:%03d-%03d=%03d  (%.3f %.3f %.3f %.3f)-(%.3f %.3f %.3f %.3f)=err:%.4f" %
              (i, thetas[i * args.n_trials], val_x.shape[0], smallest_idx,
               val_last1[1][smallest_idx], val_last2[1][smallest_idx],
               val_last1[1][smallest_idx]-val_last2[1][smallest_idx],
               val_x[smallest_idx, val_last2[1][smallest_idx], 0],
               val_x[smallest_idx, val_last2[1][smallest_idx], 1],
               val_x[smallest_idx, val_last2[1][smallest_idx], 2],
               val_x[smallest_idx, val_last2[1][smallest_idx], 3],
               val_x[smallest_idx, val_last1[1][smallest_idx], 0],
               val_x[smallest_idx, val_last1[1][smallest_idx], 1],
               val_x[smallest_idx, val_last1[1][smallest_idx], 2],
               val_x[smallest_idx, val_last1[1][smallest_idx], 3], torch.min(dev_x)))
    theta_list = np.array(theta_list)
    np.savez("%s/gait_data.npz"%(args.exp_dir_full), data=gait_data_list, thetas=theta_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="cgw_collect")
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--nt', type=int, default=200)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--num_sim_steps', type=int, default=2)
    parser.add_argument('--qp_bound', type=float, default=4.0)
    parser.add_argument('--theta_min', type=float, default=0.1305)
    parser.add_argument('--theta_max', type=float, default=0.1305)
    parser.add_argument('--n_theta', type=int, default=50)
    parser.add_argument('--dq1_min', type=float, default=-0.5)
    parser.add_argument('--dq1_max', type=float, default=0.5)
    parser.add_argument('--dq2_min', type=float, default=-1.0)
    parser.add_argument('--dq2_max', type=float, default=1.0)
    parser.add_argument('--n_trials', type=int, default=100)

    args = parser.parse_args()
    args.fine_switch = True
    args.constant_g = False
    args.changed_dynamics = False
    args.qp_bound = 4.0
    args.reset_q1_threshold = -0.03

    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.4f seconds" % (t2 - t1))