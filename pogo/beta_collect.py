import sys
import torch
import numpy as np
import argparse
import time
import tqdm
import beta_pogo_sim

sys.path.append("../")
import roa_utils


def main():
    roa_utils.set_seed_and_exp(args, offset=1)

    # save data in format
    # x, u, x_next, mode
    # N*4, N*2, N*4, N*1
    init_xs_list=[]
    us_list=[]
    new_xs_np_list=[]
    modes_list=[]
    for tri in tqdm.tqdm(range(args.num_trials)):
        xs_np = np.random.random((args.num_samples, 4))
        us_np = np.random.random((args.num_samples, 2))

        xmins = np.array(args.xmins)
        xmaxs = np.array(args.xmaxs)
        umins = np.array(args.umins)
        umaxs = np.array(args.umaxs)

        xs_np = xs_np * (xmaxs - xmins) + xmins
        us_np = us_np * (umaxs - umins) + umins
        modes_np = np.zeros((args.num_samples,))

        xs_np_aug = np.zeros((args.num_samples, 6))
        xs_np_aug[:, :4] = xs_np
        xs_np_aug[:, 4] = 0.0
        xs_np_aug[:, 5] = xs_np_aug[:, 2] - 1.0

        init_xs = torch.from_numpy(xs_np_aug).float()
        xs = init_xs
        us = torch.from_numpy(us_np).float()
        modes = torch.from_numpy(modes_np).float()

        x_list=[]
        m_list=[]

        for ti in range(args.nt):
            next_xs, next_modes = beta_pogo_sim.pytorch_sim(xs, modes, us, global_vars, check_invalid=True)
            x_list.append(xs.detach().cpu().numpy())
            m_list.append(modes.detach().cpu().numpy())
            xs, modes = next_xs, next_modes

        new_xs = np.zeros((args.num_samples, 4))
        # backtrace to find valid xs
        # and find first return apex state
        x_list=np.stack(x_list, axis=0)
        m_list=np.stack(m_list, axis=0)
        for ni in range(args.num_samples):
            # m6 = np.nonzero(m_list[:, ni] == 6)
            m6 = np.nonzero(m_list[:, ni] == 7)
            if m6[0].shape[0] > 0:
                idx = m6[0][0]
                new_xs[ni, :] = x_list[idx, ni, :4]
                modes[ni] = 1.0
            else:
                modes[ni] = -1.0

        new_xs_np = torch.from_numpy(new_xs).float()
        np.savez("%s/data_%05d.npz" % (args.exp_dir_full, tri), xs=init_xs, us=us, new_xs=new_xs_np, modes=modes)
        n_valids = int(np.nonzero(modes.detach().cpu().numpy() == 1.0)[0].size)
        print("valid %d/%d  %.4f"%(n_valids, args.num_samples, n_valids/args.num_samples))

        init_xs_list.append(init_xs)
        us_list.append(us)
        new_xs_np_list.append(new_xs_np)
        modes_list.append(modes)

    init_xs_total = np.concatenate(init_xs_list, axis=0)
    us_total = np.concatenate(us_list, axis=0)
    new_xs_np_total = np.concatenate(new_xs_np_list, axis=0)
    modes_total = np.concatenate(modes_list, axis=0)
    np.savez("%s/data.npz" % args.exp_dir_full, xs=init_xs_total, us=us_total, new_xs=new_xs_np_total, modes=modes_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="beta_data_dbg")
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--nt', type=int, default=3000)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=1000)

    parser.add_argument('--dt1', type=float, default=0.001)
    parser.add_argument('--dt2', type=float, default=0.0001)
    parser.add_argument('--l0', type=float, default=1.0)
    parser.add_argument('--g', type=float, default=10.0)
    parser.add_argument('--k', type=float, default=32000)
    parser.add_argument('--M', type=float, default=80)

    parser.add_argument('--xmins', type=float, nargs="+", default=None)
    parser.add_argument('--xmaxs', type=float, nargs="+", default=None)
    parser.add_argument('--umins', type=float, nargs="+", default=None)
    parser.add_argument('--umaxs', type=float, nargs="+", default=None)
    args = parser.parse_args()
    tt1 = time.time()

    num_steps = args.nt
    dt = args.dt1
    dt2 = args.dt2
    l0 = args.l0
    g = args.g
    k = args.k
    M = args.M

    global_vars = num_steps, dt, dt2, l0, g, k, M

    main()

    tt2 = time.time()
    print("Finished in %.4f seconds" % (tt2 - tt1))
