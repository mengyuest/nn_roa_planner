import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
sys.path.append("../")
import roa_utils as utils

from qpsolvers import solve_qp


def flight_f(s, u, global_vars, uprising=False):
    num_steps, dt, dt2, l0, g, k, M = global_vars
    new_s = torch.zeros_like(s)
    new_s[:, 0] = s[:, 0] + s[:, 1] * dt
    new_s[:, 1] = s[:, 1]
    new_s[:, 2] = s[:, 2] + s[:, 3] * dt
    new_s[:, 3] = s[:, 3] + (-g) * dt
    if uprising:
        new_s[:, 4] = s[:, 4] + (new_s[:, 0] - s[:, 0])
        new_s[:, 5] = s[:, 5] + (new_s[:, 1] - s[:, 1])
    else:
        new_s[:, 4] = new_s[:, 0] + l0 * torch.sin(u[:, 0])
        new_s[:, 5] = new_s[:, 2] - l0 * torch.cos(u[:, 0])
    return new_s


def stance_f(s, u, global_vars, is_compression=True, same_u=False):
    num_steps, dt, dt2, l0, g, k, M = global_vars
    new_s = torch.zeros_like(s)

    l = torch.sqrt((s[:, 0]-s[:, 4]) * (s[:, 0]-s[:,4]) + s[:, 2] * s[:, 2])

    new_s[:, 0] = s[:, 0] + s[:, 1] * dt2
    if is_compression or same_u:
        new_s[:, 1] = s[:, 1] + (1 / M * (s[:, 0]-s[:, 4]) / l * (u[:, 1] + k * (l0 - l))) * dt2
    else:
        new_s[:, 1] = s[:, 1] + (1 / M * (s[:, 0] - s[:, 4]) / l * (k * (l0 - l))) * dt2
    new_s[:, 2] = s[:, 2] + s[:, 3] * dt2
    if is_compression or same_u:
        new_s[:, 3] = s[:, 3] + (1 / M * s[:, 2] / l * (u[:, 1] + k * (l0 - l))-g) * dt2
    else:
        new_s[:, 3] = s[:, 3] + (1 / M * s[:, 2] / l * (k * (l0 - l)) - g) * dt2
    new_s[:, 4] = s[:, 4]
    new_s[:, 5] = s[:, 5]
    return new_s


# pytorch version
def pytorch_sim(s, modes, u, global_vars, same_u=False, check_invalid=False):
    num_steps, dt, dt2, l0, g, k, M = global_vars

    new_s = torch.zeros_like(s)
    new_modes = torch.zeros_like(modes)

    # mode 0: flight (from apex->ground)
    m0 = torch.nonzero(modes==0, as_tuple=True)[0]
    if m0.shape[0] > 0:
        new_s[m0] = flight_f(s[m0], u[m0], global_vars)
        # mode 1: event contact ()
        m1 = torch.nonzero(new_s[m0, 2]-l0*torch.cos(u[m0, 0])<=0, as_tuple=True)[0]
        if m1.shape[0] > 0:
            new_modes[m0[m1]] = 1.0
            new_s[m0[m1], 4] = new_s[m0[m1], 4] * 0.0 + new_s[m0[m1], 0] + l0 * torch.sin(u[m0[m1], 0])
            new_s[m0[m1], 5] = new_s[m0[m1], 5] * 0.0
            # new_s[m0[m1], 0] = new_s[m0[m1], 0] * 0.0 - l0 * torch.sin(u[m0[m1], 0])
            #TODO x add still accumulated values

    # mode 2: stance_compression
    m2 = torch.nonzero(torch.logical_or(modes==1, modes==2), as_tuple=True)[0]
    if m2.shape[0] > 0:
        new_s[m2] = stance_f(s[m2], u[m2],global_vars,is_compression=True, same_u=same_u)
        new_modes[m2] = 2.0

        # mode 3: event stance_mid
        m3 = torch.nonzero(new_s[m2, 3]>=0, as_tuple=True)[0]
        if m3.shape[0] > 0:
            new_modes[m2[m3]] = 3.0

        if check_invalid:
            m2_check = torch.nonzero(new_s[m2, 2]<0, as_tuple=True)[0]
            if m2_check.shape[0]>0:
                new_modes[m2[m2_check]] = -1.0

    # mode 4: stance_tension
    m4 = torch.nonzero(torch.logical_or(modes==3, modes==4), as_tuple=True)[0]
    if m4.shape[0] >0:
        new_modes[m4] = 4.0
        new_s[m4] = stance_f(s[m4], u[m4],global_vars,is_compression=False, same_u=same_u)

        # mode 5: event release
        m5 = torch.nonzero((new_s[m4, 0] - new_s[m4, 4]) * (new_s[m4, 0] - new_s[m4, 4])
                           + new_s[m4, 2] * new_s[m4, 2] >= l0*l0, as_tuple=True)[0]
        if m5.shape[0] > 0:
            new_modes[m4[m5]] = 5.0

        if check_invalid:
            m4_check = torch.nonzero(new_s[m4, 2]<0, as_tuple=True)[0]
            if m4_check.shape[0]>0:
                new_modes[m4[m4_check]] = -1.0

    # mode 6: flight (ground->apex)
    m6 = torch.nonzero(torch.logical_or(modes==5, modes==6), as_tuple=True)[0]
    if m6.shape[0] > 0:
        new_s[m6] = flight_f(s[m6], u[m6], global_vars)
        new_modes[m6] = 6.0

        # mode 7: event apex
        m7 = torch.nonzero(new_s[m6, 3]<=0, as_tuple=True)[0]
        if m7.shape[0] > 0:
            new_modes[m6[m7]] = 7.0

    # mode -1: invalid (after event apex)
    m_invalid = torch.nonzero(modes==7.0, as_tuple=True)[0]
    if m_invalid.shape[0] > 0:
        new_modes[m_invalid] = 7.0

    return new_s, new_modes


# TODO (temp) need to check whether they are same
def pytorch_sim_single(s, modes, u, global_vars, same_u=False, check_invalid=False):
    num_steps, dt, dt2, l0, g, k, M = global_vars
    new_modes = torch.zeros_like(modes)

    # mode 0: flight (from apex->ground)
    if modes[0, 0] == 0:
        new_s = flight_f(s, u, global_vars)
        # mode 1: event contact ()
        if new_s[0, 2]-l0 * torch.cos(u[0, 0])<=0:
            new_modes = torch.ones_like(modes)
            new_s[0, 4] = new_s[0, 4] * 0.0 + new_s[0, 0] + l0 * torch.sin(u[0, 0])
            new_s[0, 5] = new_s[0, 5] * 0.0

    # mode 2: stance_compression
    elif modes[0, 0] == 1 or modes[0, 0] == 2:
        new_s = stance_f(s, u, global_vars, is_compression=True, same_u=same_u)
        new_modes = torch.ones_like(modes) * 2.0
        # mode 3: event stance_mid
        if new_s[0, 3] >= 0:
            new_modes = torch.ones_like(modes) * 3.0

        if check_invalid:
            if new_s[0, 2] < 0:
                new_modes = torch.ones_like(modes) * -1.0

    # mode 4: stance_tension
    elif modes[0, 0] == 3 or modes[0, 0] == 4:
        new_modes = torch.ones_like(modes) * 4.0
        new_s = stance_f(s, u, global_vars, is_compression=False, same_u=same_u)
        # mode 5: event release
        if (new_s[0, 0] - new_s[0, 4]) * (new_s[0, 0] - new_s[0, 4]) + new_s[0, 2] * new_s[0, 2] >= l0*l0:
            new_modes = torch.ones_like(modes) * 5.0
        if check_invalid:
            if new_s[0, 2]<0:
                new_modes = torch.ones_like(modes) * -1.0

    # mode 6: flight (ground->apex)
    elif modes[0, 0] == 5 or modes[0, 0] == 6:
        new_s = flight_f(s, u, global_vars)
        new_modes = torch.ones_like(modes) * 6.0
        # mode 7: event apex
        if new_s[0, 3] <= 0:
            new_modes = torch.ones_like(modes) * 7.0

    elif modes[0,0] == 7.0:
        new_modes = torch.ones_like(modes) * 7.0

    else:
        raise NotImplementedError

    return new_s, new_modes


def visualization(s_list, m_list):
    xs=[]
    xds=[]
    ys=[]
    yds=[]
    for ti in range(len(s_list)):
        s = s_list[ti]
        # for i in range(s.shape[0]):
        i=0

        if m_list[ti][i]!=7 and ti%1==0:
            print("ti=%03d mode=%d x :%.4f xd:%.4f y :%.4f yd:%.4f xF:%.4f yF:%.4f"%(
            ti, m_list[ti][i], s[i,0], s[i,1], s[i,2],s[i,3], s[i,4],s[i,5]))
            xs.append(s[i, 0])
            xds.append(s[i, 1])
            ys.append(s[i, 2])
            yds.append(s[i, 3])

    return np.array([xs, xds, ys, yds]).T


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.relu = nn.ReLU()
        self.args = args

        self.linear_list = nn.ModuleList()
        input_dim = 5  # x,y,th,v
        output_dim = 2  # omega, accel
        self.linear_list.append(nn.Linear(input_dim, args.hiddens[0]))
        for i, hidden in enumerate(args.hiddens):
            if i == len(args.hiddens) - 1:  # last layer
                self.linear_list.append(nn.Linear(args.hiddens[i], output_dim))
            else:  # middle layers
                self.linear_list.append(nn.Linear(args.hiddens[i], args.hiddens[i + 1]))

    def forward(self, x):
        for i, hidden in enumerate(self.args.hiddens):
            x = self.relu(self.linear_list[i](x))
        x = self.linear_list[len(self.args.hiddens)](x)
        if self.args.output_type == "tanh":
            x = nn.Tanh()(x) * self.args.tanh_gain
        return x


class CLF(nn.Module):
    def __init__(self, args):
        super(CLF, self).__init__()
        self.relu = nn.ReLU()
        self.args = args

        self.linear_list = nn.ModuleList()
        input_dim = 3
        output_dim = 1
        self.tmp_list=[input_dim] + args.clf_hiddens + [output_dim]
        for i in range(len(self.tmp_list)-1):
            self.linear_list.append(nn.Linear(self.tmp_list[i], self.tmp_list[i+1]))

    def forward(self, x):
        for i, fc in enumerate(self.linear_list):
            x = fc(x)
            if i != len(self.linear_list)-1:
                x = self.relu(x)
        return x


def get_mocked_samples(limit_cycle):
    # input: limit_cycle in (N, 4)
    # return: dist samples in (N, 3+1=4)
    # we use last 3 dims
    # construct zero level samples
    # construct distance samples by nearest neighbor
    xs = []
    ds = []
    N = limit_cycle.shape[0]

    refs = limit_cycle[:, 0, 1:4]

    # zero-level points
    for pts in refs:
        xs.append(pts)
        ds.append(0)

    # random points
    mins=np.min(refs, axis=0)
    maxs=np.max(refs, axis=0)
    dmm = (maxs-mins)*args.minmax_margin
    x_w = maxs - mins

    mins, maxs = mins - dmm, maxs + dmm

    if args.grid_train:
        n_grids = args.n_grids
        N1 = n_grids**3
        x0s = np.linspace(mins[0], maxs[0], n_grids)
        x1s = np.linspace(mins[1], maxs[1], n_grids)
        x2s = np.linspace(mins[2], maxs[2], n_grids)

        # reduce 0 or 1 dimensions
        rands = np.stack(np.meshgrid(x0s, x1s, x2s, indexing='ij'), axis=-1)  # (n_grids, n_grids, n_grids, 3)
        # x012s = x012s.reshape((n_grids, n_grids, n_grids, 1, 3))
        rands = rands.reshape((N1, 3))

    else:
        if args.n_aug==-1:
            N1 = N
        else:
            N1 = args.n_aug
        rands = np.random.rand(N1, 3)
        rands = rands * (maxs - mins) + mins

    dists = np.sqrt(np.min(np.sum(
        ((rands.reshape((N1, 1, 3)) - refs.reshape((1, N ,3)))/x_w
         )**2, axis=-1), axis=1))

    for pts, d in zip(rands, dists):
        xs.append(pts)
        ds.append(d)

    xs = np.stack(xs, axis=0)
    ds = np.stack(ds, axis=0)
    ds = ds.reshape((ds.shape[0], 1))

    return xs, ds


def main0(global_vars):
    num_steps, dt, dt2, l0, g, k, M = global_vars

    N = 1
    # initialize state and control input
    s = torch.zeros(N, 6)  # (x, xd, y, yd, xf, yf)
    s[:, 1] = 3.0  #3.951056516295154  #3.0
    s[:, 2] = 1.3  #1.392705144665068  #2.0
    s[:, 5] = 1.0

    # TODO use the limit cycle control, to check consistency of simulation vs ODE
    modes = torch.zeros((N,))

    u = torch.zeros((N, 3))
    u[:, 0] = 0.23004320  #0.324894  # TODO (the theta)
    u[:, 1] = 0  #1.3949917*1000

    s_list=[]
    m_list=[]

    # simulation  # TODO (multiple intervals)
    for ti in range(num_steps):
        next_s, next_modes = pytorch_sim(s, modes, u, global_vars)
        s_list.append(s.detach().cpu().numpy())
        m_list.append(modes.detach().cpu().numpy())
        s, modes = next_s, next_modes

    visualization(s_list, m_list)


def find_limit_cycle(apex_vx, apex_h, theta, global_vars):
    num_steps, dt, dt2, l0, g, k, M = global_vars

    s = torch.zeros(1, 6)  # (x, xd, y, yd, xf, yf)
    s[:, 1] = apex_vx  # 3.951056516295154  #3.0
    s[:, 2] = apex_h  # 1.392705144665068  #2.0
    s[:, 5] = apex_vx - 1
    modes = torch.zeros((1,))
    u = torch.zeros((1, 3))
    u[0] = theta  # TODO auto-compute
    s_list = []
    m_list = []
    # simulation
    for ti in range(num_steps):
        next_s, next_modes = pytorch_sim(s, modes, u, global_vars)
        s_list.append(s.detach().cpu().numpy())
        m_list.append(modes.detach().cpu().numpy())
        s, modes = next_s, next_modes

    limit_cycle = []
    for i, mode_i in enumerate(m_list):
        if mode_i != 7:
            limit_cycle.append(s_list[i])
    return np.stack(limit_cycle, axis=0)

def get_clf_loss(est, gt):
    if args.log_loss:
        return torch.mean(torch.square(torch.log(torch.clamp(est, min=1e-7)) - torch.log(torch.clamp(gt, min=1e-7))))
    elif args.rel_loss:
        return torch.mean(torch.abs((est-gt)/torch.clamp(gt, min=1e-4)))
    elif args.l1_loss:
        return torch.mean(torch.abs(est - gt))
    elif args.l2_loss:
        return torch.sqrt(torch.mean(torch.square(est - gt)))
    else:
        return torch.mean(torch.square(est - gt))


def get_der_loss(v_k, v_k1):
    return torch.mean(torch.maximum(v_k - v_k1, 0.2 * torch.abs(v_k1)))


def get_pos_loss(v_k):
    return torch.mean(torch.maximum(v_k, 1e-4 * torch.ones_like(v_k)))


def get_grid_samples(limit_cycle):
    if os.path.exists("collected_data.npz"):
        data=np.load("collected_data.npz", allow_pickle=True)
        col_x = data["x"]
        col_d = data["d"]
        return col_x, col_d


    xs = []
    ds = []
    N = limit_cycle.shape[0]

    refs = limit_cycle[:, 0, 1:4]

    # zero-level points
    for pts in refs:
        xs.append(pts)
        ds.append(0)

    # random points
    mins=np.min(refs, axis=0)
    maxs=np.max(refs, axis=0)
    dmm = (maxs-mins) * args.minmax_margin
    mins, maxs = mins - dmm, maxs + dmm
    x_w = maxs-mins
    n_grids = args.n_grids

    x0s = np.linspace(mins[0], maxs[0], n_grids)
    x1s = np.linspace(mins[1], maxs[1], n_grids)
    x2s = np.linspace(mins[2], maxs[2], n_grids)

    # reduce 0 or 1 dimensions
    x012s = np.stack(np.meshgrid(x0s, x1s, x2s, indexing='ij'), axis=-1)  # (n_grids, n_grids, n_grids, 3)
    x012s = x012s.reshape((n_grids, n_grids, n_grids, 1, 3))
    collect_x1 = np.zeros((n_grids, n_grids, 3))
    collect_d1 = np.zeros((n_grids, n_grids))
    collect_x2 = np.zeros((n_grids, n_grids, 3))
    collect_d2 = np.zeros((n_grids, n_grids))

    refs = refs.reshape((1, N, 3))

    for i in range(n_grids):
        for j in range(n_grids):
            dist=np.sqrt(np.sum(np.square((x012s[:, i, j]-refs)/x_w), axis=-1))  # shape(n_grids, N)
            min_dist=np.min(dist, axis=-1)
            min_idx = np.argmin(min_dist)
            collect_x1[i, j, :] = x012s[min_idx, i, j]
            collect_d1[i, j] = min_dist[min_idx]

    for i in range(n_grids):
        for j in range(n_grids):
            dist=np.sqrt(np.sum(np.square((x012s[i, :, j]-refs)/x_w), axis=-1))  # shape(n_grids, N)
            min_dist=np.min(dist, axis=-1)
            min_idx = np.argmin(min_dist)
            collect_x2[i, j, :] = x012s[i, min_idx, j]
            collect_d2[i, j] = min_dist[min_idx]

    col_x = np.concatenate((collect_x1.reshape((n_grids*n_grids, 3)), collect_x2.reshape((n_grids*n_grids, 3))), axis=0)
    col_d = np.concatenate((collect_d1.reshape((n_grids*n_grids, 1)), collect_d2.reshape((n_grids*n_grids, 1))), axis=0)
    np.savez("collected_data.npz", x=col_x, d=col_d)

    return col_x, col_d


def main():
    # TODO setup exp dir
    np.random.seed(args.random_seed)
    torch.manual_seed(1007)
    utils.setup_data_exp_and_logger(args)

    num_steps = 3000
    # dt = 0.001
    # dt2 = 0.1 * dt
    dt = 0.001
    dt2 = 0.1 * dt
    l0 = 1.0
    g = 10.0
    k = 32000
    M = 80

    global_vars = num_steps, dt, dt2, l0, g, k, M

    limit_cycle = find_limit_cycle(apex_vx=3.0, apex_h=1.3, theta=0.23004320, global_vars=global_vars)

    if args.grid_test:
        grid_x, grid_d = get_grid_samples(limit_cycle)

    if args.use_test_train:
        xs, ds = grid_x, grid_d
    else:
        xs, ds = get_mocked_samples(limit_cycle)

    limit_cycle = limit_cycle[:, 0, :]

    # shuffle train/val datasets
    xs = torch.from_numpy(xs).float()
    ds = torch.from_numpy(ds).float()
    if args.grid_test:
        grid_x = torch.from_numpy(grid_x).float()
        grid_d = torch.from_numpy(grid_d).float()
    rand_idx = torch.randperm(xs.shape[0])
    split = int(xs.shape[0] * 0.8)

    train_xs=xs[rand_idx[:split]]
    train_ys=ds[rand_idx[:split]]
    test_xs=xs[rand_idx[split:]]
    test_ys=ds[rand_idx[split:]]
    policy = Policy(args)
    clf = CLF(args)

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        train_xs = train_xs.cuda()
        train_ys = train_ys.cuda()
        test_xs = test_xs.cuda()
        test_ys = test_ys.cuda()
        policy = policy.cuda()
        clf = clf.cuda()
        if args.grid_test:
            grid_x = grid_x.cuda()
            grid_d = grid_d.cuda()

    optimizer = torch.optim.SGD(clf.parameters(), args.lr)

    # monitors
    train_losses, train_clf_losses, train_der_losses, train_pos_losses,\
    test_losses, test_clf_losses, test_der_losses, test_pos_losses, test_errors = utils.get_n_meters(8+1)

    # train CLF with those points
    for epi in range(args.epochs):
        # simulation
        # TODO(change later)
        tr_sim = train_xs * 0.0
        tr_s = clf(tr_sim)

        train_s = clf(train_xs)
        # TODO diff weights for =0 \!= 0 samples
        train_clf_loss = get_clf_loss(train_s, train_ys) * args.clf_weight
        train_der_loss = get_der_loss(tr_s, tr_s) * args.der_weight
        train_pos_loss = get_pos_loss(tr_s) * args.pos_weight
        train_loss = train_clf_loss + train_der_loss + train_pos_loss

        train_loss.backward()
        # torch.nn.utils.clip_grad_norm_(params, 1e-6)
        optimizer.step()
        optimizer.zero_grad()

        test_s = clf(test_xs)
        test_clf_loss = get_clf_loss(test_s, test_ys) * args.clf_weight
        test_der_loss = get_der_loss(tr_s, tr_s) * args.der_weight
        test_pos_loss = get_pos_loss(tr_s) * args.pos_weight
        test_loss = test_clf_loss + test_der_loss + test_pos_loss

        train_losses.update(train_loss.detach().cpu().item())
        train_clf_losses.update(train_clf_loss.detach().cpu().item())
        train_der_losses.update(train_der_loss.detach().cpu().item())
        train_pos_losses.update(train_pos_loss.detach().cpu().item())
        test_losses.update(test_loss.detach().cpu().item())
        test_clf_losses.update(test_clf_loss.detach().cpu().item())
        test_der_losses.update(test_der_loss.detach().cpu().item())
        test_pos_losses.update(test_pos_loss.detach().cpu().item())

        if args.grid_test:
            grid_est = clf(grid_x)
            test_error = get_clf_loss(grid_est, grid_d)
            test_errors.update(test_error.detach().cpu().item())
        else:
            test_errors.update(0)

        if epi % args.print_freq == 0:
            print("[%05d/%05d] train:%.4f(%.4f) c:%.4f d:%.4f p:%.4f  test:%.5f(%.5f) c:%.4f d:%.4f p:%.4f | err:%.4f(%.4f)"%
                  (epi, args.epochs, train_losses.val, train_losses.avg, train_clf_losses.val,
                   train_der_losses.val, train_pos_losses.val,
                   test_losses.val, test_losses.avg,
                   test_clf_losses.val, test_der_losses.val, test_pos_losses.val, test_errors.val, test_errors.avg))

            if epi == 0:
                plt_cycle(limit_cycle, "%s/fig1.png" % args.viz_dir)


        if epi % args.viz_freq == 0:
            if args.grid_test:
                img_est = grid_est.detach().cpu().numpy().reshape((2, args.n_grids, args.n_grids))
                img_gt = grid_d.detach().cpu().numpy().reshape((2, args.n_grids, args.n_grids))
                plt_heat(img_est[0], "%s/f_est0_e%04d.png"%(args.viz_dir, epi))
                plt_heat(img_est[1], "%s/f_est1_e%04d.png"%(args.viz_dir, epi))
                if epi==0:
                    plt_heat(img_gt[0], "%s/f_est0_gt.png"%(args.viz_dir))
                    plt_heat(img_gt[1], "%s/f_est1_gt.png"%(args.viz_dir))

        if epi % args.save_freq == 0:
            torch.save(clf.state_dict(), "%s/clf_model_e%05d.ckpt"%(args.model_dir, epi))


def plt_cycle(limit_cycle, fig_name):
    ax = plt.subplot(221, projection='3d')
    ax.scatter(limit_cycle[:, 1], limit_cycle[:, 2], limit_cycle[:, 3])
    ax.set_xlabel("ydot (m/s)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel('zdot (m/s)')

    plt.subplot(222)
    plt.scatter(limit_cycle[:, 1], limit_cycle[:, 2])
    plt.xlabel("ydot (m/s)")
    plt.ylabel("z (m)")

    plt.subplot(223)
    plt.scatter(limit_cycle[:, 2], limit_cycle[:, 3])
    plt.xlabel("z (m)")
    plt.ylabel("zdot (m/s)")

    plt.subplot(224)
    plt.scatter(limit_cycle[:, 1], limit_cycle[:, 3])
    plt.xlabel("ydot (m/s)")
    plt.ylabel("zdot (m/s)")

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

def plt_heat(data, fig_name):
    plt.imshow(data)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

# TODO (Quadratic Programming Test)
def qp_test():
    np.random.seed(args.random_seed)
    torch.manual_seed(1007)
    utils.setup_data_exp_and_logger(args)
    clf = CLF(args)
    clf.load_state_dict(torch.load(args.clf_pretrained_path))
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        clf = clf.cuda()

    nt = 3000
    dt = 0.01
    dt2 = 0.1 * dt
    viz_freq=50
    counter=0
    var_counter=0

    num_steps = 3000
    # dt = 0.001
    # dt2 = 0.1 * dt
    dt = 0.001
    dt2 = 0.1 * dt
    l0 = 1.0
    g = 10.0
    k = 32000
    M = 80

    global_vars = num_steps, dt, dt2, l0, g, k, M


    # limit cycle apex point  # (apex_vx=3.0, apex_h=1.3, theta=0.23004320)
    # default theta
    theta = 0.324894128831083  #0.324894128831083 #0.23004320
    # theta = 0.23004320


    # theta_list=[0.3208, 0.2546, 0.2422, 0.2361, 0.2327]
    # initial point (x, xd, y, yd, th)
    s = torch.zeros(1, 6)

    s[0, 1] = 3.951056516295154  # 3.0
    s[0, 2] = 1.392705144665068  # 1.3
    s[0, 4] = s[0, 2] - 1.0
    s_list=[]
    u_list=[]
    m_list=[]
    v_list=[]
    n_loops=4

    for ni in range(n_loops):
        triggered = False

        u = torch.zeros(1, 2)
        mode = torch.zeros(1, 1)

        if ni>0:
            s = new_s

        # qp process
        for ti in range(nt):
            if mode==-1:
                break

            s_sub = s[:, 1:4]
            v = clf(s_sub)


            if args.use_gt_u:
                # GT_THETA=0.324894128831083
                # GT_F_C=1.394991746366024e+03
                # GT_F_T=0
                '''
                u:0.3208 934.8398 0.0000  V: 1.0000 -> 0.2500
                u:0.2546 509.3090 0.0000  V: 0.2500 -> 0.0625
                u:0.2422 254.7015 0.0000  V: 0.0625 -> 0.0156
                u:0.2361 127.4664 0.0000  V: 0.0156 -> 0.0039
                u:0.2327 0.0020 0.0000  V: 0.0039 -> 0.0039
                '''
                GT_THETAS = [0.3208, 0.2546, 0.2422, 0.2361, 0.2327]
                GT_F_CS=[934.8398, 509.3090, 254.7015, 127.4664, 0.0020]
                GT_F_TS=[0,0,0,0,0]
                GT_THETA=GT_THETAS[ni]
                GT_F_C = GT_F_CS[ni]
                GT_F_T=GT_F_TS[ni]


                u = torch.zeros(1, 2)
                u[0, 0] = GT_THETA
                if mode==2 or mode==3 or mode==4:
                    if mode==2:
                        u[0, 1] = GT_F_C
                    else:
                        u[0, 1] = GT_F_T

            else:
                if triggered or (args.neg_clf_correct and v<0):
                    triggered=True
                    u[0,0] = 0.23004320
                    u[0,1] = 0.0
                else:
                    if mode==2 or mode==3 or mode==4:  # stance mode
                        u = qp_solve(s, mode, clf, args)
                    else:  # flight mode
                        u[0,0] = theta
                        u[0,1] = 0.0

            new_mode = torch.zeros_like(mode)
            if mode==0:
                new_s = flight_f(s, u, global_vars)
                if new_s[0, 2] - l0 * torch.cos(u[0, 0]) <= 0:
                    new_mode[0, 0] = 1.0
                    new_s[0, 4] = new_s[0, 0] + l0 * torch.sin(u[0, 0])
                    new_s[0, 5] = 0.0
            elif mode==1 or mode==2:
                new_s = stance_f(s, u, is_compression=True)
                new_mode[0, 0] = 2.0
                if new_s[0, 3]>=0:
                    new_mode[0, 0] = 3.0
            elif mode==3 or mode==4:
                new_s = stance_f(s, u, is_compression=True)
                new_mode[0, 0] = 4.0
                if (new_s[0,0] - new_s[0,4])*(new_s[0,0]-new_s[0,4])+new_s[0,2]*new_s[0,2]>=l0*l0:
                    new_mode[0, 0] = 5.0
            elif mode==5 or mode==6:
                new_s = flight_f(s, u, global_vars)
                new_mode[0, 0] = 6.0
                if new_s[0, 3]<=0:
                    new_mode[0, 0] = 7.0
            elif mode==7 or mode==-1.0:
                new_s = s
                new_mode[0,0] = -1.0

            s_list.append(s)
            u_list.append(u)
            m_list.append(mode)
            v_list.append(v)



            #if mode==7:#!=-1.0:
            if mode!=-1.0:
                print("t %05d [%5s], mode=%d  F:%.4f  V:%.4f state(%.3f %.3f %.3f %.3f %.3f %.3f)" % (ti, triggered, mode, u[0,1], v,
                                                                                    s[0,0],
                                                                                    s[0,1],
                                                                                    s[0,2],
                                                                                    s[0,3],
                                                                                    s[0,4],
                                                                                    s[0,5]))


            if var_counter % viz_freq == 0:
                plt.figure(figsize=(12,5))
                plt.rcParams.update({'font.size': 16})
                plt.subplot(1, 2, 1)
                plt.plot([s[0, 0], s[0, 4]], [s[0, 2], s[0, 5]], linewidth=2)
                plt.plot(s[0, 0], s[0, 2], 'ro', markersize=4)
                plt.plot(s[0, 4], s[0, 5], 'ro', markersize=4)
                # plt.xlim(dx + head[0] - 1.5, dx + head[0] + 1.5)

                plt.text(5, 1.0, "mode=%d\ny :%.4f\nyd:%.4f (3.00)\nz :%.4f (1.30)\nzd:%.4f\nyF:%.4f\nzF:%.4f"%(
                    mode, s[0,0], s[0,1], s[0,2],s[0,3], s[0,4],s[0,5]
                ))

                plt.xlim(-1, 10.0)
                plt.ylim(-0.5, 2.5)
                plt.xlabel("y (m)")
                plt.ylabel("z (m)")
                # ax=plt.gca()
                # ax.set_aspect(3.0, adjustable='box')

                plt.subplot(1, 2, 2)
                plt.plot(range(counter+1), v_list)
                plt.xlabel("Time step")
                plt.ylabel("Lyapunov value")
                plt.xlim(0,9000)
                plt.tight_layout()
                plt.savefig("%s/t%05d.png" % (args.viz_dir, var_counter), pad_inches=0.5)
                plt.close()
            if mode==2 or mode==3 or mode==4:
                var_counter+=1
            else:
                var_counter+=1
            counter+=1
            s = new_s
            mode = new_mode

    # # plot (Bot ani + Lya value)
    # for

    return

def qp_solve(s, mode, clf, args, global_vars):
    num_steps, dt, dt2, l0, g, k, M = global_vars

    alpha = 0.5

    # derive dv/ds = NN(x) autograd
    s_sub = s[:, 1:4]
    s_sub.requires_grad = True
    v = clf(s_sub)
    # print(s_sub.shape, v.shape)
    dv_ds = torch.autograd.grad(outputs=v, inputs=s_sub, grad_outputs=torch.ones_like(v), retain_graph=True)[0]
    # print(dv_ds.shape)

    # derive ds/dt = f(s)+g(s)u
    l = torch.sqrt((s[0,0]-s[0,4])*(s[0,0]-s[0,4]) + s[0,2]*s[0,2])
    fac_x = 1 / M / l * (s[0,0]-s[0,4])
    fac_y = 1 / M / l * (s[0,2]-0)

    f_s = torch.zeros(3, 1)
    f_s[0, 0] = fac_x * (k * (l0-l))
    f_s[1, 0] = s[0, 3]
    f_s[2, 0] = fac_y * (k * (l0-l)) - g

    g_s = torch.zeros(3, 1)
    g_s[0, 0] = fac_x
    g_s[2, 0] = fac_y

    # get result
    P = np.ones((1,1))
    q = np.zeros((1,))

    G = torch.matmul(dv_ds, g_s).detach().cpu().numpy()
    h = (torch.matmul(- dv_ds, f_s) - alpha * v).detach().cpu().numpy()
    h = h.flatten()

    if v<0:
        G = np.zeros_like(G)
        h = np.zeros_like(h)

    A = np.zeros((1, 1))
    b = np.zeros((1,))

    F_min=0  #-5000
    F_max=5000  #5000
    F_min = 0  # -5000
    F_max = 500000  # 5000

    lb = np.ones((1,)) * F_min
    ub = np.ones((1,)) * F_max

    # print(P.shape, q.shape, G.shape, h.shape, A.shape, b.shape)

    x = solve_qp(P, q, G, h, A, b, solver=args.solver, verbose=args.verbose)

    # print(x.shape)
    # print(x)
    u_all = torch.zeros(1, 2)
    # u_all[0, 1] = x.item()
    u_all[0, 1] = np.clip(x.item(), F_min, F_max)
    return u_all


if __name__ == "__main__":
    # TODO CONSTANTS
    # num_steps = 3000
    # # dt = 0.001
    # # dt2 = 0.1 * dt
    # dt = 0.001
    # dt2 = 0.1 * dt
    # l0 = 1.0
    # g = 10.0
    # k = 32000
    # M = 80

    tt1=time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="debug")
    parser.add_argument('--random_seed', type=int, default=500)
    parser.add_argument('--hiddens', type=int, nargs="+", default=[16, 16])
    parser.add_argument('--clf_hiddens', type=int, nargs="+", default=[16, 16])
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--nt', type=int, default=500)
    parser.add_argument('--num_samples', type=int, default=20)

    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--viz_freq', type=int, default=500)
    parser.add_argument('--save_freq', type=int, default=2000)

    parser.add_argument('--train_policy', action='store_true')

    parser.add_argument('--clf_weight', type=float, default=1.0)
    parser.add_argument('--der_weight', type=float, default=1.0)
    parser.add_argument('--pos_weight', type=float, default=1.0)

    parser.add_argument('--grid_train', action='store_true')
    parser.add_argument('--grid_test', action='store_true')
    parser.add_argument('--minmax_margin', type=float, default=0.33)
    parser.add_argument('--n_grids', type=int, default=41)
    parser.add_argument('--use_test_train', action='store_true')

    parser.add_argument('--log_loss', action='store_true')
    parser.add_argument('--rel_loss', action='store_true')
    parser.add_argument('--l1_loss', action='store_true')
    parser.add_argument('--l2_loss', action='store_true')
    parser.add_argument('--n_aug', type=int, default=-1)

    parser.add_argument('--qp_test', action='store_true')
    parser.add_argument('--clf_pretrained_path', type=str, default=None)
    parser.add_argument('--solver', type=str, default='quadprog')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use_gt_u', action='store_true')
    parser.add_argument('--neg_clf_correct', action='store_true')
    args = parser.parse_args()
    if args.qp_test:
        qp_test()
    else:
        main()

    tt2 = time.time()
    print("Finished in %.4f seconds"%(tt2-tt1))