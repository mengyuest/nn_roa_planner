import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import argparse

sys.path.append("../")
import roa_utils as utils

# TODO
class Dyna(nn.Module):
    def __init__(self, args):
        super(Dyna, self).__init__()
        self.relu = nn.ReLU()
        self.args = args

        self.linear_list = nn.ModuleList()
        input_dim = 4
        output_dim = 3
        self.tmp_list = [input_dim] + args.hiddens + [output_dim]

        for i in range(len(self.tmp_list)-1):
            self.linear_list.append(nn.Linear(self.tmp_list[i], self.tmp_list[i+1]))

        if self.args.normalize:
            self.in_means = self.args.in_means.float()
            self.in_stds = self.args.in_stds.float()
            self.out_means = self.args.out_means.float()
            self.out_stds = self.args.out_stds.float()

    def update_device(self):
        if self.args.normalize:
            self.in_means = self.in_means.cuda()  # .to(device)
            self.in_stds = self.in_stds.cuda()  # .to(device)
            self.out_means = self.out_means.cuda()  # .to(device)
            self.out_stds = self.out_stds.cuda()  # .to(device)

    def forward(self, ori_x):
        if self.args.normalize:
            x = (ori_x - self.in_means) / self.in_stds
        else:
            x = ori_x
        for i, fc in enumerate(self.linear_list):
            x = fc(x)
            if i != len(self.linear_list) - 1:
                x = self.relu(x)
        if self.args.normalize:
            out_x = x * self.out_stds + self.out_means
        else:
            out_x = x
        return out_x


def normalize_mean_std(x, y, use_mean=None, use_std=None):
    x_mean = torch.mean(x, axis=0)
    x_std = torch.std(x, axis=0)
    return x_mean, x_std


def examine_stat(x, name="debug"):
    print("%12s %.3f %.3f %.3f %.3f"%(name, torch.min(x), torch.max(x), torch.mean(x), torch.std(x)))


def examine_stat_list(x, name_list="debug"):
    for i in range(len(name_list)):
        examine_stat(x[:, i], name_list[i])


def preproc_data(data_path, split_ratio=0.8):
    data = np.load(data_path, allow_pickle=True)
    indices = np.where(data["modes"] == 1.0)
    xs = torch.from_numpy(data["xs"][indices]).float()
    us = torch.from_numpy(data["us"][indices]).float()
    ys = torch.from_numpy(data["new_xs"][indices]).float()

    xs_us = torch.cat([xs[:, 1:3], us], dim=-1)
    ys = ys[:, :3]
    split = int(xs.shape[0] * split_ratio)
    train_x_u, test_x_u = utils.split_tensor(xs_us, split)
    train_y, test_y = utils.split_tensor(ys, split)

    print("train:%d test:%d"%(train_x_u.shape[0], test_x_u.shape[0]))

    x_u_mean, x_u_std = normalize_mean_std(train_x_u, test_x_u)
    y_mean, y_std = normalize_mean_std(train_y, test_y)

    examine_stat_list(train_x_u, ["train xd", "train y", "train th", "train f"])
    examine_stat_list(train_y, ["train new x", "train new xd", "train new y"])
    examine_stat_list(test_x_u, ["test xd", "test y", "test th", "test f"])
    examine_stat_list(test_y, ["test new x", "test new xd", "test new y"])

    return train_x_u, test_x_u, x_u_mean, x_u_std, train_y, test_y, y_mean, y_std


def cal_loss(pred, y, means, std):
    rel_err = (pred-y)/std
    return torch.mean(torch.square(rel_err))


def main():
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    utils.setup_data_exp_and_logger(args, offset=1)

    data_path = utils.smart_path(args.data_path) + "/data.npz"
    train_x_u, test_x_u, x_u_mean, x_u_std, train_y, test_y, y_mean, y_std = preproc_data(data_path, args.split)

    args.in_means = x_u_mean
    args.in_stds = x_u_std
    args.out_means = y_mean
    args.out_stds = y_std

    net = Dyna(args)

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        train_x_u = train_x_u.cuda()
        train_y = train_y.cuda()
        test_x_u = test_x_u.cuda()
        test_y = test_y.cuda()
        net = net.cuda()
        net.update_device()

    optimizer = torch.optim.SGD(net.parameters(), args.lr)

    train_losses, test_losses = utils.get_n_meters(2)
    np.savez("%s/data_mean_std.npz" % args.model_dir, x_u_mean=x_u_mean, x_u_std=x_u_std, y_mean=y_mean, y_std=y_std)
    for epi in range(args.epochs):
        if args.batch_size is not None:
            if epi % args.print_freq == 0:
                rand_idx = torch.randperm(train_x_u.shape[0])
                sub_train_x_u = train_x_u[rand_idx[:args.batch_size]]
                sub_train_y = train_y[rand_idx[:args.batch_size]]
        else:
            sub_train_x_u = train_x_u
            sub_train_y = train_y

        sub_train_pred = net(sub_train_x_u)

        train_loss = cal_loss(sub_train_pred, sub_train_y, None, net.out_stds)

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        test_pred = net(test_x_u)
        test_loss = cal_loss(test_pred, test_y, None, net.out_stds)
        train_losses.update(train_loss.detach().cpu().item())
        test_losses.update(test_loss.detach().cpu().item())
        if epi % args.print_freq == 0:
            print("[%05d/%05d] train:%.6f(%.6f) test:%.6f(%.6f)" %
                (epi, args.epochs, train_losses.val, train_losses.avg, test_losses.val, test_losses.avg))

        if epi % args.save_freq == 0:
            torch.save(net.state_dict(), "%s/model_e%05d.ckpt"%(args.model_dir, epi))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="beta_fit_dbg")
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--hiddens', type=int, nargs="+", default=[16, 16])
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
    parser.add_argument('--normalize', action="store_true", default=False)
    parser.add_argument('--split', type=float, default=0.8)

    args = parser.parse_args()
    tt1 = time.time()
    main()
    tt2 = time.time()

    print("Finished in %.4f seconds" % (tt2 - tt1))
