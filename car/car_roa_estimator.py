import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
import tqdm

import car_clf_train as za
import car_utils as sscar_utils

sys.path.append("../")
import roa_utils as utils
import roa_nn


class MockArgs(object):
    pass


def main():
    utils.set_random_seed(args.random_seed)
    utils.setup_data_exp_and_logger(args, offset=1)
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    net = roa_nn.Net(args)
    utils.safe_load_nn(net, args.net_pretrained_path, load_last=args.load_last, key="net_", model_iter=args.model_iter, mode="car")
    if args.gpus is not None:
        net = net.cuda()

    if args.load_data_path is not None:
        levelset_data = np.load(args.load_data_path)['data']
        levelset_data = torch.from_numpy(levelset_data).float()
    else:
        clf = roa_nn.CLF(None, args)
        actor = roa_nn.Actor(None, args)
        if args.pret:
            utils.safe_load_nn(clf, args.clf_pretrained_path, load_last=args.load_last, key="clf_")
        else:
            utils.safe_load_nn(clf, args.clf_pretrained_path, load_last=args.load_last, key="clf_", model_iter=args.model_iter, mode="car")
        utils.safe_load_nn(actor, args.actor_pretrained_path, load_last=args.load_last, key="actor_", model_iter=args.model_iter, mode="car", pret=args.pret)
        if args.gpus is not None:
            clf = clf.cuda()
            actor = actor.cuda()

        # generate configs
        mock_args = MockArgs
        mock_args.s_mins = args.s_mins
        mock_args.s_maxs = args.s_maxs
        mock_args.mu_choices = args.mu_choices
        mock_args.ref_mins = args.ref_mins
        mock_args.ref_maxs = args.ref_maxs
        mock_args.num_samples = args.num_samples
        init_s, params = za.init_sampling(mock_args)

        if args.grid_params:
            n_vref = 11
            n_wref = 11
            cnt = 0
            input_list=[]
            for i in range(len(args.mu_choices)):
                mu = args.mu_choices[i]
                v_ref_list = np.linspace(args.ref_mins[0], args.ref_maxs[0], n_vref)
                w_ref_list = np.linspace(args.ref_mins[1], args.ref_maxs[1], n_wref)
                for j in range(n_vref):
                    for k in range(n_wref):
                        v_ref = v_ref_list[j]
                        w_ref = w_ref_list[k]
                        input_list.append([mu, v_ref, w_ref, cnt])
                        cnt += 1
            params = torch.tensor(input_list).float()

        # generate samples
        # lqr_P, lqr_K = za.solve_for_P_K(init_cfg, car_params, args)
        if args.gpus is not None:
            init_s = init_s.cuda()
            params = params.cuda()

        if args.grid_params:
            args.num_configs = cnt
            BS = 2 * n_vref
        else:
            if args.traj_roa_alpha:
                BS = 2
            else:
                BS = 50
        n_cfg_batch = args.num_configs // BS
        N = args.num_samples

        cfg_list=[]
        c_list=[]
        cratio_list=[]
        sratio_list=[]

        for i in tqdm.tqdm(range(n_cfg_batch)):
            dbg_t1=time.time()
            # print(i, n_cfg_batch)
            bs = i * BS
            be = (i+1) * BS
            params_b = params[bs:be].unsqueeze(1).repeat(1, N, 1).reshape(BS * N, args.N_REF+1)
            init_s_b = init_s.repeat(BS, 1, 1).reshape(BS * N, args.X_DIM+args.N_REF+1)
            dbg_t2 = time.time()
            # print("FOR",i,init_s_b.shape, params_b.shape)

            init_s_b[:, args.X_DIM:] = params_b  # (BS * N, 4)
            if args.traj_roa_alpha:
                traj_s_b, _ = za.get_trajectories(init_s_b, actor, car_params, params_b,
                                                   is_lqr=False, lqr_K_cuda=None, args=args, detached=True, tail=False)
                final_s_b = traj_s_b[:,-1,:]
            else:
                final_s_b, _ = za.get_trajectories(init_s_b, actor, car_params, params_b,
                                           is_lqr=False, lqr_K_cuda=None, args=args, detached=True, tail=True)
            dbg_t3 = time.time()
            init_v_b = clf(init_s_b)
            if args.traj_roa_alpha:
                mask_b, _ = za.analyze_traj_u(final_s_b.unsqueeze(1), None, None, header=None, skip_print=True,
                                              skip_viz=True, args=args,
                                              trs_v=clf(traj_s_b))
            else:
                mask_b, _ = za.analyze_traj_u(final_s_b.unsqueeze(1), None, None, header=None, skip_print=True, skip_viz=True, args=args,
                                          trs_v=torch.stack([init_v_b, clf(final_s_b)], dim=1))
            dbg_t4 = time.time()
            for j in range(BS):
                bbs = j * N
                bbe = (j + 1) * N
                c_in, c_out = za.levelset_linesearch(
                    init_v_b[bbs:bbe], mask_b[bbs:bbe], None, None, None, None, None, args)
                c_in_idx = torch.where(init_v_b[bbs:bbe] <= c_in)[0]#.detach().cpu().numpy()
                c_in_vol = c_in_idx.shape[0]
                s_in_idx = torch.where(mask_b[bbs:bbe] == True)[0]#.detach().cpu().numpy()
                s_in_vol = s_in_idx.shape[0]
                space_vol = args.num_samples
                c_ratio = c_in_vol / space_vol
                s_ratio = s_in_vol / space_vol
                cfg_list.append(params_b[j * N, :args.N_REF])
                c_list.append(c_in)
                cratio_list.append(c_ratio)
                sratio_list.append(s_ratio)
            dbg_t5 = time.time()
            # print("%.4f %.4f %.4f %.4f | %.4f"%(dbg_t2-dbg_t1, dbg_t3-dbg_t2, dbg_t4-dbg_t3,dbg_t5-dbg_t4, dbg_t5 - dbg_t1))
        cfg_tensor = torch.stack(cfg_list, dim=0)  # args.num_configs * 3
        c_tensor = torch.tensor(c_list).float().reshape(args.num_configs, 1)
        cr_tensor = torch.tensor(cratio_list).float().reshape(args.num_configs, 1)
        sr_tensor = torch.tensor(sratio_list).float().reshape(args.num_configs, 1)
        if args.gpus is not None:
            c_tensor = c_tensor.cuda()
            cr_tensor = cr_tensor.cuda()
            sr_tensor = sr_tensor.cuda()
        levelset_data = torch.cat((cfg_tensor, c_tensor, cr_tensor, sr_tensor), dim=-1)
        np.savez("%s/../levelset_data.npz"%(args.viz_dir),
                 data=levelset_data.detach().cpu().numpy(),
                 )

    ############
    # TRAINING #
    ############
    args.net_lr = args.net_setups[0]
    args.net_epochs = int(args.net_setups[1])
    args.net_print_freq = int(args.net_setups[2])
    args.net_viz_freq = int(args.net_setups[3])
    args.net_save_freq = int(args.net_setups[4])

    if args.gpus is not None:
        levelset_data = levelset_data.cuda()

    split = 0.8
    levelset_input = levelset_data[:, 0:3]
    levelset_gt = levelset_data[:, -3:-2]
    train_input, test_input = utils.split_tensor(levelset_input, int(split * args.num_configs))
    train_gt, test_gt = utils.split_tensor(levelset_gt, int(split * args.num_configs))
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
            print("%04d/%04d train: %.4f (%.4f) %.4f %.4f  test: %.4f (%.4f) %.4f %.4f" % (
                epi, args.net_epochs, train_losses.val, train_losses.avg, train_losses1.val, train_losses2.val,
                test_losses.val, test_losses.avg, test_losses1.val, test_losses2.val))

        cmap = cm.get_cmap('inferno')
        if epi == 0 or epi == args.net_epochs - 1 or epi % args.net_viz_freq == 0:
            scale = 1.0
            if args.grid_params:
                target = levelset_input
                mu0_idx = torch.where(levelset_input[:, 0] == args.mu_choices[0])[0]
                mu1_idx = torch.where(levelset_input[:, 0] == args.mu_choices[1])[0]
                indices = [mu0_idx, mu1_idx]
                levelset_est = net(levelset_input)
                sources = [levelset_est, levelset_gt]
            else:
                target = test_input
                mu0_idx = torch.where(test_input[:, 0] == args.mu_choices[0])[0]
                mu1_idx = torch.where(test_input[:, 0] == args.mu_choices[1])[0]
                indices = [mu0_idx, mu1_idx]
                sources = [test_est, test_gt]
            if args.ref_mins[1] == args.ref_maxs[1]:
                for viz_i in range(2):
                    plt.subplot(2, 1, viz_i+1)
                    _, order_i = torch.sort(target[indices[viz_i], 1])
                    plt.plot(target[indices[viz_i], 1][order_i].detach().cpu().numpy(),
                             sources[0][indices[viz_i]][order_i].detach().cpu().numpy(), label="NN estimate")
                    plt.plot(target[indices[viz_i], 1][order_i].detach().cpu().numpy(),
                             sources[1][indices[viz_i]][order_i].detach().cpu().numpy(), label="Real")
                plt.xlabel("v_ref (m/s)")
                plt.ylabel("levelset value")
                plt.legend()
                plt.savefig("%s/viz_%05d.png" % (args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
                plt.close()
            else:
                for viz_i in range(2):
                    for viz_j in range(2):
                        plt.subplot(2, 2, viz_i * 2 + viz_j + 1)
                        plt.scatter(target[indices[viz_i], 1].detach().cpu().numpy(),
                                    target[indices[viz_i], 2].detach().cpu().numpy(), s=scale,
                                    c=sources[viz_j][indices[viz_i]].detach().cpu().numpy(),
                                    cmap=cmap)
                        plt.xlabel("v_ref (m/s)")
                        plt.ylabel("w_ref (rad/s)")
                        plt.colorbar()
                    plt.tight_layout()
                plt.savefig("%s/viz_%05d.png" % (args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
                plt.close()

        if epi == 0 or epi == args.net_epochs -1 or epi % args.net_save_freq == 0:
            torch.save(net.state_dict(), "%s/net_model_%05d.ckpt" % (args.model_dir, epi))


if __name__ == "__main__":
    args = za.hyperparameters()

    # TODO fixed setup args
    if args.u_limit is not None:
        args.tanh_w_gain = args.u_limit[0]
        args.tanh_a_gain = args.u_limit[1]
    if args.joint_pretrained_path is not None:
        args.actor_pretrained_path = args.joint_pretrained_path
        args.clf_pretrained_path = args.joint_pretrained_path
    args.actor_clip_u = True
    args.multi = True
    args.load_last = True
    args.clip_angle = True
    args.X_DIM = 7
    args.N_REF = 3  # (mu, vref, wref)
    args.U_DIM = 2  # (omega, accel)
    args.N_CFGS = args.num_samples
    args.controller_dt = args.dt

    if hasattr(args, "model_iter") == False:
        args.model_iter = None

    if args.model_iter is not None:
        bak_list = args.model_iter
        exp_name = args.exp_name
        total_t1=time.time()
        for bak_val in bak_list:
            args.model_iter = bak_val
            args.exp_name = exp_name + "_%03d"%(args.model_iter)
            t1 = time.time()
            car_params = sscar_utils.VehicleParameters()
            main()
            t2 = time.time()
            print("ROA training finished in %.4fs" % (t2 - t1))
        print("RoA Training total finished in %.4fs"%(time.time() - total_t1))
    else:
        t1=time.time()
        car_params = sscar_utils.VehicleParameters()
        main()
        t2 = time.time()
        print("ROA training finished in %.4fs" % (t2 - t1))
