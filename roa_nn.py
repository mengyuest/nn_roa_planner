import torch
import torch.nn as nn
import roa_utils as utils
import cgw_sim_choi as choi


def lc_forward(fc_list, x):
    num_layers = len(fc_list)
    for i, hidden in enumerate(range(num_layers - 1)):
        x = nn.ReLU()(fc_list[i](x))
    v = fc_list[-1](x)
    return v


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.linear_list = utils.create_fcs(args.N_REF, 1, args.net_hiddens)

    def embedding(self, ref):
        if self.args.X_DIM < 5:
            new_ref = torch.stack([
                ref[:, 0],  # mu factor
                ref[:, 1],  # wref
            ], dim=-1)
        else:
            new_ref = torch.stack([
                ref[:, 0],  # mu factor
                (ref[:, 1] - 17.5) / 12.5,  # vref
                ref[:, 2],  # wref
            ], dim=-1)
        return new_ref


    def forward(self, ref):
        level = lc_forward(self.linear_list, self.embedding(ref))
        if self.args.net_exp:
            level = torch.exp(level)
        return level


class CLF(nn.Module):
    def __init__(self, lqr_P, args):
        super(CLF, self).__init__()
        self.args = args
        self.X_DIM = X_DIM = args.X_DIM
        self.N_REF = N_REF = args.N_REF
        if self.args.clf_mode == "ellipsoid":
            if self.args.multi:
                if self.args.clf_e2e:
                    self.linear_list = utils.create_fcs(X_DIM + N_REF, 1, args.clf_hiddens)
                else:
                    self.linear_list = utils.create_fcs(N_REF, X_DIM * X_DIM, args.clf_hiddens)
                    if self.args.clf_ell_scale:
                        self.linear_list2 = utils.create_fcs(N_REF, X_DIM, args.clf_hiddens)
                    elif self.args.clf_ell_extra:
                        self.linear_list2 = utils.create_fcs(X_DIM + N_REF, 1, args.clf_hiddens)
            else:
                if self.args.clf_init_type == "eye":
                    P_init = torch.eye(X_DIM) / 2
                elif self.args.clf_init_type == "lqr":
                    P_init = lqr_P / 2
                elif self.args.clf_init_type == "rand":
                    P_init = torch.rand(X_DIM, X_DIM) / 2
                else:
                    raise NotImplementedError
                self.P = torch.nn.Parameter(P_init)
        elif self.args.clf_mode == "diag":
            if self.args.multi:
                self.linear_list = utils.create_fcs(N_REF, X_DIM, args.clf_hiddens)
            else:
                self.scales = torch.nn.Parameter(torch.ones(X_DIM, ))
        elif self.args.clf_mode == "nn":
            if self.args.multi:
                self.linear_list = utils.create_fcs(X_DIM + N_REF, 1, args.clf_hiddens)
            else:
                self.linear_list = utils.create_fcs(X_DIM, 1, args.clf_hiddens)
        else:
            raise NotImplementedError

        if self.args.normalize:
            assert self.X_DIM <=5
            # in_means (xd, y, theta, F)
            # (xd, y, xd_ref, y_ref) -> 1
            self.in_means = torch.cat([args.in_means[:2], args.in_means[:2]], dim=-1)
            self.in_stds = torch.cat([args.in_stds[:2], args.in_stds[:2]], dim=-1)
            self.out_means = torch.zeros_like(args.in_means[:1])
            self.out_stds =torch.ones_like(args.in_means[:1])

    def update_device(self):
        if self.args.normalize:
            self.in_means = self.in_means.cuda()  # .to(device)
            self.in_stds = self.in_stds.cuda()  # .to(device)
            self.out_means = self.out_means.cuda()  # .to(device)
            self.out_stds = self.out_stds.cuda()  # .to(device)

    def embedding(self, ref):
        if self.X_DIM>5:
            if hasattr(self.args, "single_clf") and self.args.single_clf:
                new_ref = ref * 0.0
            else:
                new_ref = torch.stack([
                    ref[:, 0],  # mu factor
                    (ref[:, 1] - 17.5) / 12.5,  # vref
                    ref[:, 2],  # wref
                ], dim=-1)
        else:
            new_ref = torch.stack([
                ref[:, 0],  # xdot
                ref[:, 1],  # y
            ], dim=-1)
        return new_ref


    def forward(self, x):
        if len(x.shape)==3:
            reshape = True
            N, T, K = x.shape
            x = x.reshape([N*T, K])
        else:
            reshape = False
        if self.args.normalize:
            x = (x - self.in_means) / self.in_stds
        origin_x = x[:, : self.X_DIM + self.N_REF]
        ref = x[:, self.X_DIM: self.X_DIM + self.N_REF]
        x = x[:, :self.X_DIM]

        batch_size = x.shape[0]
        if self.args.clf_mode == "ellipsoid":
            if self.args.multi:
                if self.args.clf_e2e:
                    v = lc_forward(self.linear_list, origin_x)  # (B, 1)
                    v = v * v
                else:
                    P = lc_forward(self.linear_list, self.embedding(ref)).reshape(batch_size, self.X_DIM, self.X_DIM)
                    if self.args.clf_ell_scale:
                        scale = lc_forward(self.linear_list2, self.embedding(ref)).reshape(batch_size, self.X_DIM, 1)
                        scale = torch.exp(scale)
                        Px = (scale * torch.bmm(P, torch.unsqueeze(x, dim=-1))).squeeze(-1)  # (B, N, 1)
                    elif self.args.clf_ell_extra:
                        Px = torch.bmm(P, torch.unsqueeze(x, dim=-1)).squeeze(-1)  # (B, N, 1)
                        extra = lc_forward(self.linear_list2, torch.cat([x, self.embedding(ref)], dim=-1))
                        if self.args.clf_ell_extra_tanh:
                            extra = nn.Tanh()(extra)
                        if self.args.clf_ell_extra_xnorm:
                            extra = extra * torch.norm(x, dim=-1, keepdim=True)
                    else:
                        if self.args.X_DIM<5:
                            Px = torch.bmm(P, torch.unsqueeze(x-ref, dim=-1)).squeeze(-1)  # (B, N, 1)
                            if self.args.clf_ell_eye:
                                extra = torch.norm(x-ref, dim=-1, keepdim=True)
                        else:
                            Px = torch.bmm(P, torch.unsqueeze(x, dim=-1)).squeeze(-1)  # (B, N, 1)
                    Px_norm = torch.norm(Px, dim=-1, keepdim=True)  # (B, 1)
                    v = Px_norm * Px_norm / 2   # (B, 1)
                    if self.args.clf_ell_extra:
                        v = v + extra
                    elif self.args.clf_ell_eye:
                        v = v + extra * self.args.clf_eye_scale
            else:
                PP = self.P + self.P.t()
                v = torch.bmm(torch.unsqueeze(x @ PP, dim=1), x.unsqueeze(dim=-1)).squeeze(dim=-1) / 2
        elif self.args.clf_mode == "diag":
            if self.args.multi:
                if self.args.X_DIM < 5:
                    v = torch.norm(x-ref, dim=-1, keepdim=True)
                    v = v ** 2
                else:
                    scale = lc_forward(self.linear_list, self.embedding(ref)).reshape(batch_size, self.X_DIM)
                    if self.args.clf_no_exp_scale == False:
                        scale = torch.exp(scale)
                    v = torch.norm(scale * x, dim=-1, keepdim=True)
                    v = v ** 2
            else:
                v_norm = torch.norm(x * self.scales, dim=-1, keepdim=True)
                v = v_norm ** 2
        else:
            assert self.args.clf_mode == "nn"
            if self.args.multi:
                v = lc_forward(self.linear_list, origin_x)
            else:
                v = lc_forward(self.linear_list, x)
            v = v ** 2

        if reshape:
            v = v.reshape(N, T, 1)
        return v


class Actor(nn.Module):
    def __init__(self, lqr_K, args):
        super(Actor, self).__init__()
        self.args = args
        self.X_DIM = X_DIM = args.X_DIM
        self.N_REF = N_REF = args.N_REF
        self.U_DIM = U_DIM = args.U_DIM
        if self.args.actor_mode == "mat" or self.args.actor_mode == "mat_nn":
            if self.args.multi:
                if self.args.actor_e2e:
                    self.linear_list = utils.create_fcs(X_DIM + N_REF, U_DIM, args.actor_hiddens)
                else:
                    self.linear_list = utils.create_fcs(N_REF, U_DIM * X_DIM, args.actor_hiddens)
            else:
                if self.args.actor_init_type == "lqr":
                    K_init = lqr_K.float()
                elif self.args.actor_init_type == "rand":
                    K_init = torch.random(U_DIM, X_DIM)
                else:
                    raise NotImplementedError
                self.K = torch.nn.Parameter(K_init)
        if self.args.actor_mode == "nn" or self.args.actor_mode == "mat_nn":
            if self.args.multi:
                if self.args.actor_e2e == False:
                    self.linear_list2 = utils.create_fcs(X_DIM + N_REF, U_DIM, args.actor_hiddens)
                else:
                    self.linear_list = utils.create_fcs(X_DIM + N_REF, U_DIM, args.actor_hiddens)
            else:
                self.linear_list = utils.create_fcs(X_DIM, U_DIM, args.actor_hiddens)

        if self.args.normalize:
            assert self.X_DIM <= 5
            # in_means (xd, y, theta, F)
            # out_means (x, xd, y)
            # (xd, y, xd_ref, y_ref) -> (theta, F)
            self.in_means = torch.cat([args.in_means[:2], args.in_means[:2]], dim=-1)
            self.in_stds = torch.cat([args.in_stds[:2], args.in_stds[:2]], dim=-1)
            self.out_means = args.out_means[1:]
            self.out_stds = args.out_stds[1:]
            # self.out_means = torch.zeros_like(args.out_means[1:])
            # self.out_stds = torch.tensor([1.0, 2000])
            # self.out_stds = torch.tensor([0.01, 1])

    def update_device(self):
        if self.args.normalize:
            self.in_means = self.in_means.cuda()  # .to(device)
            self.in_stds = self.in_stds.cuda()  # .to(device)
            self.out_means = self.out_means.cuda()  # .to(device)
            self.out_stds = self.out_stds.cuda()  # .to(device)

    def embedding(self, ref):
        if self.X_DIM > 5:
            new_ref = torch.stack([
                ref[:, 0],  # mu factor
                (ref[:, 1] - 17.5) / 12.5,  # vref
                ref[:, 2],  # wref
            ], dim=-1)
        else:
            new_ref = torch.stack([
                ref[:, 0],  # xdot
                ref[:, 1],  # y
            ], dim=-1)
        return new_ref


    def forward(self, x, return_K=False):
        batch_size = x.shape[0]
        if self.args.normalize:
            x = (x - self.in_means) / self.in_stds

        origin_x = x[:, :self.X_DIM + self.N_REF]
        ref = x[:, self.X_DIM: self.X_DIM + self.N_REF]
        x = x[:, :self.X_DIM]
        if self.args.multi and self.args.actor_e2e:
            u_e2e = lc_forward(self.linear_list, origin_x)
            u_e2e = torch.stack([
                torch.clamp(u_e2e[:, 0], -self.args.tanh_w_gain, self.args.tanh_w_gain),
                torch.clamp(u_e2e[:, 1], -self.args.tanh_a_gain, self.args.tanh_a_gain)
            ], dim=-1)
            u_all = u_e2e

        if self.args.actor_mode == "mat" or self.args.actor_mode == "mat_nn":
            if self.args.multi:
                if self.args.actor_e2e == False:
                    K_mat = lc_forward(self.linear_list, self.embedding(ref)).reshape(batch_size, self.U_DIM, self.X_DIM)
                    u_mat = torch.bmm(-K_mat, x.unsqueeze(-1)).squeeze(-1)
                else:
                    raise NotImplementedError
            else:
                u_mat = torch.mm(x, -self.K.T)
            if self.args.actor_clip_u:
                u_mat = torch.stack([
                    torch.clamp(u_mat[:, 0], -self.args.tanh_w_gain, self.args.tanh_w_gain),
                    torch.clamp(u_mat[:, 1], -self.args.tanh_a_gain, self.args.tanh_a_gain)
                ], dim=-1)
        if self.args.actor_mode == "nn" or self.args.actor_mode == "mat_nn":
            if self.args.multi:
                if self.args.actor_e2e == False:
                    u_nn = lc_forward(self.linear_list2, origin_x)
                else:
                    u_nn = lc_forward(self.linear_list, origin_x)
            else:
                u_nn = lc_forward(self.linear_list, x)
            u_nn = nn.Tanh()(u_nn)
            u_nn = torch.stack([
                u_nn[:, 0] * self.args.tanh_w_gain,
                u_nn[:, 1] * self.args.tanh_a_gain], dim=-1)

        if self.args.actor_mode == "mat":
            u_all = u_mat
        elif self.args.actor_mode == "nn":
            u_all = u_nn
        elif self.args.actor_mode == "mat_nn":
            ratio = self.args.actor_nn_res_ratio
            u_all = u_mat * (1-ratio) + u_nn * ratio

        if self.args.normalize:
            if self.args.u_norm==False:
                u_all = u_all * self.out_stds + self.out_means

        if return_K:
            return u_all, K_mat
        else:
            return u_all


class CGW_Actor(nn.Module):
    def __init__(self, args):
        super(CGW_Actor, self).__init__()
        self.args = args
        self.X_DIM = X_DIM = args.X_DIM
        self.N_REF = N_REF = args.N_REF
        self.U_DIM = U_DIM = args.U_DIM
        self.linear_list = utils.create_fcs(X_DIM + N_REF, U_DIM, args.actor_hiddens)

    def forward(self, x):
        u = lc_forward(self.linear_list, x)
        u_nn = nn.Tanh()(u) * self.args.u_gain

        prev_th = self.args.params.th1d
        self.args.params.th1d = x[:, -1]
        u_qp = choi.two_link_io_control_clf_qp(x[:, :-1],
               self.args.params, use_torch=True, qp_bound=self.args.qp_bound)
        self.args.params.th1d = prev_th
        u = u_nn * self.args.nn_ratio + u_qp * (1 - self.args.nn_ratio)
        return u


class CGW_CLF(nn.Module):
    def __init__(self, args):
        super(CGW_CLF, self).__init__()
        self.args = args
        self.X_DIM = X_DIM = args.X_DIM
        self.N_REF = N_REF = args.N_REF
        self.U_DIM = U_DIM = args.U_DIM
        self.gait_data = args.gait_data
        self.gait_th = args.gait_th.unsqueeze(0)  # (1, n_th)
        self.linear_list = utils.create_fcs(X_DIM + N_REF + X_DIM, U_DIM, args.clf_hiddens)

    # TODO (get ref)
    def get_ref(self, x, num_ths):
        th_tar = x[:, -1].unsqueeze(1)  # (n_samples, 1)
        th_idx = torch.argmin(torch.abs(th_tar - self.gait_th), dim=1)  # (n_samples, )
        ref_list = []
        assert x.shape[0] % num_ths == 0
        n_batch = x.shape[0] // num_ths
        for i in range(num_ths):
            bi0 = i * n_batch
            bi1 = (i + 1) * n_batch
            gait_data = self.gait_data[th_idx[bi0]].unsqueeze(0)  # (1, gait_len, 4)
            gait_idx = torch.argmin(torch.abs(gait_data[:, :, 0]-x[bi0:bi1, 0].unsqueeze(1)), dim=1)  # (n_batch, )
            ref = gait_data[0, gait_idx]
            ref_list.append(ref)
        ref_list = torch.cat(ref_list, dim=0)
        # TODO(check here)
        return ref_list


    def forward(self, x, num_ths, ref=None):
        if ref is None:
            ref = self.get_ref(x, num_ths)
        # for i in range(ref.shape[0]):
        #     print("%04d %.3f %.3f %.3f %.3f"%(i, ref[i, 0], ref[i, 1], ref[i, 2], ref[i, 3]))
        x_err = x[:, :-1] - ref
        new_x = torch.cat([x, x_err], dim=-1)  # (n_samples, 9)

        # TODO use a Q^TQ format?
        v = lc_forward(self.linear_list, new_x)

        return v**2