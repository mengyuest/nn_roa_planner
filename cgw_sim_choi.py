import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import roa_utils as utils
import torch

def get_gait_ref(x, gait):
    q1, q2, q1d, q2d = torch.split(x, split_size_or_sections=1, dim=-1)
    q1_all = gait[:, 0].reshape((1, -1))
    idx = torch.argmin(torch.abs(q1 - q1_all), dim=1)
    ref_q1 = gait[idx, 0:1]
    ref_q2 = gait[idx, 1:2]
    ref_q1d = gait[idx, 2:3]
    ref_q2d = gait[idx, 3:4]
    return torch.cat((ref_q1, ref_q2, ref_q1d, ref_q2d), dim=-1)

def get_closest(q1_query, gait_data, dim):
    q1_query_ful = q1_query.reshape((-1, 1))
    q1 = gait_data[:, 0].reshape((1, -1))
    idx = np.argmin(np.abs(q1 - q1_query_ful), axis=1)
    return gait_data[idx, dim]

def get_hjb_u(x, oracle):
    x_cache = x.detach().cpu().numpy()
    u = oracle.get_u_batch(x_cache)
    u = torch.from_numpy(u).type_as(x)
    return u


def get_qp_u(x, args, params=None):
    if params is None:
        params = args.params
    return two_link_io_control_clf_qp(x, params, use_torch=True, qp_bound=args.qp_bound)


def poly(q1, params):
    beta1, beta2, beta3, beta4 = params.beta
    th1d = params.th1d
    return - q1 * 2.0 + (q1 + th1d) * (q1 - th1d) * (beta1 + beta2 * q1 + beta3 * q1 ** 2 + beta4 * q1 ** 3)


def dpoly(q1, dq1, params):
    beta1, beta2, beta3, beta4 = params.beta
    th1d = params.th1d
    # return q1 * 2.0 - (q1 + th1d) * (q1 - th1d) * (beta1 + beta2 * q1 + beta3 * q1 ** 2 + beta4 * q1 ** 3)
    t1 = 2 * q1
    t2 = beta1 + beta2 * q1 + beta3 * q1**2 + beta4 * q1 ** 3
    t3 = (q1 - th1d) * (q1 + th1d)
    t4 = beta2 + 2 * beta3 * q1 + 3 * beta4 * q1 ** 2
    return dq1 * (- 2 + t1 * t2 + t3 * t4)


def dpoly_dq1(q1, params):
    beta1, beta2, beta3, beta4 = params.beta
    th1d = params.th1d
    t1 = 2 * q1
    t2 = beta1 + beta2 * q1 + beta3 * q1**2 + beta4 * q1 ** 3
    t3 = (q1 - th1d) * (q1 + th1d)
    t4 = beta2 + 2 * beta3 * q1 + 3 * beta4 * q1 ** 2
    return - 2 + t1 * t2 + t3 * t4


def detect_switch(x, prev_x, args, use_torch=True):
    q1 = x[:, 0]
    q2 = x[:, 1]
    dq1 = x[:, 2]
    dq2 = x[:, 3]
    prev_q1 = prev_x[:, 0]
    prev_q2 = prev_x[:, 1]

    curr_2q1_geq_q2 = -2 * q1 >= q2
    prev_2q1_lt_q2 = -2 * prev_q1 < prev_q2
    curr_q1_leq_thres = q1 <= args.reset_q1_threshold
    dq1_dq2_leq_0 = (2 * dq1 + dq2) <= 0
    if use_torch:
        t_and = torch.logical_and
    else:
        t_and = np.logical_and
    mask_bool = t_and(t_and(curr_2q1_geq_q2, prev_2q1_lt_q2), t_and(curr_q1_leq_thres, dq1_dq2_leq_0))
    if use_torch:
        return mask_bool.unsqueeze(-1).float()
    else:
        return mask_bool[:, None]


def compute_xdot(x, u, use_torch=False, args=None):
    fvec = get_fvec(x, use_torch)
    gvec = get_gvec(x, use_torch)
    if args is not None and args.constant_g:
        gvec = torch.stack([
            gvec[:, 0] * 0,
            gvec[:, 1] * 0,
            gvec[:, 2] * 0 + 1,
            gvec[:, 3] * 0 + 1,
        ], dim=-1)
    if args is not None and args.changed_dynamics:
        xdot = fvec + torch.stack(
            (u[:, 0], u[:, 1], gvec[:, 2] * u[:, 2], gvec[:, 3] * u[:, 2] + u[:, 3]), axis=-1)
    else:
        xdot = fvec + gvec * u
    return xdot


def get_fvec(x, use_torch=False):
    lL = 1.0
    mL = 1.0
    g = 9.81
    mH = 1.0

    q1 = x[:, 0]
    q2 = x[:, 1]
    dq1 = x[:, 2]
    dq2 = x[:, 3]

    if use_torch:
        cos = torch.cos
        sin = torch.sin
        stack = torch.stack
    else:
        cos = np.cos
        sin = np.sin
        stack = np.stack

    t2 = cos(q1)
    t3 = sin(q1)
    t4 = sin(q2)
    t5 = dq1 ** 2
    t6 = dq2 ** 2
    t7 = mH * 4.0
    t8 = mL * 3.0
    t9 = q2 * 2.0
    t12 = 1.0 / lL
    t10 = cos(t9)
    t11 = sin(t9)
    t13 = g * t3 * t7
    t14 = g * mL * t3 * 4.0
    t17 = dq1 * dq2 * lL * mL * t4 * 4.0
    t18 = lL * mL * t4 * t6 * 2.0
    t15 = mL * t10 * 2.0
    t16 = -t15
    t19 = t7 + t8 + t16
    t20 = 1.0 / t19

    ddq1 = -t12 * t20 * (-t14 + t17 + t18 - g * mH * t3 * 4.0 + g * mL * sin(q1 + t9) * 2.0 + lL * mL * t4 * t5 * 2.0 - lL * mL * t5 * t11 * 2.0)
    ddq2 = -t12 * t20 * (t13 + t14 - t17 - t18 + g * mH * t2 * t4 * 8.0 + g * mL * t2 * t4 * 1.0e+1 - g * mL * t2 * t11 * 2.0 - g * mL * t3 * t10 * 2.0
                       - lL * mH * t4 * t5 * 8.0 - lL * mL * t4 * t5 * 1.2e+1 + lL * mL * t5 * t11 * 4.0 + lL * mL * t6 * t11 * 2.0-g * mL * t3 * cos(q2) * 2.0
                       + dq1 * dq2 * lL * mL * t11 * 4.0)

    if use_torch:
        fvec = stack((dq1, dq2, ddq1, ddq2), dim=-1)
    else:
        fvec = stack((dq1, dq2, ddq1, ddq2), axis=-1)
    return fvec


def get_gvec(x, use_torch=False):
    lL = 1.0
    mL = 1.0
    g = 9.81
    mH = 1.0

    q2 = x[:, 1]
    if use_torch:
        cos = torch.cos
        stack = torch.stack
    else:
        cos = np.cos
        stack = np.stack

    t2 = cos(q2)
    t3 = mH * 4.0
    t4 = mL * 5.0
    t6 = 1.0 / (lL ** 2)
    t5 = t2 ** 2
    t7 = mL * t5 * 4.0
    t8 = -t7
    t9 = t3 + t4 + t8
    t10 = 1.0 / t9

    gvec3 = t6 * t10 * (t2 * 8.0 - 4.0)
    gvec4 = (t6 * t10 * (mH * 1.6e+1 + mL * 2.4e+1 - mL * t2 * 1.6e+1)) / mL

    if use_torch:
        gvec = stack((0 * gvec3, 0 * gvec3, gvec3, gvec4), dim=-1)
    else:
        gvec = stack((0 * gvec3, 0 * gvec3, gvec3, gvec4), axis=-1)

    return gvec


def compute_fine(x, prev_x, args, use_torch=True):
    if use_torch:
        q1_minus, q2_minus, _, _ = torch.split(prev_x, split_size_or_sections=1, dim=-1)
        q1_plus, q2_plus, _, _ = torch.split(x, split_size_or_sections=1, dim=-1)
    else:
        q1_minus, q2_minus, _, _ = np.split(prev_x, 4, axis=-1)
        q1_plus, q2_plus, _, _ = np.split(x, 4, axis=-1)

    denom = q2_minus - q2_plus + 2 * q1_minus - 2 * q1_plus
    if use_torch:
        alpha = (-q2_plus - 2 * q1_plus) / torch.where(denom==0, torch.ones_like(denom), denom)
    else:
        alpha = (-q2_plus - 2 * q1_plus) / np.where(denom == 0, np.ones_like(denom), denom)
    return alpha * prev_x + (1 - alpha) * x


def compute_impact(x, use_torch=False):
    if use_torch:
        cos = torch.cos
        sin = torch.sin
        stack = torch.stack
    else:
        cos = np.cos
        sin = np.sin
        stack = np.stack

    q1_minus = x[:, 0]
    q2_minus = x[:, 1]
    dq1_minus = x[:, 2]
    dq2_minus = x[:, 3]

    dq1 = dq1_minus
    dq2 = dq2_minus
    q1 = q1_minus
    q2 = q2_minus
    dx = - cos(q1) * dq1
    dy = - sin(q1) * dq1

    t2 = cos(q1)
    t3 = cos(q2)
    t4 = sin(q1)
    t5 = q1 + q2
    t6 = q1 * 2.0
    t7 = q2 * 2.0
    t14 = -q2
    t8 = cos(t6)
    t9 = cos(t7)
    t10 = sin(t6)
    t11 = sin(t7)
    t12 = cos(t5)
    t13 = sin(t5)
    t15 = dq1 * t3 * 2.0
    t16 = dq2 * t3 * 2.0
    t17 = q2 + t5
    t21 = dx * t2 * 8.0
    t22 = q1 + t14
    t23 = dy * t4 * 8.0
    t24 = t5 * 2.0
    t18 = cos(t17)
    t19 = sin(t17)
    t20 = t9 * 2.0
    t25 = -t21
    t26 = -t23
    t28 = cos(t24)
    t29 = sin(t24)
    t27 = dq1 * t20
    t30 = t20 - 7.0
    t31 = dx * t18 * 1.0e+1
    t32 = dy * t19 * 1.0e+1
    t33 = 1.0 / t30

    dq1_mid = t33 * (dq1 * -7.0 + t15 + t16 + t25 + t26 + t27 + t31 + t32)
    dq2_mid = -t33 * (dq1 * -8.0 - dq2 + t15 + t16 + t25 + t26 + t27 + t31 + t32 - dx * t12 * 8.0 - dy * t13 * 8.0 + dx * cos(t22) * 2.0 + dy * sin(t22) * 2.0)

    q1_plus = q1_minus + q2_minus
    q2_plus = -q2_minus
    dq1_plus = dq1_mid + dq2_mid
    dq2_plus = -dq2_mid
    if use_torch:
        new_x = stack((q1_plus, q2_plus, dq1_plus, dq2_plus), dim=-1)
    else:
        new_x = stack((q1_plus, q2_plus, dq1_plus, dq2_plus), axis=-1)

    return new_x


def plot_sim(x, xf, ti):
    l = 1.0
    L = 5
    plt.plot([0.0, L*np.cos(args.alpha)], [L*np.sin(args.alpha), 0.0], color="brown", label="slope")

    q1 = x[0, 0]
    q2 = x[0, 1]

    q1x = xf * np.cos(args.alpha)
    q1y = (L - xf) * np.sin(args.alpha)

    cx = q1x - l * np.sin(q1 - args.alpha)
    cy = q1y + l * np.cos(q1 - args.alpha)

    q2x = cx + l * np.sin(q1 + q2 - args.alpha)
    q2y = cy - l * np.cos(q1 + q2 - args.alpha)

    plt.plot([q1x, cx], [q1y, cy], color="blue", label="leg-stance")
    plt.plot([q2x, cx], [q2y, cy], color="red", label="leg-swing")
    plt.scatter(q2x, q2y, color="red", label="support")
    plt.scatter(cx, cy, color="black", label="CoM")

    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("scaled")
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.title("SIM (t=%04d/%04d)"%(ti, args.nt))
    plt.savefig("%s/sim_%04d.png"%(args.viz_dir, ti), bbox_inches='tight', pad_inches=0.1)
    plt.close()


def gait_next(x, args):
    if args.beta:
        q1 = x[0, 0]
        q2 = x[0, 1]
        q1d = x[0, 2]
        q2d = x[0, 3]

        beta1 = -17.9893
        beta2 = -49.9978
        beta3 = 12.9808
        beta4 = -7.7981
        th1d = -0.13

        q1_sym = -q2
        q2_sym = -2 * q1_sym + (q1_sym + th1d) * (q1_sym - th1d) * \
                 (beta1 + beta2*q1_sym + beta3 * q1_sym**2 +
                  beta4 * q1_sym**3)

        new_x = np.zeros((1, 4))

        new_q1 = q2_sym
        new_q2 = q2 + q2d * args.dt

        new_x[0, 0] = new_q1
        new_x[0, 1] = new_q2
        new_x[0, 2] = 0
        new_x[0, 3] = q2d

    else:
        q2_ref = 0.2
        q2d_ref = 0.2

        poly_1 = 3
        poly_3 = -1 / (q2_ref ** 2)

        q1 = x[0, 0]
        q2 = x[0, 1]
        q1d = x[0, 2]
        q2d = x[0, 3]

        new_x = np.zeros((1, 4))

        new_x[0, 1] = q2 + q2d * args.dt
        new_q2 = new_x[0, 1]
        new_x[0, 0] = poly_1 * new_q2 + poly_3 * (new_q2 ** 3)
        new_x[0, 2] = poly_1 * q2d_ref + 3 * poly_3 * (new_q2**2) * q2d_ref
        new_x[0, 3] = q2d_ref

    return new_x


def two_link_io_control_clf_qp(x, params, use_torch=False, qp_bound=4.0):
    u_bound = qp_bound
    LgLfy = LgLfy_gen(x, params, use_torch)
    Lf2y = Lf2y_gen(x, params, use_torch)
    if use_torch:
        LgLfy = LgLfy.unsqueeze(-1)
        Lf2y = Lf2y.unsqueeze(-1)
    v = get_clf_qp_sol(x, params, use_torch)
    u = (1 / LgLfy) * (-Lf2y + v)

    if use_torch:
        u = torch.clamp(u, -u_bound, u_bound)
    else:
        u = np.clip(u, -u_bound, u_bound)
        u = u.reshape((1, 1))
    return u


def get_clf_qp_sol(x, params, use_torch=False, use_casadi=True):
    y = y_gen(x, params, use_torch, use_casadi)
    dy = Lfy_gen(x, params, use_torch, use_casadi)
    if use_torch:
        y = y.unsqueeze(-1)
        dy = dy.unsqueeze(-1)

    V = clf_FL(y, dy, params, use_torch, use_casadi)
    LfV = lF_clf_FL(y, dy, params, use_torch, use_casadi)
    LgV = lG_clf_FL(y, dy, params, use_torch, use_casadi)

    A = LgV
    b = -LfV - params.clf_rate * V

    if use_torch:
        mu_idx = torch.where(torch.logical_and(A != 0, b < 0))
        mu = torch.zeros_like(A)
        mu[mu_idx] = b[mu_idx] / A[mu_idx]
    elif use_casadi:
        import casadi
        mu = casadi.if_else(casadi.logic_or(
            casadi.logic_and(A>0, b<0),
            casadi.logic_and(A<0, b<=0)
            ), b / A, 0)
    else:
        if A == 0:
            mu = 0
        elif A > 0:
            if b < 0:
                mu = b / A
            else:
                mu = 0
        else:
            if b > 0:
                mu = 0
            else:
                mu = b / A
    return mu


def clf_FL(y, dy, params, use_torch=False, use_casadi=True):
    if use_torch:
        y_eps = y / params.eps
        return y_eps ** 2 * params.P[0, 0] + dy ** 2 * params.P[1, 1] + 2 * y_eps * dy * params.P[0, 1]
    elif use_casadi:
        y_eps = y / params.eps
        return y_eps ** 2 * params.P[0, 0] + dy ** 2 * params.P[1, 1] + 2 * y_eps * dy * params.P[0, 1]
    else:
        eta_eps = np.array([[1/params.eps * y.item()], [dy.item()]])
        return eta_eps.T @ params.P @ eta_eps


def lF_clf_FL(y, dy, params, use_torch=False, use_casadi=True):
    if use_torch:
        y_eps = y / params.eps
        return 2 * y_eps * dy * params.P[0, 0] / params.eps + 2 * params.P[0, 1] / params.eps * dy * dy
    elif use_casadi:
        y_eps = y / params.eps
        return 2 * y_eps * dy * params.P[0, 0] / params.eps + 2 * params.P[0, 1] / params.eps * dy * dy
    else:
        F_FL_eps = np.array([[0, 1 / params.eps], [0, 0]])
        eta_eps = np.array([[1 / params.eps * y.item()], [dy.item()]])
        return (eta_eps.T) @ (F_FL_eps.T @ params.P + params.P @ F_FL_eps) @ eta_eps


def lG_clf_FL(y, dy, params, use_torch=False, use_casadi=True):
    if use_torch:
        y_eps = y / params.eps
        return 2 * params.P[1, 0] * y_eps + 2 * params.P[1, 1] * dy
    elif use_casadi:
        y_eps = y / params.eps
        return 2 * params.P[1, 0] * y_eps + 2 * params.P[1, 1] * dy
    else:
        G_FL = np.array([[0.0], [1.0]])
        eta_eps = np.array([[1 / params.eps * y.item()], [dy.item()]])
        return (2 * (G_FL.T @ params.P) @ eta_eps).T


def y_gen(x, params, use_torch=False, use_casadi=True):
    beta1, beta2, beta3, beta4 = params.beta
    th1d = params.th1d
    q1 = x[:, 0]
    q2 = x[:, 1]
    return q1 * 2.0 + q2 - (q1 + th1d) * (q1 - th1d) * (beta1 + beta2 * q1 + beta3 * q1 ** 2 + beta4 * q1 ** 3)


def Lfy_gen(x, params, use_torch=False, use_casadi=True):
    beta1, beta2, beta3, beta4 = params.beta
    th1d = params.th1d
    q1 = x[:, 0]
    dq1 = x[:, 2]
    dq2 = x[:, 3]

    t2 = beta2 * q1
    t3 = q1 + th1d
    t4 = q1 ** 2
    t5 = q1 ** 3
    t6 = -th1d
    t7 = beta3 * t4
    t8 = beta4 * t5
    t9 = q1 + t6
    t10 = beta1 + t2 + t7 + t8
    return dq2 - dq1 * (t3 * t10 + t9 * t10 + t3 * t9 * (beta2 + beta3 * q1 * 2.0 + beta4 * t4 * 3.0) - 2.0)


def LgLfy_gen(x, params, use_torch=False, use_casadi=False):
    beta1, beta2, beta3, beta4 = params.beta
    th1d = params.th1d
    q1 = x[:, 0]
    q2 = x[:, 1]
    if use_torch:
        cos = torch.cos
    elif use_casadi:
        import casadi
        cos = casadi.cos
    else:
        cos = np.cos

    t2 = cos(q2)
    t3 = beta2 * q1
    t4 = q1 + th1d
    t5 = q1 ** 2
    t6 = q1 ** 3
    t8 = -th1d
    t7 = t2 ** 2
    t9 = beta3 * t5
    t10 = beta4 * t6
    t12 = q1 + t8
    t11 = t7 * 4.0
    t15 = beta1 + t3 + t9 + t10
    t13 = t11 - 9.0
    t14 = 1.0 / t13
    return t14 * (t2 * 1.6e+1 - 4.0e+1) + t14 * (t2 * 8.0 - 4.0) * (
                t4 * t15 + t12 * t15 + t4 * t12 * (beta2 + beta3 * q1 * 2.0 + beta4 * t5 * 3.0) - 2.0)


def Lf2y_gen(x, params, use_torch=False, use_casadi=False):
    beta1, beta2, beta3, beta4 = params.beta
    th1d = params.th1d
    q1 = x[:, 0]
    q2 = x[:, 1]
    dq1 = x[:, 2]
    dq2 = x[:, 3]
    if use_torch:
        cos = torch.cos
        sin = torch.sin
    elif use_casadi:
        import casadi
        cos = casadi.cos
        sin = casadi.sin
    else:
        cos = np.cos
        sin = np.sin

    t2 = sin(q1)
    t3 = sin(q2)
    t4 = beta2 * q1
    t5 = q1 + th1d
    t6 = dq1 ** 2
    t7 = dq2 ** 2
    t8 = q2 * 2.0
    t9 = q1 ** 2
    t10 = q1 ** 3
    t11 = beta3 * q1 * 2.0
    t14 = -th1d
    t12 = cos(t8)
    t13 = sin(t8)
    t15 = beta3 * t9
    t16 = beta4 * t10
    t17 = q1 + t8
    t19 = beta4 * t9 * 3.0
    t20 = q1 + t14
    t21 = t2 * 3.924e+3
    t22 = dq1 * dq2 * t3 * 2.0e+2
    t25 = t3 * t7 * 1.0e+2
    t18 = sin(t17)
    t23 = -t21
    t24 = t12 * 1.0e+2
    t27 = beta2 + t11 + t19
    t29 = beta1 + t4 + t15 + t16
    t26 = t18 * 9.81e+2
    t28 = t24 - 3.5e+2
    t30 = 1.0 / t28

    return -t6 * (t29 * 2.0 + t5 * t27 * 2.0 + t20 * t27 * 2.0 + t5 * t20 * (
                beta3 * 2.0 + beta4 * q1 * 6.0)) - t30 * (t22 + t23 + t25 + t26 + sin(q1 - q2) * 4.905e+3 -
                sin(q1 + q2) * 3.924e+3 + t3 * t6 * 1.0e+3 - t6 * t13 * 2.0e+2 - t7 * t13 * 1.0e+2 - dq1 * dq2 * t13 * 2.0e+2) - t30 * (
                t5 * t29 + t20 * t29 + t5 * t20 * t27 - 2.0) * (
                t22 + t23 + t25 + t26 + t3 * t6 * 1.0e+2 - t6 * t13 * 1.0e+2)


class Mock(object):
    pass


def create_params(th1d=None, clf_rate=None):
    params = Mock
    # TODO
    # params.beta = [-17.9893,-49.9978,12.9808,-7.7981]
    params.beta = [-17.989285913936719, -49.997811456350071, 12.980805798265429, -7.798078525272227]
    if th1d is not None:
        params.th1d = th1d
    else:
        params.th1d = -0.129982018648464  # 0.13
    if clf_rate is not None:
        params.clf_rate = clf_rate
    else:
        params.clf_rate = 7.822782634523065  # 7.8228
    params.eps = 0.1
    # params.P = np.array([[1.2568, 0.1581], [0.1581, 0.1176]])
    params.P = np.array([[1.256779350685494, 0.158113883008419], [0.158113883008419, 0.117586806357096]])
    return params


def main():
    np.random.seed(args.random_seed)
    utils.setup_data_exp_and_logger(args)

    l = 1.0
    reset_q1_threshold = -0.03
    # (q1, q2, q1d, q2d)
    # x = np.array([[0.286, -0.052, -0.5, 3.8]])
    x = np.array([[0.129982018648464, -0.259943756071354,-0.668838559330795, 0.285101045253290]])
    # x = np.array([[0.119982018648464, -0.259943756071354, -0.968838559330795, 0.785101045253290]])
    xf = np.array([0.3])
    u = np.array([[0.1]])
    x_list=[]
    xf_list=[]
    params = create_params()


    for ti in range(args.nt):
        u = two_link_io_control_clf_qp(x, params)
        xdot = compute_xdot(x, u)
        prev_x = x
        x = x + xdot * args.dt
        print(ti, prev_x, u, xdot, x)
        #TODO switching condition
        curr_2q1_geq_q2 = (-2 * x[0, 0]) >= x[0, 1]
        prev_2q1_lt_q2 = (-2 * prev_x[0, 0]) < prev_x[0, 1]
        curr_q1_leq_thres = x[0, 0] <= reset_q1_threshold
        dq1_dq2_nonpos = (2 * x[0, 2] + x[0, 3]) <= 0
        if curr_2q1_geq_q2 and prev_2q1_lt_q2 and curr_q1_leq_thres and dq1_dq2_nonpos:
            q2 = x[0, 1]
            xf = xf + 2 * l * np.sin(q2 / 2)
            x = compute_impact(x)
            print("change!")

        x_list.append(np.array(x))
        xf_list.append(np.array(xf))
        if ti % args.print_freq == 0:
            plot_sim(x, xf, ti)

    x_np = np.concatenate(x_list, axis=0)
    plt.subplot(4, 1, 1)
    plt.plot(range(x_np.shape[0]), x_np[:, 0])
    plt.subplot(4, 1, 2)
    plt.plot(range(x_np.shape[0]), x_np[:, 1])
    plt.subplot(4, 1, 3)
    plt.plot(range(x_np.shape[0]), x_np[:, 2])
    plt.subplot(4, 1, 4)
    plt.plot(range(x_np.shape[0]), x_np[:, 3])
    plt.savefig("%s/curve.png" % (args.viz_dir), bbox_inches='tight', pad_inches=0.1)
    plt.close()

    os.system("convert -delay 5 -loop 0 %s/sim*.png %s/animation.gif" % (args.viz_dir, args.viz_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="BiP_dbg")
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--nt', type=int, default=500)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--gait', action='store_true', default=False)
    parser.add_argument('--beta', action='store_true', default=False)
    args = parser.parse_args()
    main()