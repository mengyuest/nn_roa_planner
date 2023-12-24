import os
import torch
import casadi
import numpy as np
import matplotlib.pyplot as plt
import time
import cgw_sim_choi as choi


def get_next_x(x, u, dt, opti):
    lL = 1.0
    mL = 1.0
    g = 9.81
    mH = 1.0

    q1 = x[:, 0]
    q2 = x[:, 1]
    dq1 = x[:, 2]
    dq2 = x[:, 3]

    cos = casadi.cos
    sin = casadi.sin

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

    ddq1 = -t12 * t20 * (-t14 + t17 + t18 - g * mH * t3 * 4.0 + g * mL * sin(
        q1 + t9) * 2.0 + lL * mL * t4 * t5 * 2.0 - lL * mL * t5 * t11 * 2.0)
    ddq2 = -t12 * t20 * (
                t13 + t14 - t17 - t18 + g * mH * t2 * t4 * 8.0 + g * mL * t2 * t4 * 1.0e+1 - g * mL * t2 * t11 * 2.0 - g * mL * t3 * t10 * 2.0
                - lL * mH * t4 * t5 * 8.0 - lL * mL * t4 * t5 * 1.2e+1 + lL * mL * t5 * t11 * 4.0 + lL * mL * t6 * t11 * 2.0 - g * mL * t3 * cos(
            q2) * 2.0
                + dq1 * dq2 * lL * mL * t11 * 4.0)

    q2 = x[:, 1]
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

    next_x = opti.variable(1, 4)

    next_x[:, 0] = x[:, 0] + dq1 * dt
    next_x[:, 1] = x[:, 1] + dq2 * dt
    next_x[:, 2] = x[:, 2] + (ddq1 + gvec3 * u[:, 0]) * dt
    next_x[:, 3] = x[:, 3] + (ddq2 + gvec4 * u[:, 0]) * dt

    return next_x


def compute_fine(next_x, prev_x):
    q1_minus = prev_x[0, 0]
    q2_minus = prev_x[0, 1]
    q1_plus = next_x[0, 0]
    q2_plus = next_x[0, 1]

    denom = q2_minus - q2_plus + 2 * q1_minus - 2 * q1_plus
    clip_denom = casadi.if_else(denom==0, 1, denom)
    alpha = (-q2_plus - 2 * q1_plus) / clip_denom
    return alpha * prev_x + (1 - alpha) * next_x


def compute_impact(x, opti):
    cos = casadi.cos
    sin = casadi.sin

    q1_minus = x[0, 0]
    q2_minus = x[0, 1]
    dq1_minus = x[0, 2]
    dq2_minus = x[0, 3]

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

    new_x = opti.variable(1, 4)
    new_x[0, 0] = q1_plus
    new_x[0, 1] = q2_plus
    new_x[0, 2] = dq1_plus
    new_x[0, 3] = dq2_plus

    return new_x


def detect_switch(x, prev_x, reset_q1_threshold):
    l_and = casadi.logic_and

    q1 = x[0, 0]
    q2 = x[0, 1]
    dq1 = x[0, 2]
    dq2 = x[0, 3]
    prev_q1 = prev_x[0, 0]
    prev_q2 = prev_x[0, 1]

    curr_2q1_geq_q2 = -2 * q1 >= q2
    prev_2q1_lt_q2 = -2 * prev_q1 < prev_q2
    curr_q1_leq_thres = q1 <= reset_q1_threshold
    dq1_dq2_leq_0 = (2 * dq1 + dq2) <= 0

    mask_bool = l_and(l_and(curr_2q1_geq_q2, prev_2q1_lt_q2), l_and(curr_q1_leq_thres, dq1_dq2_leq_0))

    return mask_bool


def two_link_io_control_clf_qp(x, params, qp_bound=4.0):
    u_bound = qp_bound
    LgLfy = choi.LgLfy_gen(x, params, use_casadi=True)
    Lf2y = choi.Lf2y_gen(x, params, use_casadi=True)
    v = choi.get_clf_qp_sol(x, params, use_casadi=True)
    u = (1 / LgLfy) * (-Lf2y + v)
    # u = np.clip(u, -u_bound, u_bound)
    # u = u.reshape((1, 1))
    u = casadi.fmax(casadi.fmin(u, u_bound), -u_bound)
    return u


def plan_gait_mpc(x_init, th1d, horizon, dt, num_sim_steps, u_mpc_bound, u_bound, args):
    # x_sim (1, 4)
    # x_target (1, 4)
    x_dim = 4
    simT = horizon
    quiet = True
    mpc_max_iters = args.mpc_max_iters
    reset_q1_threshold = -0.03

    x_init = x_init.detach().cpu().numpy()

    #th1d = x_target[0, 0]
    params = choi.create_params(th1d=-th1d)
    beta1, beta2, beta3, beta4 = params.beta

    opti = casadi.Opti()
    x = opti.variable(simT + 1, x_dim)
    u = opti.variable(simT, 1)
    u_qp = opti.variable(simT, 1)
    # u_out1 = opti.variable(simT, 1)
    u_out2 = opti.variable(simT, 1)
    mode = opti.variable(simT + 1)

    for i in range(x_dim):
        x[0, i] = x_init[0, i]
    opti.set_initial(u, np.zeros((simT, 1)))
    opti.set_initial(mode, np.zeros((simT+1, 1)))

    for ti in range(simT):
        prev_x = x[ti:ti+1, :]
        next_x = x[ti:ti+1, :] * 1.0
        u_qp[ti:ti+1, :] = two_link_io_control_clf_qp(x[ti:ti+1, :], params)
        u_out1 = casadi.fmin(casadi.fmax(u[ti:ti+1, :], -u_mpc_bound), u_mpc_bound)
        u_out2[ti:ti + 1, :] = casadi.fmin(casadi.fmax(u_out1 + u_qp[ti:ti+1, :], -u_bound), u_bound)
        for tti in range(num_sim_steps):
            next_x = get_next_x(next_x, u_out2[ti:ti+1, :], dt/num_sim_steps, opti)
        # x[ti+1:ti+2, :] = next_x
        x_mid = compute_fine(next_x, prev_x)
        x_plus = compute_impact(x_mid, opti)
        mask = detect_switch(next_x, prev_x, reset_q1_threshold)
        x[ti+1:ti+2, :] = casadi.if_else(mask, x_plus, next_x)
        mode[ti+1] = casadi.if_else(casadi.sumsqr(mode[:ti])==0, mask, 0)

    opti.subject_to(casadi.sum1(casadi.sum2(mode[1:])) > 0.5)
    # opti.subject_to(x[1:, 2] < 0)
    # opti.subject_to(u_out2 < u_bound)
    # opti.subject_to(u_out2 > -u_bound)
    # opti.subject_to(x[1:, 0] < 0.25)
    # opti.subject_to(x[1:, 0] > -0.25)
    # loss_gait = casadi.sum1(mode * (x[:, 0] - x_target[0, 0]) * (x[:, 0] - x_target[0, 0])) + \
    #             casadi.sum1(mode * (x[:, 1] - x_target[0, 1]) * (x[:, 1] - x_target[0, 1])) + \
    #             casadi.sum1(mode * (x[:, 2] - x_target[0, 2]) * (x[:, 2] - x_target[0, 2])) + \
    #             casadi.sum1(mode * (x[:, 3] - x_target[0, 3]) * (x[:, 3] - x_target[0, 3]))

    loss_gait = casadi.sum1(mode * (x[:, 0] - th1d) * (x[:, 0] - th1d)) + \
                casadi.sum1(mode * (x[:, 1] + 2 * th1d) * (x[:, 1] + 2 * th1d))

    # curr_q2 = x[:, 1]
    # poly_q2 = x[:, 0] * 2.0 - (x[:, 0] + th1d) * (x[:, 0] - th1d) * \
    #           (beta1 + beta2 * x[:, 0] + beta3 * x[:, 0] * x[:, 0] +
    #            beta4 * x[:, 0] * x[:, 0] * x[:, 0])
    # loss_gait = casadi.sumsqr(curr_q2 - poly_q2)

    opti.minimize(loss_gait)

    def callback_func(i):
        ov = opti.debug.value
        x_np = ov(x)
        u_np = ov(u)
        m_np = ov(mode)

        for ti in range(simT):
            if ti % 10 == 0:
                print("%04d  t:%03d x:%.3f %.3f %.3f %.3f mode:%.3f u:%.3f  " % (
                    i, ti, x_np[ti,0], x_np[ti,1], x_np[ti,2], x_np[ti,3], m_np[ti], u_np[ti]
                ))

    # opti.callback(lambda i: callback_func(i))

    p_opts = {"expand": True}
    s_opts = {"max_iter": mpc_max_iters, "tol": 1e-5}
    if quiet:
        p_opts["print_time"] = 0
        s_opts["print_level"] = 0
        s_opts["sb"] = "yes"
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol1 = opti.solve()
    except RuntimeError as err:
        print("CATCH", err)

    x_np = opti.debug.value(x)
    u_np = opti.debug.value(u)
    u_out2_np = opti.debug.value(u_out2)
    mode_np = opti.debug.value(mode)
    x_tensor = torch.from_numpy(x_np).float().reshape((-1, x_dim))
    u_tensor = torch.from_numpy(u_np).float().reshape((-1, 1))
    u_out2_tensor = torch.from_numpy(u_out2_np).float().reshape((-1, 1))
    mode_tensor = torch.from_numpy(mode_np).float().reshape((-1, 1))
    return u_out2_tensor, u_tensor, x_tensor, mode_tensor



class MockArgs:
    pass


def main():
    t1 = time.time()
    u_out, u, x, mode = plan_gait_mpc(
        # x_init=np.array([[0.15828843, -0.35657686, -0.73985833,  0.61868256]]),
        # x_init=np.array([[0.13051651, -0.26103302, -0.66095102, 0.30043977]]),
        x_init=torch.tensor([[0.120, -0.239, -0.628, 0.230]]).float(),
                         #x_target=np.array([[0.13051651, -0.26103302, -0.66095102, 0.30043977]]),
                        th1d=0.13051651,
                        horizon=100,
                        dt=0.01,
                        num_sim_steps=2,
                        u_mpc_bound=1,
                        u_bound=4,
                         args=None)
    for i in range(x.shape[0]-1):
        print("%03d %.3f %.3f %.3f %.3f | %.1f | %.3f" % (i, x[i, 0], x[i, 1], x[i, 2], x[i, 3], mode[i, 0], u[i, 0]))
    t2 = time.time()
    print("Finished in %.4f seconds"%(t2-t1))

    # real test
    reset_q1_threshold=-0.03
    dt = 0.01
    params = choi.create_params(th1d=-0.13051651)
    l = 1
    x_cache = x
    u_out_cache = u_out
    u_cache = u
    m_cache = mode
    num_sim_steps = 2

    x = x[0:1, :]
    x_list = []
    for ti in range(100):
        prev_x = x
        # u_qp = choi.two_link_io_control_clf_qp(x, use_torch=True, params=params)
        # u_mpc = u_cache[ti:ti+1, :]
        # u_mpc = torch.clamp(u_mpc, -1, 1)
        # u = torch.clamp(u_mpc + u_qp, -4, 4)
        u_out = u_out_cache[ti:ti+1, :]
        u = u_cache[ti:ti+1, :]
        for tti in range(num_sim_steps):
            xdot = choi.compute_xdot(x, u_out, use_torch=True)
            x = x + xdot * (dt / num_sim_steps)

        mock_args = MockArgs()
        mock_args.reset_q1_threshold = reset_q1_threshold

        x_mid = choi.compute_fine(x, prev_x, None, use_torch=True)
        x_plus = choi.compute_impact(x_mid, use_torch=True)
        mask = choi.detect_switch(x, prev_x, mock_args)
        x = x * (1 - mask) + x_plus * mask
        x_list.append(x)

        print("t:%03d %.3f %.3f %.3f %.3f | %.3f %.3f" % (
            ti, prev_x[0, 0], prev_x[0, 1], prev_x[0, 2], prev_x[0, 3], u[0, 0], u_out[0, 0]))




if __name__ == "__main__":
    main()

