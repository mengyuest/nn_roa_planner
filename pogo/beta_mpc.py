import os
import torch
import casadi
import numpy as np
import matplotlib.pyplot as plt
import beta_utils


def f_f_new_b(s, ti, u, ui, opti, global_vars, extra_vars):
    num_steps, dt, dt2, l0, g, k, M = global_vars
    delta_x, floor0, ceil0, ref_v0, floor1, ceil1, ref_v1 = extra_vars
    new_s = opti.variable(8)
    new_s[0], new_s[1], new_s[2], new_s[3] = \
        s[ti, 0] + s[ti, 1] * dt, \
        s[ti, 1], \
        s[ti, 2] + s[ti, 3] * dt, \
        s[ti, 3] + (-g) * dt

    # foot-x placement
    new_s[4] = new_s[0] + l0 * casadi.sin(u[ui, 0])

    # jump case
    new_s[7] = casadi.if_else(new_s[4] > delta_x, 1, 0)
    new_s[2] = casadi.if_else(
        casadi.logic_and(new_s[7] == 1, s[ti, 7] == 0),
        new_s[2] + floor0 - floor1, new_s[2])

    # foot-y placement
    new_s[5] = new_s[2] - l0 * casadi.cos(u[ui, 0])

    # mode switch
    new_s[6] = casadi.if_else(casadi.logic_or(s[ti, 6]==0, s[ti, 6]==5),
        casadi.if_else(s[ti, 6]==0,
                       casadi.if_else(new_s[5] > 0, 0, 1),
                       6),
        casadi.if_else(s[ti, 6]==6,
                       casadi.if_else(new_s[3] <= 0, 7, 6),
                       0),
    )
    new_s[4] = casadi.if_else(new_s[6] == 1, new_s[0] + l0 * casadi.sin(u[ui, 0]), new_s[4])
    new_s[5] = casadi.if_else(new_s[6] == 1, new_s[5] * 0, new_s[5])
    return new_s


def s_f_new_b(s, ti, u, ui, opti, global_vars, extra_vars):
    num_steps, dt, dt2, l0, g, k, M = global_vars
    new_s = opti.variable(8)
    l = ((s[ti, 0]-s[ti, 4]) * (s[ti, 0] - s[ti, 4]) + s[ti, 2]*s[ti, 2]) ** 0.5
    new_s[0], new_s[1], new_s[2], new_s[3] = \
        s[ti, 0] + s[ti, 1] * dt2, \
        s[ti, 1] + (1 / M * (s[ti, 0] - s[ti, 4]) / l * (u[ui, 1] + k * (l0 - l))) * dt2, \
        s[ti, 2] + s[ti, 3] * dt2, \
        s[ti, 3] + (1 / M * s[ti, 2] / l * (u[ui, 1] + k * (l0 - l)) - g) * dt2
    new_s[4] = s[ti, 4]
    new_s[5] = s[ti, 5]
    new_s[6] = casadi.if_else(casadi.logic_or(s[ti, 6]==2, s[ti, 6]==4),
                  casadi.if_else(
                       s[ti, 6]==2,
                       casadi.if_else(new_s[3]>=0, 3, 2),
                       casadi.if_else((new_s[4]-new_s[0])**2 + (new_s[5]-new_s[2])**2 - l0*l0 > 0, 5, 4)),
                  casadi.if_else(s[ti, 6]==1, 2, 4)  # 1->2, 3->4
              )
    new_s[7] = s[ti, 7]
    return new_s


def plan_waypoint_mpc(x_sim, offset, seg_list, seg_i, global_vars, extra_vars, args):
    simT = 3000
    quiet = True
    n_check = 500
    print_freq = 100
    mpc_max_iters = args.mpc_max_iters
    num_steps, dt, dt2, l0, g, k, M = global_vars
    dt = 1 * dt
    dt2 = 1 * dt2
    global_vars1 = num_steps, dt, dt2, l0, g, k, M

    delta_x, floor0, ceil0, ref_v0, floor1, ceil1, ref_v1 = extra_vars
    ref_h0 = beta_utils.ref_h(floor0, ceil0)
    ref_h1 = beta_utils.ref_h(floor1, ceil1)

    dx_0 = x_sim[0, 0].detach().item() - (delta_x - 5)
    dx_1 = delta_x - x_sim[0, 0].detach().item()

    ref_hh = (ref_h0 * dx_1 + ref_h1 * dx_0)/(dx_0+dx_1)

    x_dim = 8

    opti = casadi.Opti()
    x = opti.variable(simT+1, x_dim)  # x, xd, y, yd, xf, yf, mode, offset-bin
    flight_mode = opti.variable(simT+1)
    apex_mode = opti.variable(simT+1)
    u = opti.variable(1, 2)  # th, F
    gamma = opti.variable(1, 4)
    x_init = np.array([x_sim[0, 0].detach().item(), x_sim[0, 1].detach().item(),
                       x_sim[0, 2].detach().item(), x_sim[0, 3].detach().item(),
                       x_sim[0, 0].detach().item(), x_sim[0, 2].detach().item() - l0,
                       x_sim[0, 6].detach().item(), 0])


    # set initial values
    for i in range(x_dim):
        x[0, i] = x_init[i]

    heur = (x_init[1] - ref_v0) + (x_init[2] - ref_hh)

    init_u = 0

    opti.set_initial(u, np.array([[0, init_u]]))

    opti.subject_to(u[0, 0] <= 0.5)
    opti.subject_to(u[0, 0] >= -0.5)
    opti.subject_to(u[0, 1] <= 5000)
    opti.subject_to(u[0, 1] >= -5000)

    new_x_list=[x]

    for ti in range(simT):
        # print(ti)
        x[ti+1,:] = casadi.if_else(
            casadi.logic_or(x[ti, 6]==0, x[ti, 6]>=5),
            f_f_new_b(x, ti, u, 0, opti, global_vars1, extra_vars),
            s_f_new_b(x, ti, u, 0, opti, global_vars1, extra_vars))
        new_x_list.append(x[ti+1])
        # loss_vref = loss_vref + casadi.if_else(new_x[6]==0, (new_x[1]-sel_vref)**2, 0)
    for ti in range(simT+1):
        flight_mode[ti] = casadi.if_else(casadi.logic_or(x[ti, 6] == 0, x[ti, 6] >= 5), 1, 0)
        apex_mode[ti] = casadi.if_else(x[ti, 6] == 7, 1, 0)

    apex_h = casadi.sum1(casadi.sum2(apex_mode * x[:,2]))
    apex_stg = casadi.sum1(casadi.sum2(apex_mode * x[:,7]))
    loss_ceil = (apex_h + floor0 - ref_hh)**2 * (1-apex_stg) + (apex_h + floor1 - ref_h1)**2 * apex_stg
    loss_ceil = loss_ceil * 100

    loss_vref = casadi.sumsqr((x[:, 1] - ref_v0) * (1 - x[:, 7]) * flight_mode + (x[:, 1] - ref_v1) * x[:, 7] * flight_mode)

    opti.minimize(loss_ceil + loss_vref + 10 * casadi.sumsqr(gamma))


    def callback_func(i):
        ov = opti.debug.value
        x_np = ov(x)
        u_np = ov(u)
        idx = np.where(x_np[:,6]==7)
        flight_mode_np = ov(flight_mode)
        maxh1 = np.max((x_np[:,2] + floor0) * (1 - x_np[:, 7]) * flight_mode_np)
        maxh2 = np.max((x_np[:, 2] + floor1) * (x_np[:, 7]) * flight_mode_np)
        vio_0 = ov((x[:, 2] + floor0 - ceil0 - gamma[0, 0] ** 2) * (1 - x[:, 7]) * flight_mode)
        vio_1 = ov((x[:, 2] + floor1 - ceil1 - gamma[0, 1] ** 2) * x[:, 7] * flight_mode)
        vio_2 = ov((x[:, 5] + gamma[0, 2] ** 2) * (1 - x[:, 7]) * flight_mode)
        vio_3 = ov((x[:, 5] + gamma[0, 3] ** 2) * x[:, 7] * flight_mode)

        print("%04d  F:%.3f C:%.3f  u:%.3f  %.3f  ap_h:%.3f  maxH:%.3f %.3f ap_v%.3f  ap_x:%.3f  mode:%d  loss_ceil:%.3f  loss1_vref:%.3f loss_gamma:%.3f "
              "%.2f %.2f %.2f %.2f" % (
            i, floor0, ceil0, u_np[0], u_np[1], x_np[idx, 2], maxh1, maxh2, x_np[idx, 1], x_np[idx,0], x_np[idx,6],
            ov(loss_ceil), ov(loss_vref), ov(10 * casadi.sumsqr(gamma)), np.max(vio_0), np.max(vio_1), np.min(vio_2), np.min(vio_3),
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
    fli_np = opti.debug.value(flight_mode)

    u_np = opti.debug.value(u)
    if len(u_np.shape)==1:
        u_np = np.reshape(u_np, (1, -1))

    u_tensor = torch.from_numpy(u_np).float()
    return u_tensor

if __name__ == "__main__":
    import torch
    import beta_pogo_sim
    import time
    viz_freq = 100
    num_steps = 5000
    dt1 = 0.001
    dt2 = 0.0001
    l0 = 1.0
    g = 10.0
    k = 32000
    M = 80

    delta_x = 1000
    floor0 = 0
    ceil0 = 4
    floor1 = 0
    ceil1 = 4
    ref_v0 = 2.0
    ref_v1 = 2.0
    extra_vars = delta_x, floor0, ceil0, ref_v0, floor1, ceil1, ref_v1

    global_vars = num_steps, dt1, dt2, l0, g, k, M
    # mode (x, xd, y, yd, xf, yf, mode)
    x_sim = np.array([[0, 1.8, 1.5, 0, 0, 0.8, 0]])
    x_sim = torch.from_numpy(x_sim).float()
    traj_list=[x_sim]
    for i in range(3):
        global_vars = num_steps, dt1, dt2, l0, g, k, M
        t1 = time.time()
        u_sim = plan_waypoint_mpc(x_sim, None, None, None, global_vars, extra_vars, None)
        t2 = time.time()
        print("MPC took %.4f seconds"%(t2-t1))
        # u_sim = torch.from_numpy(u).float()
        for ti in range(num_steps):
            next_x, next_modes = beta_pogo_sim.pytorch_sim_single(x_sim[:, :6], x_sim[:, 6:], u_sim, global_vars, check_invalid=True)
            print(ti, x_sim)
            x_sim_seg = torch.cat([next_x, next_modes], dim=-1)
            traj_list.append(x_sim_seg)
            if x_sim_seg[0, 6] == 7:
                x_sim_seg[0, 6] = 0
                x_sim = x_sim_seg
                break
            x_sim = x_sim_seg

    traj_list = torch.cat(traj_list, dim=0)
    os.makedirs("bak/dbg", exist_ok=True)
    for ti in range(traj_list.shape[0]):
        if ti % viz_freq == 0:
            s = traj_list[ti]
            plt.plot([s[0], s[4]], [s[2], s[5]], linewidth=2)
            plt.plot(s[0], s[2], 'ro', markersize=4)
            plt.plot(s[4], s[5], 'ro', markersize=4)
            plt.xlim(-1, 10.0)
            plt.ylim(-0.5, 2.5)
            plt.xlabel("y (m)")
            plt.ylabel("z (m)")
            plt.tight_layout()
            plt.savefig("bak/dbg/t%04d.png" % (ti), pad_inches=0.5)
            plt.close()

