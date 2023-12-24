import casadi
import numpy as np
import torch
import car_utils as sscar_utils
import time

STCar = sscar_utils.STCar

def next_x_mpc(x, u, dt, opti, car_params, params):
    # f = opti.variable(1, 7)

    next_x = opti.variable(1, 7)

    # Extract the parameters
    mu_factor = params[..., STCar.MU]
    v_ref = params[..., STCar.VREF]
    omega_ref = params[..., STCar.WREF]
    a_ref = v_ref * 0.0

    # Extract the state variables and adjust for the reference
    v = x[:, STCar.VE] + v_ref
    psi_e = x[:, STCar.PSI_E]
    psi_e_dot = x[:, STCar.PSI_E_DOT]
    psi_dot = psi_e_dot + omega_ref
    beta = x[:, STCar.BETA]
    delta = x[:, STCar.DELTA]
    sxe = x[:, STCar.SXE]
    sye = x[:, STCar.SYE]

    # set gravity constant
    gravity = 9.81  # [m/s^2]

    # create equivalent bicycle parameters
    mu = car_params.tire_p_dy1 * mu_factor
    C_Sf = -car_params.tire_p_ky1 / car_params.tire_p_dy1
    C_Sr = -car_params.tire_p_ky1 / car_params.tire_p_dy1
    lf = car_params.a
    lr = car_params.b
    m = car_params.m
    Iz = car_params.I_z
    ss_h_s = car_params.h_s
    h = ss_h_s

    # We want to express the error in x and y in the reference path frame, so
    # we need to get the dynamics of the rotated global frame error
    dsxe_r = v * np.cos(psi_e + beta) - v_ref + omega_ref * sye
    dsye_r = v * np.sin(psi_e + beta) - omega_ref * sxe

    f_x = dsxe_r
    f_y = dsye_r
    f_delta = 0
    f_v = -a_ref

    use_kinematic_model = v*v < 0.1*0.1

    # Single-track dynamics
    f_psi_normal = psi_e_dot
    # Sorry this is a mess (it's ported from the commonroad models)
    f_psi_dot_normal = (
            -(mu * m / (v * Iz * (lr + lf)))
            * (lf ** 2 * C_Sf * gravity * lr + lr ** 2 * C_Sr * gravity * lf)
            * psi_dot
            + (mu * m / (Iz * (lr + lf)))
            * (lr * C_Sr * gravity * lf - lf * C_Sf * gravity * lr)
            * beta
            + (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * gravity * lr) * delta
    )
    f_beta_normal = (
            (
                    (mu / (v ** 2 * (lr + lf))) * (C_Sr * gravity * lf * lr - C_Sf * gravity * lr * lf)
                    - 1
            )
            * psi_dot
            - (mu / (v * (lr + lf))) * (C_Sr * gravity * lf + C_Sf * gravity * lr) * beta
            + mu / (v * (lr + lf)) * (C_Sf * gravity * lr) * delta
    )

    # Kinematic dynamics
    lwb = lf + lr

    f_psi_km = v[0] * np.cos(beta[0]) / lwb * np.tan(delta[0]) - omega_ref
    f_psi_dot_km = 0
    f_beta_km = 0

    f_psi = casadi.if_else(use_kinematic_model, f_psi_km, f_psi_normal)
    f_psi_dot = casadi.if_else(use_kinematic_model, f_psi_dot_km, f_psi_dot_normal)
    f_beta = casadi.if_else(use_kinematic_model, f_beta_km, f_beta_normal)


    # TODO g
    g_delta_w = 1
    g_v_a = 1
    g_psi_dot_a_normal = (
            -(mu * m / (v * Iz * (lr + lf)))
            * (-(lf ** 2) * C_Sf * h + lr ** 2 * C_Sr * h)
            * psi_dot
            + (mu * m / (Iz * (lr + lf))) * (lr * C_Sr * h + lf * C_Sf * h) * beta
            - (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * h) * delta
    )
    g_beta_a = (
            (mu / (v ** 2 * (lr + lf))) * (C_Sr * h * lr + C_Sf * h * lf) * psi_dot
            - (mu / (v * (lr + lf))) * (C_Sr * h - C_Sf * h) * beta
            - mu / (v * (lr + lf)) * C_Sf * h * delta
    )

    beta_dot = 1 / (1 + (np.tan(delta) * lr / lwb) ** 2) * lr / (lwb * np.cos(delta) ** 2)
    g_beta_w_km = beta_dot
    g_beta_w_normal = 0
    g_psi_dot_a_km = 1 / lwb * (np.cos(beta[0]) * np.tan(delta[0]))
    g_psi_dot_w_km = 1 / lwb * (
                    -v[0] * np.sin(beta[0]) * np.tan(delta[0]) * beta_dot[0]
                    + v[0] * np.cos(beta[0]) / np.cos(delta[0]) ** 2)
    g_psi_dot_w_normal = 0
    g_psi_dot_a = casadi.if_else(use_kinematic_model, g_psi_dot_a_km, g_psi_dot_a_normal)
    g_psi_dot_w = casadi.if_else(use_kinematic_model, g_psi_dot_w_km, g_psi_dot_w_normal)
    g_beta_w = casadi.if_else(use_kinematic_model, g_beta_w_km, g_beta_w_normal)

    next_x[0, 0] = x[0, 0] + f_x * dt
    next_x[0, 1] = x[0, 1] + f_y * dt
    next_x[0, 2] = x[0, 2] + (f_delta + g_delta_w * u[0, 0]) * dt
    next_x[0, 3] = x[0, 3] + (f_v + g_v_a * u[0, 1] ) * dt
    next_x[0, 4] = x[0, 4] + f_psi * dt
    next_x[0, 5] = x[0, 5] + (f_psi_dot + g_psi_dot_w * u[0, 0] + g_psi_dot_a * u[0, 1]) * dt
    next_x[0, 6] = x[0, 6] + (f_beta + g_beta_w * u[0, 0] + g_beta_a * u[0, 1]) * dt

    return next_x


def to_np(x):
    return x.detach().cpu().numpy()

def to_torch(x):
    return torch.from_numpy(x).float().cuda()

def plan_mpc(x_init, ref1_list, ref2_list, params_a, params_b, nt=100, dt=0.01, args=None):
    debug_t1=time.time()
    opti = casadi.Opti()
    mpc_max_iters = 1000
    quiet = True

    x_init = to_np(x_init)
    ref1_list = to_np(ref1_list)
    ref2_list = to_np(ref2_list)
    params_a = to_np(params_a)
    params_b = to_np(params_b)

    debug_t2 = time.time()
    x = opti.variable(nt + 1, 7)
    u = opti.variable(nt, 2)
    for i in range(7):
        x[0, i] = x_init[0, i]
    car_params = sscar_utils.VehicleParameters()

    debug_t3 = time.time()
    for ti in range(nt):
        opti.subject_to(u[ti, 0] <= 12)
        opti.subject_to(u[ti, 0] >= -12)
        opti.subject_to(u[ti, 1] <= 12)
        opti.subject_to(u[ti, 1] >= -12)

        if ti < ref1_list.shape[0]:
            params = params_a
        else:
            params = params_b
            x[ti+1:ti+2, :] = world_to_err_mpc(err_to_world_mpc(x[ti+1:ti+2, :], ref1_list[-1:, :], opti), ref2_list[0:1, :], opti)
        x[ti+1:ti+2, :] = next_x_mpc(x[ti:ti+1, :], u[ti:ti+1, :], dt, opti, car_params, params)
    loss = casadi.sumsqr(x)
    opti.minimize(loss)

    debug_t4 = time.time()

    p_opts = {"expand": True}
    s_opts = {"max_iter": mpc_max_iters, "tol": 1e-5}
    if quiet:
        p_opts["print_time"] = 0
        s_opts["print_level"] = 0
        s_opts["sb"] = "yes"
    opti.solver("ipopt", p_opts, s_opts)

    sol1 = opti.solve()
    x_np = opti.debug.value(x)
    u_np = opti.debug.value(u)

    debug_t5 = time.time()

    return to_torch(x_np), to_torch(u_np)


def err_to_world_mpc(err, ref, opti):
    state = opti.variable(1, 7)
    x_e, y_e, delta_e, v_e, psi_e, psi_dot_e, beta_e = err[0, 0], err[0, 1], err[0, 2], err[0, 3], \
                                                       err[0, 4], err[0, 5], err[0, 6]
    x_ref, y_ref, psi_ref, v_ref = ref[0, 0], ref[0, 1], ref[0, 2], ref[0, 3]

    x = x_ref + x_e * casadi.cos(psi_ref) - y_e * casadi.sin(psi_ref)
    y = y_ref + x_e * casadi.sin(psi_ref) + y_e * casadi.cos(psi_ref)
    delta = 0.0 + delta_e
    v = v_ref + v_e
    psi = psi_ref + psi_e
    psi_dot = 0.0 + psi_dot_e
    beta = 0.0 + beta_e

    state[0, 0] = x
    state[0, 1] = y
    state[0, 2] = delta
    state[0, 3] = v
    state[0, 4] = psi
    state[0, 5] = psi_dot
    state[0, 6] = beta

    return state


def world_to_err_mpc(state, ref, opti):
    err = opti.variable(1, 7)
    x, y, delta, v, psi, psi_dot, beta = state[0, 0], state[0, 1], state[0, 2], state[0, 3], \
                                         state[0, 4], state[0, 5], state[0, 6]
    x_ref, y_ref, psi_ref, v_ref = ref[0, 0], ref[0, 1], ref[0, 2], ref[0, 3]
    x_e = (x - x_ref) * casadi.cos(psi_ref) + (y - y_ref) * casadi.sin(psi_ref)
    y_e = -(x - x_ref) * casadi.sin(psi_ref) + (y - y_ref) * casadi.cos(psi_ref)
    delta_e = delta - 0
    v_e = v - v_ref
    # psi_e = (psi - psi_ref + np.pi) % (np.pi * 2) - np.pi
    psi_e = casadi.if_else(psi - psi_ref >= 0,
                           casadi.mod(psi - psi_ref + np.pi, np.pi * 2) - np.pi,
                           casadi.mod(psi - psi_ref - np.pi, np.pi * 2) + np.pi)

    psi_dot_e = psi_dot - 0.0
    beta_e = beta - 0

    err[0, 0] = x_e
    err[0, 1] = y_e
    err[0, 2] = delta_e
    err[0, 3] = v_e
    err[0, 4] = psi_e
    err[0, 5] = psi_dot_e
    err[0, 6] = beta_e
    return err


def main():
    opti = casadi.Opti()
    mpc_max_iters = 1000
    quiet = True
    simT=100
    dt=0.01
    x = opti.variable(simT+1, 7)
    u = opti.variable(simT, 2)
    x_init = np.ones((1, 7)) * 0.5
    for i in range(7):
        x[0, i] = x_init[0, i]

    # (mu, vref, wref, 0)
    car_params = sscar_utils.VehicleParameters()
    params = np.array([1.0, 5.0, 0.0, 0.0])

    for ti in range(simT):
        opti.subject_to(u[ti, 0] <= 12)
        opti.subject_to(u[ti, 0] >= -12)
        opti.subject_to(u[ti, 1] <= 12)
        opti.subject_to(u[ti, 1] >= -12)

        x[ti+1:ti+2,:] = next_x_mpc(x[ti:ti+1, :], u[ti:ti+1, :], dt, opti, car_params, params)

    loss = casadi.sumsqr(x)
    opti.minimize(loss)

    p_opts = {"expand": True}
    s_opts = {"max_iter": mpc_max_iters, "tol": 1e-5}
    if quiet:
        p_opts["print_time"] = 0
        s_opts["print_level"] = 0
        s_opts["sb"] = "yes"
    opti.solver("ipopt", p_opts, s_opts)

    sol1 = opti.solve()

    x_np = opti.debug.value(x)
    u_np = opti.debug.value(u)
    for i in range(simT):
        print("%4d  x %.3f   y %.3f   th %.3f   v %.3f   psi %.3f   psid %.3f  beta:%.3f  w:%.3f a:%.3f" % (
            i, x_np[i, 0], x_np[i, 1], x_np[i, 2], x_np[i, 3], x_np[i, 4], x_np[i, 5], x_np[i, 6],
            u_np[i, 0], u_np[i, 1]
        ))


if __name__ == "__main__":
    main()