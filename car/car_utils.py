import numpy as np
import scipy.linalg
import torch
import time

class STCar:
    # Number of states and controls
    N_DIMS = 7
    N_CONTROLS = 2

    # State indices
    SXE = 0
    SYE = 1
    DELTA = 2
    VE = 3
    PSI_E = 4
    PSI_E_DOT = 5
    BETA = 6
    # Control indices
    VDELTA = 0
    ALONG = 1

    # Reference indices
    MU = 0
    VREF = 1
    WREF = 2


class VehicleParameters(object):
    """A restricted set of the commonroad vehicle parameters, which can be
    found as forked at https://github.com/dawsonc/commonroad-vehicle-models"""

    def __init__(self):
        super(VehicleParameters, self).__init__()

        self.steering_min = -1.066  # minimum steering angle [rad]
        self.steering_max = 1.066  # maximum steering angle [rad]
        self.steering_v_min = -0.4  # minimum steering velocity [rad/s]
        self.steering_v_max = 0.4  # maximum steering velocity [rad/s]

        self.longitudinal_a_max = 11.5  # maximum absolute acceleration [m/s^2]

        self.tire_p_dy1 = 1.0489  # Lateral friction Muy
        self.tire_p_ky1 = -21.92  # Maximum value of stiffness Kfy/Fznom

        # distance from spring mass center of gravity to front axle [m]  LENA
        self.a = 0.3048 * 3.793293
        # distance from spring mass center of gravity to rear axle [m]  LENB
        self.b = 0.3048 * 4.667707
        self.h_s = 0.3048 * 2.01355  # M_s center of gravity above ground [m]  HS
        self.m = 4.4482216152605 / 0.3048 * (74.91452)  # vehicle mass [kg]  MASS
        # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
        self.I_z = 4.4482216152605 * 0.3048 * (1321.416)

def _f(x, car_params, params):
    """
    Return the control-independent part of the control-affine dynamics.
    args:
        x: bs x self.n_dims tensor of state
        params: a dictionary giving the parameter values for the system. If None,
                default to the nominal parameters used at initialization
    returns:
        f: bs x self.n_dims x 1 tensor
    """
    # Extract batch size and set up a tensor for holding the result
    batch_size = x.shape[0]
    f = torch.zeros((batch_size, STCar.N_DIMS, 1))
    f = f.type_as(x)

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
    g = 9.81  # [m/s^2]

    # create equivalent bicycle parameters
    mu = car_params.tire_p_dy1 * mu_factor
    C_Sf = -car_params.tire_p_ky1 / car_params.tire_p_dy1
    C_Sr = -car_params.tire_p_ky1 / car_params.tire_p_dy1
    lf = car_params.a
    lr = car_params.b
    m = car_params.m
    Iz = car_params.I_z

    # We want to express the error in x and y in the reference path frame, so
    # we need to get the dynamics of the rotated global frame error
    dsxe_r = v * torch.cos(psi_e + beta) - v_ref + omega_ref * sye
    dsye_r = v * torch.sin(psi_e + beta) - omega_ref * sxe

    f[:, STCar.SXE, 0] = dsxe_r
    f[:, STCar.SYE, 0] = dsye_r
    f[:, STCar.VE, 0] = -a_ref
    f[:, STCar.DELTA, 0] = 0.0

    # Use the single-track dynamics if the speed is high enough, otherwise fall back
    # to the kinematic model (since single-track becomes singular for small v)
    use_kinematic_model = v.abs() < 0.1

    # Single-track dynamics
    f[:, STCar.PSI_E, 0] = psi_e_dot
    # Sorry this is a mess (it's ported from the commonroad models)
    f[:, STCar.PSI_E_DOT, 0] = (
            -(mu * m / (v * Iz * (lr + lf)))
            * (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
            * psi_dot
            + (mu * m / (Iz * (lr + lf)))
            * (lr * C_Sr * g * lf - lf * C_Sf * g * lr)
            * beta
            + (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * g * lr) * delta
    )
    f[:, STCar.BETA, 0] = (
            (
                    (mu / (v ** 2 * (lr + lf))) * (C_Sr * g * lf * lr - C_Sf * g * lr * lf)
                    - 1
            )
            * psi_dot
            - (mu / (v * (lr + lf))) * (C_Sr * g * lf + C_Sf * g * lr) * beta
            + mu / (v * (lr + lf)) * (C_Sf * g * lr) * delta
    )

    # Kinematic dynamics
    lwb = lf + lr
    km = use_kinematic_model
    if len(params.shape)>1:
        f[km, STCar.PSI_E, 0] = (
                v[km] * torch.cos(beta[km]) / lwb * torch.tan(delta[km]) - omega_ref[km]
        )
    else:
        f[km, STCar.PSI_E, 0] = (
                v[km] * torch.cos(beta[km]) / lwb * torch.tan(delta[km]) - omega_ref
        )
    f[km, STCar.PSI_E_DOT, 0] = 0.0
    f[km, STCar.BETA, 0] = 0.0

    return f

def _g(x, car_params, params):
    ss_tire_p_dy1 = car_params.tire_p_dy1
    ss_tire_p_ky1 = car_params.tire_p_ky1
    ss_a = car_params.a
    ss_b = car_params.b
    ss_m = car_params.m
    ss_I_z = car_params.I_z
    ss_h_s = car_params.h_s

    batch_size = x.shape[0]
    g = torch.zeros((batch_size, STCar.N_DIMS, STCar.N_CONTROLS))
    g = g.type_as(x)

    # Extract the parameters
    v_ref = params[..., STCar.VREF]
    omega_ref = params[..., STCar.WREF]
    mu_factor = params[..., STCar.MU]

    # Extract the state variables and adjust for the reference
    v = x[:, STCar.VE] + v_ref
    psi_e_dot = x[:, STCar.PSI_E_DOT]
    psi_dot = psi_e_dot + omega_ref
    beta = x[:, STCar.BETA]
    delta = x[:, STCar.DELTA]

    # create equivalent bicycle parameters
    mu = ss_tire_p_dy1 * mu_factor
    C_Sf = -ss_tire_p_ky1 / ss_tire_p_dy1
    C_Sr = -ss_tire_p_ky1 / ss_tire_p_dy1
    lf = ss_a
    lr = ss_b
    h = ss_h_s
    m = ss_m
    Iz = ss_I_z

    # Use the single-track dynamics if the speed is high enough, otherwise fall back
    # to the kinematic model (since single-track becomes singular for small v)
    use_kinematic_model = v.abs() < 0.1
    # Single-track dynamics
    g[:, STCar.DELTA, STCar.VDELTA] = 1.0
    g[:, STCar.VE, STCar.ALONG] = 1.0

    g[:, STCar.PSI_E_DOT, STCar.ALONG] = (
            -(mu * m / (v * Iz * (lr + lf)))
            * (-(lf ** 2) * C_Sf * h + lr ** 2 * C_Sr * h)
            * psi_dot
            + (mu * m / (Iz * (lr + lf))) * (lr * C_Sr * h + lf * C_Sf * h) * beta
            - (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * h) * delta
    )
    g[:, STCar.BETA, STCar.ALONG] = (
            (mu / (v ** 2 * (lr + lf))) * (C_Sr * h * lr + C_Sf * h * lf) * psi_dot
            - (mu / (v * (lr + lf))) * (C_Sr * h - C_Sf * h) * beta
            - mu / (v * (lr + lf)) * C_Sf * h * delta
    )

    # Kinematic dynamics
    lwb = lf + lr
    km = use_kinematic_model
    beta_dot = (
            1
            / (1 + (torch.tan(delta) * lr / lwb) ** 2)
            * lr
            / (lwb * torch.cos(delta) ** 2)
    )
    g[km, STCar.PSI_E_DOT, STCar.ALONG] = (
            1 / lwb * (torch.cos(beta[km]) * torch.tan(delta[km]))
    )
    g[km, STCar.PSI_E_DOT, STCar.VDELTA] = (
            1
            / lwb
            * (
                    -v[km] * torch.sin(beta[km]) * torch.tan(delta[km]) * beta_dot[km]
                    + v[km] * torch.cos(beta[km]) / torch.cos(delta[km]) ** 2
            )
    )
    g[km, STCar.BETA, 0] = beta_dot[km]

    return g


def compute_Act(x0, car_params, params, args):
    g = 9.81  # [m/s^2]

    ss_tire_p_dy1 = car_params.tire_p_dy1
    ss_tire_p_ky1 = car_params.tire_p_ky1
    ss_a = car_params.a
    ss_b = car_params.b
    ss_m = car_params.m
    ss_I_z = car_params.I_z
    ss_h_s = car_params.h_s

    mu_factor = params[STCar.MU].detach().cpu().item()
    v_ref = params[STCar.VREF].detach().cpu().item()
    w_ref = params[STCar.WREF].detach().cpu().item()

    mu = ss_tire_p_dy1 * mu_factor
    C_Sf = -ss_tire_p_ky1 / ss_tire_p_dy1
    C_Sr = -ss_tire_p_ky1 / ss_tire_p_dy1
    lf = ss_a
    lr = ss_b
    m = ss_m
    Iz = ss_I_z

    A = np.zeros((STCar.N_DIMS, STCar.N_DIMS))

    A[STCar.SXE, STCar.VE] = 1.0
    A[STCar.SXE, STCar.SYE] = w_ref
    A[STCar.SYE, STCar.SXE] = -w_ref
    A[STCar.SYE, STCar.PSI_E] = v_ref
    A[STCar.SYE, STCar.BETA] = v_ref

    A[STCar.PSI_E, STCar.PSI_E_DOT] = 1.0

    A[STCar.PSI_E_DOT, STCar.VE] = (
            (mu * m / (v_ref ** 2 * Iz * (lr + lf)))
            * (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
            * w_ref
    )
    A[STCar.PSI_E_DOT, STCar.PSI_E_DOT] = -(
            mu * m / (v_ref * Iz * (lr + lf))
    ) * (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
    A[STCar.PSI_E_DOT, STCar.BETA] = +(mu * m / (Iz * (lr + lf))) * (
            lr * C_Sr * g * lf - lf * C_Sf * g * lr
    )
    A[STCar.PSI_E_DOT, STCar.DELTA] = (mu * m / (Iz * (lr + lf))) * (
            lf * C_Sf * g * lr
    )

    A[STCar.BETA, STCar.VE] = (
            -2
            * (mu / (v_ref ** 3 * (lr + lf)))
            * (C_Sr * g * lf * lr - C_Sf * g * lr * lf)
            * w_ref
            - mu
            / (v_ref ** 2 * (lr + lf))
            * (C_Sf * g * lr)
            * x0[0, STCar.DELTA]
    )
    A[STCar.BETA, STCar.PSI_E_DOT] = (mu / (v_ref ** 2 * (lr + lf))) * (
            C_Sr * g * lf * lr - C_Sf * g * lr * lf
    ) - 1
    A[STCar.BETA, STCar.BETA] = -(mu / (v_ref * (lr + lf))) * (
            C_Sr * g * lf + C_Sf * g * lr
    )
    A[STCar.BETA, STCar.DELTA] = (
            mu / (v_ref * (lr + lf)) * (C_Sf * g * lr)
    )
    return A

def compute_Act_b(x0, car_params, params, args):
    g = 9.81  # [m/s^2]

    ss_tire_p_dy1 = car_params.tire_p_dy1
    ss_tire_p_ky1 = car_params.tire_p_ky1
    ss_a = car_params.a
    ss_b = car_params.b
    ss_m = car_params.m
    ss_I_z = car_params.I_z
    ss_h_s = car_params.h_s

    mu_factor = params[..., STCar.MU].detach().cpu().numpy()
    v_ref = params[..., STCar.VREF].detach().cpu().numpy()
    w_ref = params[..., STCar.WREF].detach().cpu().numpy()

    mu = ss_tire_p_dy1 * mu_factor
    C_Sf = -ss_tire_p_ky1 / ss_tire_p_dy1
    C_Sr = -ss_tire_p_ky1 / ss_tire_p_dy1
    lf = ss_a
    lr = ss_b
    m = ss_m
    Iz = ss_I_z

    A = np.zeros((params.shape[0], STCar.N_DIMS, STCar.N_DIMS))

    A[:, STCar.SXE, STCar.VE] = 1.0
    A[:, STCar.SXE, STCar.SYE] = w_ref
    A[:, STCar.SYE, STCar.SXE] = -w_ref
    A[:, STCar.SYE, STCar.PSI_E] = v_ref
    A[:, STCar.SYE, STCar.BETA] = v_ref

    A[:, STCar.PSI_E, STCar.PSI_E_DOT] = 1.0

    A[:, STCar.PSI_E_DOT, STCar.VE] = (
            (mu * m / (v_ref ** 2 * Iz * (lr + lf)))
            * (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
            * w_ref
    )
    A[:, STCar.PSI_E_DOT, STCar.PSI_E_DOT] = -(
            mu * m / (v_ref * Iz * (lr + lf))
    ) * (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
    A[:, STCar.PSI_E_DOT, STCar.BETA] = +(mu * m / (Iz * (lr + lf))) * (
            lr * C_Sr * g * lf - lf * C_Sf * g * lr
    )
    A[:, STCar.PSI_E_DOT, STCar.DELTA] = (mu * m / (Iz * (lr + lf))) * (
            lf * C_Sf * g * lr
    )

    A[:, STCar.BETA, STCar.VE] = (
            -2
            * (mu / (v_ref ** 3 * (lr + lf)))
            * (C_Sr * g * lf * lr - C_Sf * g * lr * lf)
            * w_ref
            - mu
            / (v_ref ** 2 * (lr + lf))
            * (C_Sf * g * lr)
            * x0[:, STCar.DELTA]
    )
    A[:, STCar.BETA, STCar.PSI_E_DOT] = (mu / (v_ref ** 2 * (lr + lf))) * (
            C_Sr * g * lf * lr - C_Sf * g * lr * lf
    ) - 1
    A[:, STCar.BETA, STCar.BETA] = -(mu / (v_ref * (lr + lf))) * (
            C_Sr * g * lf + C_Sf * g * lr
    )
    A[:, STCar.BETA, STCar.DELTA] = (
            mu / (v_ref * (lr + lf)) * (C_Sf * g * lr)
    )
    return A


def compute_Bct(x0, car_params, params, args):
    return _g(x0, car_params, params).squeeze().cpu().numpy()

def compute_Bct_b(x0, car_params, params, args):
    return _g(x0, car_params, params).cpu().numpy()


def yue_compute_K(x0, car_params, params, args):
    Act = compute_Act(x0, car_params, params, args)
    Bct = compute_Bct(x0, car_params, params, args)

    A = np.eye(STCar.N_DIMS) + args.controller_dt * Act
    B = args.controller_dt * Bct

    # Define cost matrices as identity
    Q = np.eye(STCar.N_DIMS)
    R = np.eye(STCar.N_CONTROLS)

    # Get feedback matrix
    K = torch.tensor(lqr(A, B, Q, R))
    return K

def _yue_xdot(x, car_params, params, u_all):
    dbg_t0=time.time()
    dbg_t1 = time.time()

    # Extract the parameters
    mu_factor = params[STCar.MU]
    v_ref = params[STCar.VREF]
    omega_ref = params[STCar.WREF]
    a_ref = v_ref * 0.0

    mu = car_params.tire_p_dy1 * mu_factor
    C_Sf = -car_params.tire_p_ky1 / car_params.tire_p_dy1
    C_Sr = -car_params.tire_p_ky1 / car_params.tire_p_dy1
    lf = car_params.a
    lr = car_params.b
    m = car_params.m
    Iz = car_params.I_z
    h = car_params.h_s

    dbg_t2 = time.time()

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
    g = 9.81  # [m/s^2]

    dbg_t3 = time.time()

    # PSI_E_DOT_F_TERM
    psi_e_dot_f_term = -(mu * m / (v * Iz * (lr + lf))) * (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf) * psi_dot\
                       + (mu * m / (Iz * (lr + lf))) * (lr * C_Sr * g * lf - lf * C_Sf * g * lr) * beta\
                       + (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * g * lr) * delta
    # PSI_E_DOT_G_TERM
    psi_e_dot_g_term = -(mu * m / (v * Iz * (lr + lf))) * (-(lf ** 2) * C_Sf * h + lr ** 2 * C_Sr * h) * psi_dot\
                       + (mu * m / (Iz * (lr + lf))) * (lr * C_Sr * h + lf * C_Sf * h) * beta\
                       - (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * h) * delta

    # BETA_F_TERM
    beta_f_term = ((mu / (v ** 2 * (lr + lf))) * (C_Sr * g * lf * lr - C_Sf * g * lr * lf) - 1) * psi_dot \
                  - (mu / (v * (lr + lf))) * (C_Sr * g * lf + C_Sf * g * lr) * beta\
                  + mu / (v * (lr + lf)) * (C_Sf * g * lr) * delta

    # BETA_G_TERM
    beta_g_term = (mu / (v ** 2 * (lr + lf))) * (C_Sr * h * lr + C_Sf * h * lf) * psi_dot\
                  - (mu / (v * (lr + lf))) * (C_Sr * h - C_Sf * h) * beta\
                  - mu / (v * (lr + lf)) * C_Sf * h * delta

    dbg_t4 = time.time()

    xdot_f = torch.stack(
        [
            v * torch.cos(psi_e + beta) - v_ref + omega_ref * sye,
            v * torch.sin(psi_e + beta) - omega_ref * sxe,
            0.0 * v,
            -a_ref * torch.ones_like(v),
            psi_e_dot,
            psi_e_dot_f_term,
            beta_f_term
        ], axis=-1
    )

    use_kinematic_model = v.abs() < 0.1
    km = use_kinematic_model
    lwb = lf + lr
    xdot_f[km, STCar.PSI_E] = \
        v[km] * torch.cos(beta[km]) / lwb * torch.tan(delta[km]) - omega_ref
    xdot_f[km, STCar.PSI_E_DOT] = 0.0
    xdot_f[km, STCar.BETA] = 0.0

    dbg_t5 = time.time()

    xdot_gu = torch.stack(
        [
            v * 0.0,
            v * 0.0,
            u_all[:, 0, 0],
            u_all[:, 1, 0],
            v * 0.0,
            psi_e_dot_g_term * u_all[:, 1, 0],
            beta_g_term * u_all[:, 1, 0],
        ], axis=-1
    )

    beta_dot = (
            1
            / (1 + (torch.tan(delta) * lr / lwb) ** 2)
            * lr
            / (lwb * torch.cos(delta) ** 2)
    )

    xdot_gu[km, STCar.PSI_E_DOT] = u_all[km, 1, 0] / lwb * (torch.cos(beta[km]) * torch.tan(delta[km])) + \
                                   u_all[km, 0, 0] / lwb * (-v[km] * torch.sin(beta[km]) * torch.tan(delta[km]) * beta_dot[km]
                                                        + v[km] * torch.cos(beta[km]) / torch.cos(delta[km]) ** 2)
    xdot_gu[km, STCar.BETA] = beta_dot[km] * u_all[km, 0, 0]

    dbg_t6 = time.time()

    # print("T: %.4f %.4f %.4f %.4f %.4f %.4f"%(dbg_t1 - dbg_t0, dbg_t2 - dbg_t1, dbg_t3 - dbg_t2, dbg_t4 - dbg_t3, dbg_t5 - dbg_t4, dbg_t6 - dbg_t5))

    return xdot_f + xdot_gu


def compute_x0_from_s(s, car_params, params, args, K_hist, pre_K=None, pure_s=False):
    g = 9.81  # [m/s^2]
    ss_tire_p_dy1 = car_params.tire_p_dy1
    ss_tire_p_ky1 = car_params.tire_p_ky1
    ss_a = car_params.a
    ss_b = car_params.b
    ss_m = car_params.m
    ss_I_z = car_params.I_z
    ss_h_s = car_params.h_s

    mu_factor = params[..., STCar.MU]
    v_ref = params[..., STCar.VREF]
    w_ref = params[..., STCar.WREF]

    mu = ss_tire_p_dy1 * mu_factor
    C_Sf = -ss_tire_p_ky1 / ss_tire_p_dy1
    C_Sr = -ss_tire_p_ky1 / ss_tire_p_dy1
    lf = ss_a
    lr = ss_b
    m = ss_m
    Iz = ss_I_z
    if len(params.shape)>1:
        x0 = torch.zeros_like(s)
    else:
        if pure_s:
            x0 = torch.zeros_like(s[0:1])
        else:
            x0 = torch.zeros_like(s[0:1, :-1])
    x0[:, STCar.PSI_E_DOT] = w_ref
    x0[:, STCar.DELTA] = (
            (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
            / (lf * C_Sf * g * lr)
            * w_ref
            / v_ref
    )
    x0[:, STCar.DELTA] /= lf * C_Sf * g * lr
    return x0


def cal_lqr(x, car_params, params, K, args):
    x0 = compute_x0_from_s(x, car_params, params, args=None, K_hist=None, pre_K=None, pure_s=True)
    # x0 = x0.type_as(x)
    # u = -(K @ (x.detach() - x0).T).T
    u = torch.bmm(-K, (x.detach() - x0).unsqueeze(-1)).squeeze(-1)
    assert u.shape[1] == 2
    u = torch.stack([
        torch.clamp(u[:, 0], min=-args.tanh_w_gain, max=args.tanh_w_gain),
        torch.clamp(u[:, 1], min=-args.tanh_a_gain, max=args.tanh_a_gain)
    ], dim=-1)
    return u


def ss_car_lqr(s, car_params, params, args, K_hist, pre_K=None, pure_s=False, return_K=False):
    x0 = compute_x0_from_s(s, car_params, params, args, K_hist, pre_K=None, pure_s=pure_s)
    K_key = (params[STCar.MU].detach().cpu().item(),
             params[STCar.VREF].detach().cpu().item(),
             params[STCar.WREF].detach().cpu().item())

    if K_key in K_hist:
        K = K_hist[K_key]
    else:
        K = yue_compute_K(x0, car_params, params, args)
        K_hist[K_key] = K

    if pure_s==False:
        x = s[:, :-1]
    else:
        x = s

    x0 = x0.type_as(x)
    u = -(K.type_as(x) @ (x.detach() - x0).T).T

    upper_limit = torch.tensor(
        [
            args.tanh_w_gain,  # self.car_params.steering_v_max,
            args.tanh_a_gain,
        ]
    )
    lower_limit = torch.tensor(
        [
            -args.tanh_w_gain,  # self.car_params.steering_v_min,
            -args.tanh_a_gain,
        ]
    )


    for dim_idx in range(STCar.N_CONTROLS):
        u[:, dim_idx] = torch.clamp(
            u[:, dim_idx],
            min=lower_limit[dim_idx].item(),
            max=upper_limit[dim_idx].item(),
        )

    if return_K:
        return u, K
    else:
        return u

def lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    return_eigs: bool = False,
):
    """Solve the discrete time lqr controller.
    x_{t+1} = A x_t + B u_t
    cost = sum x.T*Q*x + u.T*R*u
    Code adapted from Mark Wilfred Mueller's continuous LQR code at
    http://www.mwm.im/lqr-controllers-with-python/
    Based on Bertsekas, p.151
    Yields the control law u = -K x
    """

    # first, try to solve the ricatti equation
    X = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = scipy.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    if not return_eigs:
        return K
    else:
        eigVals, _ = scipy.linalg.eig(A - B * K)
        return K, eigVals



