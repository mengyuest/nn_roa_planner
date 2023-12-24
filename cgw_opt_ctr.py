import numpy as np
import time
import scipy.io as sio
import h5py
import cgw_sim_choi as choi
import pickle
import matplotlib.pyplot as plt

class Grid():
    def __init__(self, path):
        # 'N', 'axis', 'bdry', 'bdryData', 'dim', 'dx', 'max', 'min', 'shape', 'vs', 'xs'
        self.N = None
        self.axis = None
        self.bdry = None
        self.bdryData = None
        self.dim = None
        self.dx = None
        self.max = None
        self.min = None
        self.shape = None
        self.vs = None
        self.xs = None

        self.path = path
        self.load()

    def load(self):
        t1 = time.time()
        f = h5py.File(self.path, "r")
        d = f.get("grid")

        self.N = d['N'][:].T  # (4, 1)
        self.axis = d['axis'][:].T  # (2, )

        self.bdry = None  # "@addGhostExtrapolate"
        self.bdryData = None  # "[]"

        self.dim = d['dim'][:].T  # (1, 1)
        self.dx = d['dx'][:].T  # (4, 1)
        self.max = d['max'][:].T  # (4, 1)
        self.min = d['min'][:].T  # (4, 1)
        self.shape = d['shape'][:].T  # (1, 4)
        self.vs = [d[d['vs'][0, i]][:].T for i in range(4)]  # [(41, 1), (81, 1), (81, 1), (81, 1)]
        self.xs = [d[d['xs'][0, i]][:].T for i in range(4)]  # [(41, 81, 81, 81) * 4]
        t2 = time.time()
        print("Loaded grid object in %.4f seconds" % (t2 - t1))


class OptimalPolicy():
    def __init__(self, args):
        self.args = args
        self.main_file_id_str = "result_ral_main_"
        self.grid = None
        self.gait = None
        self.gait_params = None
        self.target_function = None
        self.tau = None
        self.ttr = None
        self.derivs = None
        self.load()

    def cat(self, key):
        return "./walker/%s%s.mat" % (self.main_file_id_str, key)

    def load(self):  # get_opt_ctrl_evaluation_setup.m
        load_t1 = time.time()

        # TODO a special object (nested HDF5 object, see above Grid class)
        self.grid = Grid(self.cat("grid"))

        cache = sio.loadmat(self.cat("gait"))
        self.gait = cache["gait"]  # (746, 4)
        self.gait_params = cache["gait_params"]  # (5, 1)

        # (41, 81, 81, 81)
        with h5py.File(self.cat("target"), "r") as f:
            self.target_function = f.get('target_function')[:].T

        dt = 0.01
        t_max = 1.50
        self.tau = np.linspace(0, t_max, int(t_max/dt)+1)

        # (41, 81, 81, 81)
        self.ttr = sio.loadmat(self.cat("sparse_ttr_%d"%(t_max*1000)))['ttr']

        # cell (4,1) -> (41, 81, 81, 81)
        base_path = "./walker/"
        with open("%s/sparse_derivs.pkl"%(base_path), "rb") as f:
            self.derivs = pickle.load(f)

        load_t2 = time.time()
        print("Loaded HJB  in %s seconds"%(load_t2 - load_t1))

    def get_indices_neighbor(self, x):
        lower = []
        upper = []
        for i in range(4):
            index = np.argmin(np.abs(self.grid.vs[i] - x[i]))
            if i==0:
                index = min(max(index, 1), 39)
            else:
                index = min(max(index, 1), 79)
            if self.grid.vs[i][index, 0] <= x[i]:
                lower.append(index)
                upper.append(index + 1)
            else:
                lower.append(index - 1)
                upper.append(index)
        lower = np.array(lower)
        upper = np.array(upper)
        nei = np.tile(lower, [16, 1]).reshape((2, 2, 2, 2, 4))
        nei[:, :, :, 1, -1] = upper[-1]
        nei[:, :, 1, :, -2] = upper[-2]
        nei[:, 1, :, :, -3] = upper[-3]
        nei[1, :, :, :, -4] = upper[-4]
        indices_neighbor = nei.reshape((16, 4))
        return indices_neighbor

    def get_weights_neighbor(self, x, indices_neighbor):
        weights_neighbor = np.zeros((16, 1))
        for i in range(16):
            weight_i = 1
            indices_i = indices_neighbor[i, :]
            # print(indices_i)
            for j in range(4):
                weight_i *= (self.grid.dx[j, 0] - np.abs(x[j] - self.grid.vs[j][indices_i[j]]))
            weights_neighbor[i, 0] = weight_i
        return weights_neighbor

    def eval_floor_ttr(self, x):
        # get lower/upper indices TODO handle out-bound cases
        indices_neighbor = self.get_indices_neighbor(x)

        # get ttrs (inf->max)
        adjacent_ttrs = np.zeros((16, 1))
        for i in range(16):
            i1, i2, i3, i4 = indices_neighbor[i, :]
            adjacent_ttrs[i, 0] = self.ttr[i1, i2, i3, i4]

        # assert not all(adjacent_ttrs == np.inf)
        if all(adjacent_ttrs == np.inf):
            adjacent_ttrs = 1.410
        else:
            adjacent_ttrs[adjacent_ttrs == np.inf] = np.max(adjacent_ttrs[adjacent_ttrs != np.inf])


        # get weights
        weights_neighbor = self.get_weights_neighbor(x, indices_neighbor)

        # compute weighted ttr

        ttr_floor = np.sum(weights_neighbor * adjacent_ttrs) / np.sum(weights_neighbor)
        return ttr_floor

    def interpolation(self, x):
        derivs_x = self.derivs
        ret = np.zeros((4, 1))
        # handle periods (q1, q2, dq1, dq2)
        x = np.array(x)
        x[0] = (x[0] + np.pi) % (np.pi * 2) - np.pi
        x[1] = (x[1] + np.pi) % (np.pi * 2) - np.pi
        # find the indices

        indices_neighbor = self.get_indices_neighbor(x)
        weights_neighbor = self.get_weights_neighbor(x, indices_neighbor)

        adjacent_derivs = np.zeros((16, 4))
        for j in range(4):
            for i in range(16):
                i1, i2, i3, i4 = indices_neighbor[i, :]
                adjacent_derivs[i, j] = derivs_x[j, 0][i1, i2, i3, i4]
            ret[j, 0] = np.sum(weights_neighbor * adjacent_derivs[:, j:j+1]) / np.sum(weights_neighbor)
        return ret

    def get_u(self, x):  # run_optimal_controller.m::get_optctrl
        # x ~ shape(1, 4)
        # return u: shape(1, )
        tau = np.linspace(0.00, 1.50, 151)
        dt_tau = 0.01
        assert x.shape == (1, 4)
        ttr_x = self.eval_floor_ttr(x[0])
        # print("ttr_x=", ttr_x)
        if ttr_x > dt_tau / 2:
            tau_interest = tau - ttr_x <= 0
            index_t = np.argmax(tau[tau_interest])
        else:
            index_t = 1

        # interpolation
        derivs_x = self.interpolation(x[0])
        # print("derivs_x=", derivs_x)

        # compute LgV_x
        g_vec = choi.get_gvec(x, use_torch=False)
        LgV_x = g_vec @ derivs_x  # (1, 4) * (4, 1)
        # print("LgV_x=", LgV_x)

        # compute control (bang-bang control)
        # uOpt = (LgV > eps) * obj.uMin + (LgV < -eps) * obj.uMax;
        eps = 0
        u = np.array([-self.args.hjb_u_bound]) if LgV_x > eps else np.array([self.args.hjb_u_bound])
        # print("u=", u)
        return u

    def get_indices_neighbor_batch(self, x):
        N, _ = x.shape
        lower_list=[]
        upper_list=[]
        for i in range(4):
            index = np.argmin(np.abs(self.grid.vs[i].T - x[:, i:i+1]), axis=-1)
            if i==0:
                index = np.clip(index, 1, 39)
            else:
                index = np.clip(index, 1, 79)
            mask_idx = np.where(self.grid.vs[i][index, 0] > x[:, i])
            lower = index
            upper = index + 1
            lower[mask_idx] -= 1
            upper[mask_idx] -= 1
            lower_list.append(lower)
            upper_list.append(upper)
        lower = np.stack(lower_list, axis=-1)
        upper = np.stack(upper_list, axis=-1)  # (N, 4)
        # lower .shape ()
        nei = np.tile(lower[:, None, :], [1, 16, 1]).reshape((N, 2, 2, 2, 2, 4))
        nei[:, :, :, :, 1, -1] = np.tile(upper[:, -1, None], [1, 8]).reshape((N, 2, 2, 2))
        nei[:, :, :, 1, :, -2] = np.tile(upper[:, -2, None], [1, 8]).reshape((N, 2, 2, 2))
        nei[:, :, 1, :, :, -3] = np.tile(upper[:, -3, None], [1, 8]).reshape((N, 2, 2, 2))
        nei[:, 1, :, :, :, -4] = np.tile(upper[:, -4, None], [1, 8]).reshape((N, 2, 2, 2))
        indices_neighbor = nei.reshape((N, 16, 4))
        return indices_neighbor

    def get_weights_neighbor_batch(self, x, indices_neighbor):
        N, _ = x.shape
        weight_i = np.ones((N, 16))
        for j in range(4):
            dx = np.abs(x[:, None, j] - self.grid.vs[j][indices_neighbor[:, :, j], 0])  # (N, 16)
            weight_i *= self.grid.dx[j, 0] - dx
        return weight_i[:, :, None]

    def interpolation_batch(self, x):
        N, _ = x.shape
        derivs_x = self.derivs
        # ret = np.zeros((x.shape[0], 4))
        # handle periods (q1, q2, dq1, dq2)
        x = np.array(x)
        x[:, 0] = (x[:, 0] + np.pi) % (np.pi * 2) - np.pi
        x[:, 1] = (x[:, 1] + np.pi) % (np.pi * 2) - np.pi
        indices_neighbor = self.get_indices_neighbor_batch(x)  # (N, 16, 4)
        weights_neighbor = self.get_weights_neighbor_batch(x, indices_neighbor)  # (N, 16, 1)

        i1 = indices_neighbor[:, :, 0]  # (N, 16)
        i2 = indices_neighbor[:, :, 1]  # (N, 16)
        i3 = indices_neighbor[:, :, 2]  # (N, 16)
        i4 = indices_neighbor[:, :, 3]  # (N, 16)
        ret = np.zeros((N, 4))
        for j in range(4):
            adj_der = derivs_x[j, 0][i1, i2, i3, i4]  # (N, 16)
            ret[:, j] = np.sum(weights_neighbor[:, :, 0] * adj_der, axis=1) / np.sum(weights_neighbor[:, :, 0], axis=1)
        return ret  # (N, 4)

    def get_u_batch(self, x):  # run_optimal_controller.m::get_optctrl
        # x ~ shape(N, 4)
        # return u: shape(N, 1)
        N, _ = x.shape

        # interpolation
        derivs_x = self.interpolation_batch(x)  # (N, 4)
        # print("derivs_x=", derivs_x)

        # compute LgV_x
        g_vec = choi.get_gvec(x, use_torch=False)  # (N, 4)
        LgV_x = np.sum(derivs_x * g_vec, axis=-1, keepdims=True)  # (N, 4) * (N, 4)

        # print("LgV_x", LgV_x)

        # compute control (bang-bang control)
        eps = 0
        u = np.ones((N, 1)) * self.args.hjb_u_bound
        u[LgV_x > eps] = - self.args.hjb_u_bound
        return u


def main():
    args = None
    policy = OptimalPolicy(args)
    # TODO(temp) plot target function
    plt.imshow(policy.target_function[:, :, 40, 40], origin="lower")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.4f seconds" % (t2 - t1))