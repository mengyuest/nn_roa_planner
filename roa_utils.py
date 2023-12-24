import os
import sys
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
from os.path import join as ospj
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import matplotlib

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def display():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def print_mat(mat):
    n_rows, n_cols = mat.shape
    s="\n"
    for i in range(n_rows):
        s += " ".join(["%.4f"%x for x in mat[i]]) + "\n"
    return s

def mat_print(mat, format="%7.3f"):
    s_list=[]
    for row in mat:
        s_list.append(" ".join([format % x for x in row]))
    print("\n".join(s_list))


def center_title(word, symbol, length):
    word = " %s " % word
    if length < len(word):
        length = 2 * len(word)
    left = (length - len(word)) //2
    right = length - left - len(word)
    return (symbol * left)[:left] + word + (symbol * right)[:right]

def rescale_min_max(xmin, xmax, ratio):
    center = (xmax + xmin) / 2
    radius = (xmax - xmin ) / 2 * ratio
    new_min = center - radius
    new_max = center + radius
    return new_min, new_max


def quadratic(x, P):
    return torch.bmm(torch.unsqueeze(x @ P, dim=1), x.unsqueeze(dim=-1)).squeeze(dim=-1)


def quadratic_multi(x, P):  # (b, n) * (b, n, n) -> (b, 1)
    return torch.bmm(torch.unsqueeze(x, dim=1), torch.bmm(P, torch.unsqueeze(x, dim=-1))).squeeze(-1)


def set_random_seed(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def uniform_sample_tsr(num_samples, x_mins, x_maxs, sample_ratio=1.0):
    return torch.from_numpy(uniform_sample(num_samples, x_mins, x_maxs, sample_ratio)).float()


def rand_choices_tsr(num_samples, choices):
    return torch.from_numpy(rand_choices(num_samples, choices)).float()


def uniform_sample(num_samples, x_mins, x_maxs, sample_ratio=1.0):
    x_mins = np.array(x_mins)
    x_maxs = np.array(x_maxs)
    if sample_ratio != 1.0:
        x_mids = (x_mins + x_maxs)/2
        x_deltas = (x_maxs - x_mins)/2
        x_mins = x_mids - x_deltas * sample_ratio
        x_maxs = x_mids + x_deltas * sample_ratio
        print("x_mins", x_mins)
        print("x_maxs", x_maxs)
    x = np.random.rand(num_samples, x_mins.size)
    return x * (x_maxs-x_mins) + x_mins


def rand_choices(num_samples, choices):
    return np.random.choice(choices, size=(num_samples, 1))


def round_sample(num_samples, radius):
    rth = np.random.rand(num_samples, 2)
    r = rth[:,0] * (radius-radius/2) + radius/2
    th = rth[:,1] * np.pi * 2
    return np.stack((r*np.cos(th), r*np.sin(th)), axis=-1)


def create_fcs(input_dim, output_dim, hiddens):
    linear_list = nn.ModuleList()
    linear_list.append(nn.Linear(input_dim, hiddens[0]))
    for i, hidden in enumerate(hiddens):
        if i == len(hiddens) - 1:
            linear_list.append(nn.Linear(hiddens[i], output_dim))
        else:
            linear_list.append(nn.Linear(hiddens[i], hiddens[i+1]))

    return linear_list


def get_parameters(net_list):
    params=[]
    for net in net_list:
        params += list(net.parameters())
    return params


def get_poss_dirs():
    old_poss_dirs = ["./"]+["../"*(x+1) for x in range(10)]
    old_poss_dirs = [x+"exps_cyclf" for x in old_poss_dirs]
    n_poss = len(old_poss_dirs)
    poss_dirs = []
    for i in range(n_poss):
        poss_dirs.append(old_poss_dirs[i] + "/bsl_rl")
        poss_dirs.append(old_poss_dirs[i])
    return poss_dirs

def smart_path(path):
    if path[0] != "/":
        poss_dirs = get_poss_dirs()
        for try_dir in poss_dirs:
            try_path = ospj(try_dir, path)
            if os.path.exists(try_path):
                return try_path
        raise None
    else:
        return path


def find_recent_model(path, key, format=".ckpt"):
    ckpt_files = [(os.path.getmtime(ospj(path, x)), ospj(path, x))
                  for x in os.listdir(path) if format in x and key in x]
    ckpt_files = sorted(ckpt_files, key=lambda x: x[0], reverse=True)
    return ckpt_files[0][1]


def safe_load_nn(nn, pretrained_path, load_last=False, key=None, model_iter=None, mode=None, pret=False):
    if pretrained_path is not None:
        smart_pretrained_path = smart_path(pretrained_path)
        if load_last:
            if ".ckpt" not in smart_pretrained_path:
                if model_iter is not None:
                    if mode=="car":
                        if pret:
                            smart_pretrained_path = ospj(smart_pretrained_path, "models",
                                                         "%smodel_%06d.ckpt" % (key, model_iter))
                        else:
                            smart_pretrained_path = ospj(smart_pretrained_path, "models", "%smodel_e%03d_000.ckpt"%(key, model_iter))
                    elif mode=="pogo":
                        smart_pretrained_path = ospj(smart_pretrained_path, "models",
                                                     "%se%05d.ckpt" % (key, model_iter))
                else:
                    smart_pretrained_path = find_recent_model(ospj(smart_pretrained_path, "models"), key=key)
        print("Load from %s ..."%(smart_pretrained_path))
        nn.load_state_dict(torch.load(smart_pretrained_path))
    return nn


def gen_rlp_path_list(mode, methods):
    rlp_paths=[]
    cnt = {"rl-ppo":1007, "rl-sac":1007, "rl-ddpg":1007}
    for me_i, method in enumerate(methods):
        if "rl-" in method:
            rlp_paths.append("%s_%s_%d"%(mode, method.split("rl-")[1], cnt[method]))
            cnt[method] += 1
        else:
            rlp_paths.append(None)
    return rlp_paths


def safe_load_rl_nn(rlp_path, method, auto_rl=True):
    if auto_rl:
        rlp_path = find_rl_path(rlp_path)
    rlp_path = smart_path(rlp_path)
    if ".p" not in rlp_path:
        rlp_path = find_recent_model(ospj(rlp_path, "models"), key="model_", format=".p")
    print("Load from %s ..."%(rlp_path))
    with open(rlp_path, "rb") as f:
        rlp_models = pickle.load(f)
        if method.split("rl-")[1] in ["ddpg", "ppo", "trpo", "vpg"]:
            rlp_policy, _, rlp_running = rlp_models
        elif method.split("rl-")[1] == "a2c":
            rlp_policy, rlp_running = rlp_models
        elif method.split("rl-")[1] == "sac":
            rlp_policy, _, _, _, rlp_running = rlp_models
        elif method.split("rl-")[1] == "td3":
            rlp_policy, _, _, rlp_running = rlp_models
        rlp_running.get_tensor()
    return [rlp_policy, rlp_running]

def get_loaded_rl_list(paths, methods, auto_rl=True):
    rlp_list = []
    for me_i, method in enumerate(methods):
        rlp_list.append([])
        if "rl-" not in method:
            rlp_list[-1] = [None, None]
            continue
        rlp_list[-1] = safe_load_rl_nn(paths[me_i], method, auto_rl)
    return rlp_list


def load_rl_model(rlp_path, rl_method):
    with open(rlp_path, "rb") as f:
        rlp_models = pickle.load(f)
        if rl_method.split("rl-")[1] in ["ddpg", "ppo", "trpo", "vpg"]:
            rlp_policy, _, rlp_running = rlp_models
        elif rl_method.split("rl-")[1] == "a2c":
            rlp_policy, rlp_running = rlp_models
        elif rl_method.split("rl-")[1] == "sac":
            rlp_policy, _, _, _, rlp_running = rlp_models
        elif rl_method.split("rl-")[1] == "td3":
            rlp_policy, _, _, rlp_running = rlp_models
        rlp_running.get_tensor()
    return rlp_policy, rlp_running


def canvas_adjust(xs, ys, margin):
    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    delta = max(y_max - y_min, x_max - x_min) / 2
    radius = delta + margin
    canvas_x_min = x_center - radius
    canvas_x_max = x_center + radius
    canvas_y_min = y_center - radius
    canvas_y_max = y_center + radius
    return canvas_x_min, canvas_x_max, canvas_y_min, canvas_y_max


def new_list(n):
    return [[] for _ in range(n)]

def mask_mean(val, mask):
    # if len(val.shape) == len(mask.shape) + 1:
    #     return torch.sum(val * mask) / torch.clamp(torch.sum(mask) * val.shape[1], min=1.0)
    # else:
    assert val.shape == mask.shape
    return torch.sum(val * mask) / torch.clamp(torch.sum(mask), min=1.0)

def mask_sum(val, mask):
    # if len(val.shape) == len(mask.shape) + 1:
    #     return torch.sum(val * mask) / torch.clamp(torch.sum(mask) * val.shape[1], min=1.0)
    # else:
    assert val.shape == mask.shape
    return torch.sum(val * mask)

def split_tensor(x, split, reshape=None, from_idx=0, dim=0, rand_idx=None):
    if reshape is not None:
        x_reshaped = x.reshape(reshape)
        x1, x2 = torch.split(x_reshaped, [split, x_reshaped.shape[dim] - split], dim=dim)
        x1_shape = torch.tensor(x.shape)
        x2_shape = torch.tensor(x.shape)
        x1_shape[from_idx] = split * reshape[from_idx]
        x2_shape[from_idx] = (x_reshaped.shape[dim] - split) * reshape[from_idx]
        return x1.reshape(torch.Size(x1_shape)), x2.reshape(torch.Size(x2_shape))
    else:
        if rand_idx is not None:
            return x[rand_idx[:split]], x[rand_idx[split:]]
        else:
            return x[:split], x[split:]




def update(xs, x):
    xs.update(x.detach().cpu().item())

import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def setup_data_exp_and_logger(args, just_local=False, offset=0):
    logger = Logger()
    sys.stdout = logger
    if offset == 0:
        EXP_ROOT_DIR = get_exp_dir(just_local)
    elif offset == 1:
        EXP_ROOT_DIR = "../"+get_exp_dir(just_local)
    else:
        raise NotImplementedError
    exp_dir_full = os.path.join(EXP_ROOT_DIR, "g%s_%s" % (logger._timestr, args.exp_name))
    args.exp_dir_full = exp_dir_full
    args.viz_dir = os.path.join(exp_dir_full, "viz")
    args.bak_dir = os.path.join(exp_dir_full, "src")
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs(args.bak_dir, exist_ok=True)

    args.model_dir = os.path.join(exp_dir_full, "models")
    os.makedirs(args.model_dir, exist_ok=True)

    logger.create_log(exp_dir_full)
    write_cmd_to_file(exp_dir_full, sys.argv)

    for fname in os.listdir('./'):
        if fname.endswith('.py'):
            shutil.copy(fname, os.path.join(args.bak_dir, fname))
    np.savez(os.path.join(exp_dir_full, 'args'), args=args)
    return args


def set_seed_and_exp(args, set_gpus=True, just_local=False, offset=0):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    setup_data_exp_and_logger(args, just_local, offset=offset)
    if set_gpus and hasattr(args, "gpus") and args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        os.environ["HIP_VISIBLE_DEVICES"] = args.gpus


def get_exp_dir(just_local=False):
    os.makedirs(exp_path, exist_ok=True)
    return exp_path


class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")

    def create_log(self, log_path):
        self.log = open(log_path + "/log-%s.txt" % self._timestr, "a", 1)

    def write(self, message):
        self._terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class Recorder:
    def __init__(self, larger_is_better=True):
        self.history = []
        self.larger_is_better = larger_is_better
        self.best_at = None
        self.best_val = None

    def is_better_than(self, x, y):
        if self.larger_is_better:
            return x > y
        else:
            return x < y

    def update(self, val):
        self.history.append(val)
        if len(self.history) == 1 or self.is_better_than(val, self.best_val):
            self.best_val = val
            self.best_at = len(self.history) - 1

    def is_current_best(self):
        return self.best_at == len(self.history) - 1


def get_n_meters(n):
    return [AverageMeter() for _ in range(n)]

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def n_meters(n):
    return [AverageMeter() for _ in range(n)]


def get_timestr(exp_dirname):
    timestr = exp_dirname.split("g")[1].split("_")[0]
    return timestr

def to_tensor(arr):
    return torch.from_numpy(arr).float()

def to_np(tensor):
    return tensor.detach().cpu().numpy()


def to_item(tensor):
    return tensor.detach().cpu().item()


def plt_mask_scatter(data_x, data_y, mask, color_list, scale_list, label_list=None):
    for i in range(2):
        idx = torch.where(mask==i)[0]
        if label_list is not None:
            plt.scatter(to_np(data_x[idx]), to_np(data_y[idx]), color=color_list[i], s=scale_list[i], label=label_list[i])
        else:
            plt.scatter(to_np(data_x[idx]), to_np(data_y[idx]), color=color_list[i], s=scale_list[i])


def write_cmd_to_file(log_dir, argv):
    with open(ospj(log_dir, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(argv))


def load_np_from_str(s):
    s = s.replace("]", "")
    s = s.replace("[", "")
    s = s.replace(",", " ")
    arr = []
    for l in s.split("\n"):
        arr_line = [float(x) for x in l.strip().split()]
        if len(arr_line)<2:
            continue
        arr.append(arr_line)
    return np.array(arr)


def cuda(obj, args):
    if args.gpus is not None:
        obj_cuda = obj.cuda()
    return obj_cuda


def cudas(obj_list, args):
    obj_cudas = []
    if args.gpus is not None:
        for obj in obj_list:
            obj_cuda = obj.cuda()
            obj_cudas.append(obj_cuda)
    return obj_cudas


def rl_u(obs, running, policy, umin=None, umax=None, mbpo=False):
    if mbpo:
        obs = obs.cuda().float()
        u = policy.select_action(obs.cpu(), batched=True, evaluate=True, numpy=False)
    else:
        rl_state = running(obs, update=False, use_torch=True).float()
        if hasattr(policy, "action"):
            u = policy.action(rl_state)
        elif hasattr(policy, "common"):
            rl_embed = policy.common(rl_state)
            u = policy.policy(rl_embed)
        elif hasattr(policy, "actor"):
            rl_embed = policy.actor.common(rl_state)
            u = policy.actor.policy(rl_embed)
        else:
            raise NotImplementedError
    if umin is not None and umax is not None:
        u = torch.clamp(u, min=umin, max=umax)
    return u



def find_rl_path(key):
    rl_paths = '''./exps_cyclf/bsl_rl/g0509-164019_car_sac_1007
./exps_cyclf/bsl_rl/g0509-170616_car_ppo_1007
./exps_cyclf/bsl_rl/g0509-170844_car_ddpg_1007
./exps_cyclf/bsl_rl/g0509-171058_beta_sac_1007
./exps_cyclf/bsl_rl/g0509-171100_beta_ppo_1007
./exps_cyclf/bsl_rl/g0509-171102_beta_ddpg_1007
./exps_cyclf/bsl_rl/g0509-171958_bp_ddpg_1007
./exps_cyclf/bsl_rl/g0509-175922_bp_sac_1007
./exps_cyclf/bsl_rl/g0509-175928_bp_ppo_1007
./exps_cyclf/bsl_rl/g0510-180244_car_ddpg_1009
./exps_cyclf/bsl_rl/g0510-180244_car_ppo_1009
./exps_cyclf/bsl_rl/g0510-180244_car_sac_1009
./exps_cyclf/bsl_rl/g0510-181619_beta_sac_1009
./exps_cyclf/bsl_rl/g0510-182254_beta_ppo_1009
./exps_cyclf/bsl_rl/g0510-182525_bp_sac_1009
./exps_cyclf/bsl_rl/g0510-182529_bp_ppo_1009
./exps_cyclf/bsl_rl/g0510-182533_bp_ddpg_1009
./exps_cyclf/bsl_rl/g0510-182613_beta_ddpg_1009
./exps_cyclf/bsl_rl/g0510-185345_car_sac_1008
./exps_cyclf/bsl_rl/g0510-185805_car_ppo_1008
./exps_cyclf/bsl_rl/g0510-185816_car_ddpg_1008
./exps_cyclf/bsl_rl/g0510-190623_beta_sac_1008
./exps_cyclf/bsl_rl/g0510-194350_beta_ppo_1008
./exps_cyclf/bsl_rl/g0510-194353_beta_ddpg_1008
./exps_cyclf/bsl_rl/g0510-194407_bp_sac_1008
./exps_cyclf/bsl_rl/g0510-194415_bp_ppo_1008
./exps_cyclf/bsl_rl/g0510-194547_bp_ddpg_1008'''
    rl_paths = rl_paths.split("\n")
    rl_paths = [x.split("/")[-1] for x in rl_paths]
    matched = 0
    saved_list = []
    for rl_path in rl_paths:
        if key in rl_path:
            matched += 1
            saved_list.append(rl_path)
    if matched == 1:
        return saved_list[0]
    elif matched > 1:
        print("Multiple matching for %s, %s"%(key, saved_list))
    else:
        print("Cannot find %s" % (key))
    raise NotImplementedError


def get_cs_d():
    cs_d = {
        "rl": "#7D7858",
        "rl-sac": "#4D4828",
        "rl-ppo": "#7D7858",
        "rl-ddpg": "#ADA888",
        "sac": "#5D5838",  # "#4D4828",
        "ppo": "#7D7858",
        "ddpg": "#9D9878",  # "#ADA888"
        "lqr": "#83777A",
        "mpc": "#8AAA9A",
        "hjb": "#417A68",
        "qp": "#92AEBB",
        "clf": "#C6C4C2",
        "mbpo": "#FC3468",
        "pets": "#964B00",
        "ours": "#E7452E",
        "ours-d": "#E7452E",
    }
    return cs_d


def trans_d(keyword):
    new_key = keyword
    if keyword == "runtime_step":
        new_key = "computation time"

    if keyword == "masked_dev":
        new_key = "lane deviation (m)"

    if keyword == "masked_rmse":
        new_key = "mean square error"

    if keyword == "valid":
        new_key = "valid percentage"

    if keyword == "v_err":
        new_key = "velocity error (m/s)"

    if keyword == "x_ratio":
        new_key = "jump distance"

    if keyword == "hit":
        new_key = "collision rate"

    if keyword == "dist_goal":
        new_key = "distance to goal (m)"

    if keyword == "dist_goal_rel":
        new_key = "relative dist to goal"

    if keyword == "dist_goal_t":
        new_key = "time to goal"

    if keyword == "succ":
        new_key = "success rate"

    if keyword == "fail":
        new_key = "failure rate"

    if keyword == "invalid":
        new_key = "invalid rate"

    if keyword == "goal_len":
        new_key = "distance to goal (m)"

    if keyword == "goal_ratio":
        new_key = "relative dist to goal"

    return new_key


def metric_bar_plot(cache, metric, args, index=False, masked=False, header="", cs_d=None, mode=None, rl_merged=False, new_dir=None):
    if args.use_d:
        cache = dict(cache)
        n_keys = len(cache)
        assert args.methods[-2] == "ours" and args.methods[-1] == "ours-d"
        cache[n_keys-2] = cache[n_keys-1]
        del cache[n_keys-1]
        del args.methods[-1]

    import matplotlib as mpl
    SMALL_SIZE = 16 # 20
    plt_d = {"font.size": SMALL_SIZE,
             "axes.titlesize": SMALL_SIZE,
             "axes.labelsize": SMALL_SIZE,
             "xtick.labelsize": SMALL_SIZE,
             "ytick.labelsize": SMALL_SIZE,
             "legend.fontsize": SMALL_SIZE,
             "figure.titlesize": SMALL_SIZE}

    xlabel_unit = ""
    ylabel_unit = ""
    y_gain = 1
    if metric == "runtime_step":
        ylabel_unit = " (ms)"
        y_gain = 1000

    if metric == "valid":
        ylabel_unit = " (%)"
        y_gain = 100
    if mode == "pogo" and metric in ["hit"]:
        ylabel_unit = " (%)"
        y_gain = 100


    if mode == "walker" and metric in ["rmse", "valid", "succ", "fail", "invalid"]:
        std_gain = 0.05
    elif mode == "car" and metric in ["masked_rmse", "dist_goal", "dist_goal_rel", "dist_goal_t", "masked_dev"]:
        std_gain = 0.1
    elif mode == "pogo" and metric in ["hit", "goal_len", "goal_ratio", "runtime_step"]:
        if metric in ["goal_len", "goal_ratio", "runtime_step"]:
            std_gain = 0.2
        else:
            std_gain = 0.1
    else:
        std_gain = 1.0

    y_max = None
    # if mode == "walker" and metric in ["runtime_step"]:
    #     y_max = 50

    y_min = None
    if mode == "walker":
        if metric == "rmse":
            if args.multi_target:
                y_min = 0.75
            else:
                y_min = 1
        if metric == "succ":
            y_min = 0.1
        if metric == "fail":
            if args.multi_target:
                y_min = 0.1
            else:
                y_min = 0.2
        if metric == "invalid":
            if args.multi_target:
                y_min = 0.1
            else:
                y_min = 0.25

    if mode == "car":
        if metric == "masked_dev":
            y_min = 0.3

    unique_xlabels = unique(args.methods)
    all_xlabels = [x for x in args.methods]

    # assert index
    with mpl.rc_context(plt_d):
        plt.figure(figsize=(8, 4))
        ax = plt.gca()

        values_mean = []
        values_std = []
        print(args.methods, cache.keys())
        for me_i, method in enumerate(args.methods):
            mean_val = mask_avg(cache[me_i][metric], masked=masked).cpu().numpy()
            std_val = mask_std(cache[me_i][metric], masked=masked).cpu().numpy()
            # mean_val = mask_avg(cache[method][metric], masked=masked).cpu().numpy()
            # std_val = mask_std(cache[method][metric], masked=masked).cpu().numpy()
            values_mean.append(mean_val)
            values_std.append(std_val)

        values_mean_d = {}
        values_std_d = {}
        for me_i, method in enumerate(args.methods):
            if method not in values_mean_d:
                values_mean_d[method] = []
                values_std_d[method] = []
            values_mean_d[method].append(values_mean[me_i])
            values_std_d[method].append(values_std[me_i])
        for method in unique_xlabels:
            values_mean_d[method] = np.mean(values_mean_d[method])
            values_std_d[method] = np.mean(values_std_d[method])

        if mode=="walker":
            if "rl-ddpg" in values_std_d:
                del values_mean_d["rl-ddpg"]
                del values_std_d["rl-ddpg"]
                unique_xlabels.remove("rl-ddpg")

        # TODO update the rl's names
        key_list = [key for key in values_mean_d if "rl-" in key]
        for key in key_list:
            if "rl-" in key:
                key_less = key.split("rl-")[1]
                values_mean_d[key_less] = values_mean_d[key]
                values_std_d[key_less] = values_std_d[key]
                unique_xlabels[unique_xlabels.index(key)] = key_less

        if rl_merged:
            xlabels = unique_xlabels
            x_indices = np.arange(len(unique_xlabels))  # the label locations
            new_values_mean = [values_mean_d[xxx] for xxx in unique_xlabels]
            new_values_std = [values_std_d[xxx] for xxx in unique_xlabels]
        else:
            xlabels = all_xlabels
            x_indices = np.arange(len(all_xlabels))  # the label locations
            new_values_mean = values_mean
            new_values_std = values_std

        new_values_mean = np.array(new_values_mean) * y_gain
        new_values_std = np.array(new_values_std) * std_gain * y_gain

        # if mode == "walker" and metric == "rmse":
        #     if np.min(new_values_mean) < 1:
        #         y_min = 0.0

        width = 0.75
        ax.bar(x_indices, new_values_mean, width,
               yerr=new_values_std, label=metric, capsize=5,
               color=[cs_d[method] for method in xlabels])

        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        if y_max is not None:
            ax.set_ylim(top=y_max)

        if mode in ["walker", "pogo"] and metric == "runtime_step" or (mode=="pogo" and metric=="hit"):
            ax.set_yscale('log', base=10)

        ax.set_xticks(x_indices)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("methods" + xlabel_unit)
        ax.set_ylabel(trans_d(metric) + ylabel_unit)
        if new_dir is not None:
            dir_to_viz = new_dir
        else:
            dir_to_viz = args.exp_dir_full
        plt.savefig("%s/bar%s_%s.png" % (dir_to_viz, header if rl_merged else "z" + header,
                                       metric), bbox_inches='tight', pad_inches=0.1)
        plt.close()
    if args.use_d:
        args.methods.append("ours-d")


# TODO make it like list(set(x_list))
def unique(x_list):
    new_list=[]
    for x in x_list:
        if x in new_list:
            continue
        new_list.append(x)
    return new_list


def convert_to_tensor(xs):
    if isinstance(xs[0], float) or isinstance(xs[0], int):
        tensor = torch.tensor(xs).float()
    else:
        tensor = torch.stack(xs, dim=0)
    return tensor


def convert_to_np(xs):
    if isinstance(xs[0], float) or isinstance(xs[0], int):
        arr = np.array(xs).astype(np.float32)
    elif torch.is_tensor(xs):
        arr = xs.detach().cpu().numpy()
    elif torch.is_tensor(xs[0]):
        arr = torch.stack(xs, dim=0).cpu().numpy()
    else:
        arr = xs
    return arr


def mask_avg(array, masked=False):
    tensor = convert_to_tensor(array)
    assert len(tensor.shape) == 1
    if masked:
        mask = (tensor>=0).float()
        if torch.sum(mask)==0:
            return torch.sum(mask) - 1
        else:
            return torch.sum(tensor * mask) / torch.sum(mask)
    else:
        return torch.mean(tensor)

def mask_std(array, masked=False):
    tensor = convert_to_tensor(array)
    assert len(tensor.shape)==1
    if masked:
        mask = (tensor>=0).float()
        if torch.sum(mask)==0:
            return torch.sum(mask) - 1
        else:
            return torch.std(tensor[torch.where(mask>0)])
    else:
        return torch.std(tensor)


def fine_log_file(exp_name):
    exp_dir = smart_path(exp_name)
    log_name = "log-%s.txt"%(exp_dir.split("/")[-1][1:1+len("0511-223957")].replace("_","-"))
    return ospj(exp_dir, log_name)

'''
# 1009 (7000+)
ddpg g0510-180244_car_ddpg_1009 100 1000 2000
ddpg g0511-223205_car_ddpg_1010 100 1000 2000

ours g0517-200718_EST_grow_000 g0128-111623_JOI_ROA_U12_SA_grow/models/actor_model_e000_000.ckpt
ours g0517-201256_EST_grow_001 g0128-111623_JOI_ROA_U12_SA_grow/models/actor_model_e001_000.ckpt
ours g0517-201825_EST_grow_002 g0128-111623_JOI_ROA_U12_SA_grow/models/actor_model_e002_000.ckpt

'''
def parse_from_file(from_file_path, mode=None):
    lines = open(from_file_path).readlines()
    methods = []
    rlp_paths = []
    clf_paths = []
    actor_paths = []
    roa_paths = []
    ds_list = []

    after_ds_list = False

    for l in lines:
        if l.startswith("#"):
            continue
        if len(l.strip())<5:
            continue
        if after_ds_list == False:
            after_ds_list = True
            ds_list = [int(xxx) for xxx in l.strip().split()]
            continue
        key = l.strip().split()[0]
        if "ours" in key:
            methods.append(key)
            rlp_paths.append(None)
            if mode=="walker":
                roa_paths.append(None)
            else:
                roa_paths.append(l.strip().split()[1])
            if len(l.strip().split())<=2:
                if mode == "walker":
                    clf_paths.append(l.strip().split()[1])
                    actor_paths.append(l.strip().split()[1].replace("clf_", "actor_"))
                else:
                    log_file = fine_log_file(roa_paths[-1])
                    lines=open(log_file).readlines()
                    actor_loaded = False
                    clf_loaded = False
                    for l in lines:
                        if "Load from" in l:
                            if "models/actor_" in l:
                                actor_path = "/".join(l.strip().split("/")[-3:]).split(" ")[0]
                                actor_loaded=True
                            elif "models/clf_" in l:
                                clf_path = "/".join(l.strip().split("/")[-3:]).split(" ")[0]
                                clf_loaded=True
                        if actor_loaded and clf_loaded:
                            break
                    actor_paths.append(actor_path)
                    clf_paths.append(clf_path)
            else:
                actor_paths.append(l.strip().split()[2])
                clf_paths.append(l.strip().split()[2].replace("actor_", "clf_"))

        else:
            rl_path = l.strip().split()[1]
            rest_idx = l.strip().split()[2:]
            # print("RL", rest_idx)
            for idx in rest_idx:
                methods.append(key)
                clf_paths.append(None)
                actor_paths.append(None)
                roa_paths.append(None)
                rlp_paths.append(ospj(rl_path, "models", "model_%06d.p"%(int(idx))))
    print(methods, rlp_paths, clf_paths)
    print("methods=",methods)
    data_from_file = {}
    data_from_file["methods"] = methods
    data_from_file["rlp_paths"] = rlp_paths
    data_from_file["clf_paths"] = clf_paths
    data_from_file["actor_paths"] = actor_paths
    data_from_file["roa_paths"] = roa_paths
    data_from_file["ds_list"] = ds_list
    return data_from_file




def gen_plot_rl_std_curve(ds_list, methods, cache, img_path, **kwargs):
    n_pts = np.sum(["ours" in xxx for xxx in methods])
    # each method, each seed, n_pts curves
    dd = {}
    for me_i, method in enumerate(methods):
        if method not in dd:
            dd[method] = {"reward": []}
        dd[method]["reward"].append(np.mean(convert_to_np(cache[me_i]["reward"])))

    c_d_tmp = get_cs_d()
    means = {}
    stds = {}
    for method in dd:
        dd[method]["reward"] = np.array(dd[method]["reward"]).reshape((-1, n_pts))
        means[method] = np.mean(dd[method]["reward"], axis=0)
        stds[method] = np.std(dd[method]["reward"], axis=0) if "ours" not in method else 0
    plot_std_curve(ds_list, means, stds, c_d_tmp, img_path, **kwargs)


def plot_std_curve(xs, means, stds, c_d, img_path, **kwargs):
    plt.figure(figsize=(8, 4))
    linewidth = 4.0
    fontsize = 20
    label_fontsize = 16

    c_d["rl-sac"] = "#4f72b0"  # "blue"
    c_d["rl-ppo"] = "#80b985"  # "green"
    c_d["rl-ddpg"] = "#e59a71"  # "orange"

    if np.min([means[kk] for kk in means])<-50:
        y_min=-60
    else:
        y_min=None
    for m in means:
        new_m = m.replace("rl-","")
        plt.fill_between(xs, means[m] - stds[m], means[m] + stds[m], edgecolor=None, facecolor=c_d[m], alpha=0.3)
        plt.plot(xs, means[m], color=c_d[m], label=new_m, linewidth=linewidth)
    plt.xlabel("Training samples", fontsize=fontsize)
    plt.ylabel("Rewards", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    ax = plt.gca()
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    ax.xaxis.offsetText.set_fontsize(label_fontsize)
    plt.yticks(fontsize=fontsize)
    if "loc" in kwargs:
        plt.legend(fontsize=label_fontsize, loc=kwargs["loc"], borderpad=0.2, labelspacing=0.1)
    else:
        plt.legend(fontsize=label_fontsize, loc="lower right", borderpad=0.2, labelspacing=0.1)
                   #bbox_to_anchor=(1.0, -0.05), borderpad=0.2, labelspacing=0.1)
    plt.savefig("%s" % img_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

