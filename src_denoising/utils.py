'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import argparse
import functools
import os
import pickle
import sys

import torch

### Directories ###

def get_result_dir(basedir, suffix):
    imax = 0
    dirs = os.listdir(basedir)
    for d in dirs:
        try:
            i = int(d[:4])
            if i >= imax:
                imax = i+1
        except: pass

    return "{:04d}-{}/".format(imax, suffix)

def check_expdir(expdir, basedir=None):
    if os.path.exists(expdir):
        return expdir

    expdir_concat = os.path.join(basedir, expdir)
    if os.path.exists(expdir_concat):
        return expdir_concat

    dirs = os.listdir(basedir)
    dirs = filter(lambda s: s.startswith(expdir), dirs)

    return os.path.join(basedir, dirs.__iter__().__next__())

def walklevel(some_dir, level=0):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level < num_sep_this and level >= 0:
            del dirs[:]

### Argument handling ###

def parsed_args_to_obj(args):
    dargs = dict(vars(args))
    ks = list(dargs)
    for k in ks:
        parts = k.split(".")
        if len(parts) > 1:
            o = dargs
            for p in parts[:-1]:
                try:
                    o[p]
                except:
                    o[p] = {}
                o = o[p]
            o[parts[-1]] = dargs[k]
    return argparse.Namespace(**dargs)


def add_commandline_flag(parser, name_true, name_false, default):
    parser.add_argument(name_true, action="store_true", dest=name_true[2:])
    parser.add_argument(name_false, action="store_false", dest=name_true[2:])
    parser.set_defaults(**{name_true[2:]: default})

def add_commandline_networkparams(parser, name, features, depth, kernel, activation, bn):
    parser.add_argument("--{}.{}".format(name, "features"), type=int, default=features)
    parser.add_argument("--{}.{}".format(name, "depth"), type=int, default=depth)
    parser.add_argument("--{}.{}".format(name, "kernel"), type=int, default=kernel)

    bnarg = "--{}.{}".format(name, "bn")
    nobnarg = "--{}.{}".format(name, "no-bn")
    add_commandline_flag(parser, bnarg, nobnarg, bn)

def get_args(args, basedir, saveload=True, args_default=None):
    args_d = dict(vars(args))
    if args.resume:
        if not args.resumedir:
            raise(Exception("resumedir must be specified"))
        else:
            resumedir = check_expdir(args.resumedir, basedir)
            args_other = load_other_expargs(resumedir)
            if args_default is not None:
                args_other = update_namespace(args_default, args_other)
            args_other_d = dict(vars(args_other))
            # args_resume = parser.parse_args(namespace=args_other)
            args_resume_d = dict(args_other_d)
            args_resume_d.update(args_d)
            all_arg_diff = functools.reduce(lambda s1,s2: s1 | s2, dict_diff(args_other_d, args_resume_d), set())-{"resume", "resumedir", "resume_for_train", "resumeepoch"}
            if len(all_arg_diff) > 0 and saveload:
                print("Arguments differ:")
                for arg in all_arg_diff:
                    print("{}: {} \t\t {}".format(arg, args_other_d[arg] if arg in args_other_d else "-", args_resume_d[arg]))
                # exit()
            args_resume = argparse.Namespace(**args_resume_d)
            args_resume.resumedir = resumedir
            return args_resume
    elif args.eval:
        if not args.evaldir:
            raise(Exception("evaldir must be specified"))
        else:
            evaldir = check_expdir(args.evaldir, basedir)
            args_other = load_other_expargs(evaldir)
            if args_default is not None:
                args_other = update_namespace(args_default, args_other)
            args_other_d = dict(vars(args_other))
            args_eval_d = dict(args_other_d)
            args_eval_d.update(args_d)
            args_eval = argparse.Namespace(**args_eval_d)
            args_eval.evaldir = evaldir
            return args_eval
    return args

def update_namespace(n1, n2):
    n1_d = dict(vars(n1))
    n1_d.update(dict(vars(n2)))
    return argparse.Namespace(**n1_d)

def load_other_expargs(expdir):
    with open(os.path.join(expdir, "args.pkl"), "rb") as f:
        d = pickle.load(f)
        return d["parsed_args"]

def save_script_call(filename, parsed_args):
    with open(filename, "wb") as f:
        pickle.dump({"argv": sys.argv, "parsed_args": parsed_args},f)


### Utils for managing network parameters ###

def get_module_name_dict(root, rootname="/"):
    def _rec(module, d, name):
        for key, child in module.__dict__["_modules"].items():
            d[child] = name + key + "/"
            _rec(child, d, d[child])

    d = {root: rootname}
    _rec(root, d, d[root])
    return d

def parameters_by_module(net, name=""):
    modulenames = get_module_name_dict(net, name + "/")
    params = [{"params": p, "name": n, "module": modulenames[m]} for m in net.modules() for n,p in m._parameters.items() if p is not None]
    return params

def parameter_count(net):
    parameters = parameters_by_module(net)

    nparams = 0
    for pg in parameters:
        for p in pg["params"]:
            nparams+=p.data.numel()

    return nparams

### Misc ###

def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)

# adapted from https://stackoverflow.com/questions/6632244/difference-in-a-dict
def dict_diff(left, right):
    left_only = set(left) - set(right)
    right_only = set(right) - set(left)
    different = {k for k in set(left) & set(right) if not left[k]==right[k]}
    return different, left_only, right_only


def load_checkpoint(net, optimizer, expdir, withoptimizer=True, resume_epoch=-1):
    print("Searching for checkpoints")
    checkpoint_dir = os.path.join(expdir, "checkpoint/")

    for i in range(1310):
        filename = '%03d_ckpt.t7' % (i)
        if not os.path.exists(os.path.join(checkpoint_dir, filename)):
            break
    epoch = i if resume_epoch<0 else resume_epoch +1
    summaries = {}
    if epoch > 0:
        filename = '%03d_ckpt.t7' % (epoch-1)
        print("Loading {}".format(os.path.join(checkpoint_dir, filename)))
        checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
        net.load_state_dict(checkpoint["net"])
        if withoptimizer:
            optimizer.load_state_dict(checkpoint["optim"])
        try:
            summaries = checkpoint["summaries"]
        except:
            pass

    return summaries, epoch

def save_checkpoint(experiment):
    net = experiment.net
    optimizer = experiment.optimizer
    summaries = experiment.summaries
    expdir = experiment.expdir
    epoch = experiment.epoch
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optim' : optimizer.state_dict(),
        'epoch': epoch,
        'summaries': summaries
    }
    checkpoint_dir = os.path.join(expdir, "checkpoint/")
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = '%03d_ckpt.t7' % (epoch)
    torch.save(state, os.path.join(checkpoint_dir, filename))

# Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, ema_decay=0.99, name=None):
        self.name = name
        self.reset()
        self.ema_decay = ema_decay

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.ema = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.ema == 0:
            self.ema = val
        else:
            self.ema = self.ema_decay * self.ema + (1-self.ema_decay) * val

        return self