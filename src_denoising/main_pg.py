'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import argparse
import os
import pickle

import torch
from torch.autograd import Variable

import experiment_pg
import metrics
from preprocess_pg import Bayer, cfa_to_depth, depth_to_cfa
from progressbar import progress_bar
import utils

import dnd_dataset

base_expdir = "../results_poissongaussian_denoising/"

parser = argparse.ArgumentParser(description='N3Net for Poisson-Gaussian image denoising')

# Noise level function used for testing
parser.add_argument("--testnlf.a", type=float, default=0.01)
parser.add_argument("--testnlf.b", type=float, default=0.0001)

# Data preprocessing
utils.add_commandline_flag(parser, "--clip", "--no_clip", True) # whether to clip noisy images
utils.add_commandline_flag(parser, "--bayer", "--no_bayer", False) # whether to train on monochrome or images with color filter array
utils.add_commandline_flag(parser, "--inputnoisemap", "--no_inputnoisemap", True) # whether to concatenate noise level function to input

# DnCNN
utils.add_commandline_networkparams(parser, "dncnn", 64, 6, 3, "relu", True) # Specification of DnCNNs: features, depth, kernelsize, activation, batchnorm
parser.add_argument("--nfeatures_interm", type=int, default=8) # output channels of intermediate DnCNNs
parser.add_argument("--ndncnn", type=int, default=3) # number of DnCNN networks

# Nonlocal block
utils.add_commandline_networkparams(parser, "embedcnn", 64, 3, 3, "relu", True) # Specification of embedding CNNs: features, depth, kernelsize, activation, batchnorm
parser.add_argument("--embedcnn.nplanes_out", type=int, default=8) # output channels of embedding CNNs
parser.add_argument("--nl_k", type=int, default=7) # number of neighborhood volumes
# stride and patchsize for extracting patches in non-local block
parser.add_argument("--nl_patchsize", type=int, default=10)
parser.add_argument("--nl_stride", type=int, default=5)
utils.add_commandline_flag(parser, "--nl_temp.external_temp", "--nl_temp.no_external_temp", True) # whether to have separate temperature CNN
parser.add_argument("--nl_temp.temp_bias", type=float, default=0.1) # constant bias of temperature
utils.add_commandline_flag(parser, "--nl_temp.distance_bn", "--nl_temp.no_distance_bn", True) # whether to have batch norm layer after calculat of pairwise distances
utils.add_commandline_flag(parser, "--nl_temp.avgpool", "--nl_temp.no_avgpool", default=True) # in case of separate temperature CNN: whether to average pool temperature of each patch or to take temperature of center pixel

# Optimizer
parser.add_argument('--optimizer', default="adam", choices=["adam", "sgd"]) # which optimizer to use
# parameters for Adam
parser.add_argument("--adam.beta1", type=float, default=0.9)
parser.add_argument("--adam.beta2", type=float, default=0.999)
parser.add_argument("--adam.eps", type=float, default=1e-8)
parser.add_argument("--adam.weightdecay", type=float, default=1e-4)
parser.add_argument('--adam.lr', type=float, default=0.001)
# parameters for SGD
parser.add_argument("--sgd.momentum", type=float, default=0.9)
parser.add_argument("--sgd.weightdecay", type=float, default=1e-4)
parser.add_argument('--sgd.lr', type=float, default=0.1)

# Run mode
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resumedir', default=None)
parser.add_argument('--resumeepoch', type=int, default=-1)
parser.add_argument('--resume_for_train', action="store_true")

parser.add_argument('--eval', action='store_true')
parser.add_argument('--evaldir', default="")
parser.add_argument('--eval_epoch', type=int)
parser.add_argument("--eval_dnd", action="store_true")
parser.add_argument("--eval_image")

parser.add_argument('--suffix', default="")

# Training options
parser.add_argument("--batchsize", type=int, default=32)
parser.add_argument("--patchsize", type=int, default=80)
parser.add_argument("--trainsetiters", type=int, default=128)

# Misc
utils.add_commandline_flag(parser, "--use_gpu", "--use_cpu", True)
parser.add_argument("--base_expdir", default=base_expdir)

def add_summary(experiment, log, name, value, iter=None):
    if iter is None:
        iter = experiment.step
    if not name in log:
        log[name] = {}
    log[name][iter] = value
    try:
        experiment.writer.add_scalar(name, value, iter)
    except:
        pass

def get_stats():
    loss = utils.AverageMeter(name="Loss")
    psnr = utils.AverageMeter(name="PSNR")
    ssim = utils.AverageMeter(name="SSIM")
    stats = {"loss":loss, "psnr": psnr, "ssim": ssim}
    return stats

def test_epoch(epoch, experiment):
    testloaders = experiment.create_test_dataloaders()
    use_cuda = experiment.use_cuda
    net = experiment.net
    summaries = experiment.summaries
    criterion = experiment.criterion

    net.eval()
    utils.set_random_seeds(1234)

    with torch.no_grad():
        for testloader, testname in testloaders:
            stats = get_stats()
            print("Testing on {}".format(testname))
            for batch_idx, inputs in enumerate(testloader):
                experiment.step = epoch*len(experiment.trainloader) + int(batch_idx/len(testloader)*len(experiment.trainloader))
                experiment.iter = batch_idx
                torch.cuda.empty_cache()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs, targets = experiment.data_preprocessing(inputs)
                inputs, targets = Variable(inputs, requires_grad=False), Variable(targets, requires_grad=False)
                pred = net(inputs)
                batch_loss = criterion(pred, targets)
                loss = batch_loss.mean()
                stats["loss"].update(loss.data)
                psnr_iter = metrics.psnr(pred, targets, maxval=1).mean().data
                ssim_iter = metrics.ssim(pred, targets)

                stats["psnr"].update(psnr_iter, pred.size(0))
                stats["ssim"].update(ssim_iter.data, pred.size(0))

                progress_bar(batch_idx, len(testloader), 'Loss: %.5f | PSNR: %.2f | SSIM: %.3f'
                    % (stats["loss"].avg, stats["psnr"].avg, stats["ssim"].avg))

                del pred, inputs, targets

            add_summary(experiment, summaries, testname + "/epoch", epoch)
            for k,stat in stats.items():
                add_summary(experiment, summaries, testname + "/" + k, stat.avg)

def evaluate(experiment):
    net = experiment.net
    logdir = experiment.logdir
    args = experiment.args

    print("Evaluation mode")
    checkpoint_dir = os.path.join(args.evaldir, "checkpoint/")
    eval_file =  os.path.join(logdir, "_data.pkl")
    print("Checkpoints from " + args.evaldir)
    eval_count = 1
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    while os.path.exists(eval_file):
        eval_file =  os.path.join(logdir, "_data_{}.pkl".format(eval_count))
        eval_count+=1
    print("Writing to " + eval_file)
    experiment.summaries = {}
    epoch = args.eval_epoch
    print("Epoch {}".format(epoch))
    filename = '%03d_ckpt.t7' % (epoch)
    checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
    net.load_state_dict(checkpoint["net"])
    test_epoch(epoch, experiment)
    with open(eval_file, "wb") as f:
        pickle.dump(dict(summaries=experiment.summaries), f, protocol=pickle.HIGHEST_PROTOCOL)

def evaluate_image(experiment):
    net = experiment.net
    args = experiment.args
    image_path = args.eval_image
    print("Evaluating on image {}".format(image_path))
    print("Checkpoints from " + args.evaldir)
    checkpoint_dir = os.path.join(args.evaldir, "checkpoint/")
    epoch = args.eval_epoch
    print("Epoch {}".format(epoch))
    filename = '%03d_ckpt.t7' % (epoch)
    checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
    net.load_state_dict(checkpoint["net"])
    net.eval()
    if image_path[-4:] == ".mat":
        import h5py
        import numpy as np
        import scipy.io as sio
        h5file = h5py.File(image_path, 'r')
        noisy = torch.from_numpy(np.float32(np.array(h5file['Inoisy']).T)).contiguous()
        noisy = noisy.view(1,1, noisy.shape[0], noisy.shape[1])
        noisy = Variable(noisy, requires_grad=False).cuda()
        nlf_h5 = h5file["nlf"]
        nlf = {}
        nlf["a"] = nlf_h5["a"][0][0]
        nlf["b"] = nlf_h5["b"][0][0]
        denoised = torch.zeros_like(noisy)

        noisy = experiment.add_input_channels(noisy, nlf["a"], nlf["b"])

        for yy in range(2):
            for xx in range(2):
                print("Denoising CFA channel {},{}".format(yy,xx))
                denoised_chan = net(noisy[:,:,yy::2,xx::2])
                denoised.data[0,0,yy::2,xx::2] = denoised_chan.data[0,0,:,:]

        denoised = denoised[0,...].data.cpu().numpy()
        denoised = np.transpose(denoised, [1,2,0])
        save_file = image_path[:-4] + "_n3net.mat"
        print("Saving to {}".format(save_file))
        sio.savemat(save_file, {'Idenoised': denoised})


def evaluate_dnd(experiment):
    net = experiment.net
    logdir = experiment.logdir
    args = experiment.args

    print("Evaluating on DND")
    checkpoint_dir = os.path.join(args.evaldir, "checkpoint/")
    out_dir =  os.path.join(logdir, "dnd/")
    os.makedirs(out_dir, exist_ok=True)
    print("Checkpoints from " + args.evaldir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    experiment.summaries = {}
    epoch = args.eval_epoch
    print("Epoch {}".format(epoch))
    filename = '%03d_ckpt.t7' % (epoch)
    checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
    net.load_state_dict(checkpoint["net"])
    def preproc(Inoisy, nlf):
        if experiment.args.bayer:
            Inoisy = cfa_to_depth(Inoisy)
        Inoisy = experiment.add_input_channels(Inoisy, nlf["a"], nlf["b"])
        return Inoisy
    def postproc(Idenoised, nlf):
        if experiment.args.bayer:
            return depth_to_cfa(Idenoised)
        else:
            return Idenoised
    mode = "raw_full" if experiment.args.bayer else "raw"
    dnd_dataset.eval_dnd(experiment, mode, out_dir, True, preproc=preproc, postproc=postproc)


def train_epoch(experiment):
    use_cuda = experiment.use_cuda
    net = experiment.net
    optimizer = experiment.optimizer
    summaries = experiment.summaries
    criterion = experiment.criterion
    epoch = experiment.epoch

    lr = experiment.base_lr * experiment.learning_rate_decay(epoch)
    for group in experiment.optimizer.param_groups:
        group['lr'] = lr
    print('\nEpoch: %d, Learning rate: %f, Expdir %s' % (epoch, lr, experiment.expname))

    net.train()

    stats = get_stats()

    trainloader = experiment.trainloader
    for batch_idx, inputs in enumerate(trainloader):
        experiment.epoch_frac = float(batch_idx) / len(trainloader)
        experiment.step = epoch*len(trainloader) + batch_idx
        experiment.iter = batch_idx
        if use_cuda:
            inputs = inputs.cuda()
        optimizer.zero_grad()
        inputs, targets = experiment.data_preprocessing(inputs)
        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets, requires_grad=False)

        pred = net(inputs)
        batch_loss = criterion(pred, targets)

        loss = batch_loss.mean()
        psnr_iter = metrics.psnr(pred, targets, maxval=1).mean().data
        ssim_iter = metrics.ssim(pred, targets)

        stats["loss"].update(loss.data, pred.size(0))
        stats["psnr"].update(psnr_iter, pred.size(0))
        stats["ssim"].update(ssim_iter.data, pred.size(0))

        loss.backward()
        del(loss)
        optimizer.step()

        if batch_idx % (len(trainloader) // 10) == 0:
            progress_bar(batch_idx, len(trainloader),"")
            print("Batch {:05d}, ".format(batch_idx), end='')
            for k,stat in stats.items():
                print("{}: {:.4f}, ".format(stat.name, stat.avg), end='')
            print("")
        progress_bar(batch_idx, len(trainloader), 'Loss: %.5f | PSNR: %.2f | SSIM: %.3f'
                % (stats["loss"].ema, stats["psnr"].ema, stats["ssim"].ema))

    stop = (lr == 0)
    progress_bar(batch_idx, len(trainloader), 'Loss: %.5f | PSNR: %.2f | SSIM: %.3f'
                % (stats["loss"].avg, stats["psnr"].avg, stats["ssim"].avg))

    add_summary(experiment, summaries, "train/epoch", epoch)
    for k,stat in stats.items():
        add_summary(experiment, summaries, "train/" + k, stat.avg)
    print("")

    return stop

def trainloop(experiment):
    stop = False
    while not stop:
        stop = train_epoch(experiment)
        utils.save_checkpoint(experiment)
        with open(os.path.join(experiment.expdir + "_data.pkl"), "wb") as f:
            pickle.dump(dict(summaries=experiment.summaries), f, protocol=pickle.HIGHEST_PROTOCOL)
        experiment.epoch += 1

def run(experiment):
    args = experiment.args
    if args.eval_dnd:
        evaluate_dnd(experiment)
    elif args.eval_image is not None:
        evaluate_image(experiment)
    elif args.eval:
        evaluate(experiment)
    else:
        trainloop(experiment)

def main():
    args = parser.parse_args()
    if args.eval_dnd or args.eval_image is not None:
        args.eval = True

    if args.eval:
        allargs = dict(vars(args)).keys()
        defaults = {k: parser.get_default(k) for k in allargs}
        parser.set_defaults(**{n:None for n in allargs})
        args = parser.parse_args()
        parser.set_defaults(**defaults)
        set_args = argparse.Namespace(**{k:v for k,v in dict(vars(args)).items() if v is not None})
        my_experiment = load(args.evaldir, set_args, parseargs=False, resume_for_train=False)
    else:
        args = utils.get_args(args, args.base_expdir)
        my_experiment = experiment_pg.Experiment(args)
        my_experiment.setup()
    run(my_experiment)

def load(resumedir, newargs, parseargs=False, resume_for_train=True):
    default_args = parser.parse_args(args=[])

    if parseargs:
        newargs = parser.parse_args(namespace=newargs)

    if resume_for_train:
        newargs.resume = True
        newargs.resumedir = resumedir
        newargs.resume_for_train = resume_for_train
    else:
        newargs.resume = False
        newargs.eval = True
        newargs.evaldir = resumedir
    args = utils.get_args(newargs, base_expdir, args_default=default_args)

    my_experiment = experiment_pg.Experiment(args)
    my_experiment.setup()
    return my_experiment


if __name__ == '__main__':
    main()