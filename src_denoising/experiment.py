'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import os

import tensorboardX as tbx
import torch
import torchvision.transforms as transforms

import denoising_data
import img_dataset
import preprocess
import utils

import models.n3net as n3net

class Experiment:
    def __init__(self, args):
        self.args = utils.parsed_args_to_obj(args)
        self.base_expdir = args.base_expdir

    def create_network(self):
        args = self.args
        ninchannels=1
        noutchannels=1

        temp_opt = args.nl_temp

        n3block_opt = dict(
            k=args.nl_k,
            patchsize=args.nl_patchsize,
            stride=args.nl_stride,
            temp_opt=temp_opt,
            embedcnn_opt=args.embedcnn)

        dncnn_opt = args.dncnn
        dncnn_opt["residual"] = True

        net = n3net.N3Net(ninchannels, noutchannels, args.nfeatures_interm,
                          nblocks=args.ndncnn, block_opt=dncnn_opt, nl_opt=n3block_opt, residual=False)

        return net

    def create_test_dataloaders(self):
        transform_test = transforms.Compose([
            img_dataset.ToGrayscale(),
            transforms.ToTensor(),
        ])

        testsets = [
            (img_dataset.PlainImageFolder(root=denoising_data.set12_val_dir, transform=transform_test, cache=True), "Set12"),
            (img_dataset.PlainImageFolder(root=denoising_data.bsds500_val68_dir, transform=transform_test, cache=True), "val68"),
            (img_dataset.PlainImageFolder(root=denoising_data.urban_val_dir, transform=transform_test, cache=True), "Urban100")
         ]
        testloaders = [(torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1), name)
                      for testset,name in testsets]

        return testloaders

    def create_train_dataloaders(self, patchsize, batchsize, trainsetiters):
        transform_train = transforms.Compose([
            transforms.RandomCrop(patchsize),
            preprocess.RandomOrientation90(),
            transforms.RandomVerticalFlip(),
            img_dataset.ToGrayscale(),
            transforms.ToTensor(),
        ])
        self.batchsize=batchsize

        train_folders = [
            denoising_data.bsds500_train_dir,
            denoising_data.bsds500_test_dir
        ]

        trainset = img_dataset.PlainImageFolder(root=train_folders, transform=transform_train, cache=True)
        trainset = torch.utils.data.ConcatDataset([trainset]*trainsetiters)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                                  shuffle=True, num_workers=20)

        return trainloader

    def data_preprocessing(self, input):
        args = self.args
        sigma = args.sigma / 255.0
        noise = torch.zeros_like(input)
        noise.normal_(0, 1)
        noise *= sigma
        noisy = input + noise
        return noisy, input

    def create_loss(self):
        args = self.args
        lossfac = 1600.0 / (args.patchsize**2) # to match loss magnitude of DnCNN training

        def criterion(pred, targets):
            loss = (0.5*(pred-targets)**2).view(pred.shape[0],-1).sum(dim=1, keepdim=True) * lossfac
            return loss

        return criterion

    def create_optimizer(self):
        args = self.args
        parameters = utils.parameters_by_module(self.net)
        if args.optimizer == "sgd":
            self.base_lr = args.sgd["lr"]
            optimizer =  torch.optim.SGD(parameters, lr=self.base_lr, momentum=args.sgd["momentum"], weight_decay=args.sgd["weightdecay"])
        elif args.optimizer == "adam":
            self.base_lr = args.adam["lr"]
            optimizer = torch.optim.Adam(parameters, lr=self.base_lr, weight_decay=args.adam["weightdecay"], betas=(args.adam["beta1"], args.adam["beta2"]), eps=args.adam["eps"])

        # bias parameters do not get weight decay
        for pg in optimizer.param_groups:
            if pg["name"]=="bias":
                pg["weight_decay"] = 0

        return optimizer

    def learning_rate_decay(self, epoch):
        if epoch > 50:
            return 0
        decay = 10**(-3.0*epoch/50.0)
        return decay

    def experiment_dir(self):
        if self.args.resume:
            self.expname = os.path.split(self.args.resumedir)[-1]
            return self.args.resumedir
        elif self.args.eval:
            self.expname = os.path.split(self.args.evaldir)[-1]
            return self.args.evaldir

        expname = utils.get_result_dir(self.base_expdir, self.args.suffix)
        self.expname = expname
        return os.path.join(self.base_expdir, expname)

    def get_logdir(self):
        expdir = self.expdir
        if not self.args.eval:
            logdir = "{}/{}".format(expdir, "train")
        else:
            i = 0
            while True:
                logdir = "{}/{}{:02d}".format(expdir, "test", i)
                if not os.path.exists(os.path.join(logdir)):
                    break
                else:
                    i += 1
        return logdir


    def setup(self):
        args = self.args
        self.expdir = self.experiment_dir()
        self.logdir = self.get_logdir()
        self.writer = tbx.SummaryWriter(log_dir=self.logdir)
        os.makedirs(self.expdir, exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)

        if not args.eval:
            utils.save_script_call(os.path.join(self.expdir, "args.pkl"), args)
        else:
            utils.save_script_call(os.path.join(self.logdir, "args.pkl"), args)

        self.use_cuda = torch.cuda.is_available() and args.use_gpu

        self.trainloader = self.create_train_dataloaders(args.patchsize, args.batchsize, args.trainsetiters)
        self.net = self.create_network()
        self.optimizer = self.create_optimizer()
        self.criterion = self.create_loss()

        print(self.expname)
        print(self.base_expdir)
        print(self.net)
        nparams = utils.parameter_count(self.net)
        print("#Parameter {}".format(nparams))

        self.summaries = {}
        self.epoch = 0
        if args.resume:
            self.summaries, self.epoch = utils.load_checkpoint(self.net, self.optimizer, self.expdir, withoptimizer=args.resume_for_train, resume_epoch=args.resumeepoch)

        if self.use_cuda:
            self.net.cuda()
