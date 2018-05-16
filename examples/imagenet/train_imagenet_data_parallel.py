#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset
with multiple GPUs using data parallelism.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

You need to install chainer with NCCL to run this example.
Please see https://github.com/nvidia/nccl#build--run .

"""
from __future__ import print_function
import argparse

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import updaters

import alex
import googlenet
import googlenetbn
import nin
import resnet50
import vgg16
import train_imagenet


class SyntheticDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dim=3, insize=224, length=1024):
        self.dim = dim
        self.insize = insize
        self.length = length

    def __len__(self):
        return self.length

    def get_example(self, i):
        # It generates the i-th image/label pair and return the pair
        image = np.ndarray((self.dim, self.insize, self.insize), dtype=np.float32)
        image.fill(0.5)
        label = np.array(1, dtype=np.int32)
        # label.fill(1)
        return image, label


def main():
    archs = {
        'alex': alex.Alex,
        'alex_fp16': alex.AlexFp16,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'googlenetbn_fp16': googlenetbn.GoogLeNetBNFp16,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50,
        'resnext50': resnet50.ResNeXt50,
        'vgg16': vgg16.VGG16,
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(),
                        default='nin', help='Convnet architecture')
    parser.add_argument('--insize', '-is', default=224, type=int,
                        help='The size of input images')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=1,
                        help='Number of epochs to train')
    parser.add_argument('--iteration', '-i', type=int, default=0,
                        help='Number of iterations to train')
    parser.add_argument('--gpus', '-g', type=int, nargs="*",
                        default=[0, 1, 2, 3])
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--synthetic', '-s', action='store_true', default=False,
                    help='User cynthetic images')
    parser.set_defaults(synthetic=False)
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--ooc',
                        action='store_true', default=False,
                        help='Functions of out-of-core')
    parser.add_argument('--lwr',
                        action='store_true', default=False,
                        help='Functions of layer-wised reduction')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--debug', '-d', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    if args.debug:
        import pdb
        pdb.set_trace()

    # Check cudnn version
    import cupy
    if chainer.cuda.cudnn_enabled:
        cudnn_v = cupy.cuda.cudnn.getVersion()
        print('cuDNN Version:', cudnn_v)

    # Initialize the model to train
    if args.arch == "googlenet" or args.arch == "resnet50":
        model = archs[args.arch](insize=args.insize)
    else:
        model = archs[args.arch]()
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    # Load the datasets and mean file
    mean = np.load(args.mean)
    if args.synthetic:
        train = SyntheticDataset(insize=model.insize)
        val = SyntheticDataset(insize=model.insize)
    else:
        train = train_imagenet.PreprocessedDataset(
            args.train, args.root, mean, model.insize)
        val = train_imagenet.PreprocessedDataset(
            args.val, args.root, mean, model.insize, False)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    devices = tuple(args.gpus)

    train_iters = [
        chainer.iterators.MultiprocessIterator(i,
                                               args.batchsize,
                                               n_processes=args.loaderjob)
        for i in chainer.datasets.split_dataset_n_random(train, len(devices))]
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    if args.lwr != 0:
        updater = updaters.LWRParallelUpdater(train_iters, optimizer,
                                              devices=devices)
    else:
        updater = updaters.MultiprocessParallelUpdater(train_iters, optimizer,
                                                       devices=devices)
    if args.iteration > 0:
        trainer = training.Trainer(updater, (args.iteration, 'iteration'),
                                   args.out)
    else:
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    if args.test:
        val_interval = 5, 'epoch'
        log_interval = 1, 'epoch'
    else:
        val_interval = 1000, 'iteration'
        log_interval = 100, 'iteration'
        # val_interval = 1, 'epoch'
        # log_interval = 1000, 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpus[0]),
                   trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=100))
    # trainer.extend(extensions.ProgressBar(update_interval=2))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if args.ooc:
        # with chainer.out_of_core_mode(
        #        fine_granularity=True, devices=devices):
        with chainer.out_of_core_mode(
                fine_granularity=True, async=False, devices=devices):
            trainer.run()
    else:
        trainer.run()


if __name__ == '__main__':
    main()
