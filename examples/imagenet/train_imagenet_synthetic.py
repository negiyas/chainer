#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

"""
import argparse
import random
import sys
import time

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

import alex
import googlenet
import googlenetbn
import nin
import resnet50
import resnext50
import vgg16

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


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype(np.float32)
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


archs = {
    'alex': alex.Alex,
    'alex_fp16': alex.AlexFp16,
    'googlenet': googlenet.GoogLeNet,
    'googlenetbn': googlenetbn.GoogLeNetBN,
    'googlenetbn_fp16': googlenetbn.GoogLeNetBNFp16,
    'nin': nin.NIN,
    'resnet50': resnet50.ResNet50,
    'resnext50': resnext50.ResNeXt50,
    'vgg16': vgg16.VGG16,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--synthetic', '-s', action='store_true', default=False,
                        help='Use synthetic images')
    parser.set_defaults(synthetic=False)
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--iteration', '-I', type=int, default=0,
                        help='Number of iteration to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
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
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--memprof', '-M', type=int, default=0,
                        help='Enable memory profile')
    parser.add_argument('--memcallstack', '-C', type=int, default=0,
                        help='Enable memory call stack profile')
    parser.add_argument('--memprint', '-N', type=int, default=0,
                        help='Enable memory print')
    parser.add_argument('--timer', '-T', type=int, default=0,
                        help='Enable timer hook')
    parser.add_argument('--printvar', '-P', type=int, default=0,
                        help='Enable print variable hook')
    parser.add_argument('--ooc', '-O', type=int, default=0,
                        help='Enable out-of-core mode')
    parser.add_argument('--usepool', '-U', type=int, default=1,
                        help='Use memory pool')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=False)
    args = parser.parse_args()
    return args

def main(args):
    if args.debug:
        import pdb
        pdb.set_trace()
    stime = time.time()
    # Initialize the model to train
    model = archs[args.arch]()
    if args.initmodel:
        print('Load model from {}'.format(args.initmodel))
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(
            args.gpu).use()  # Make the GPU current
        model.to_gpu()

    # Load the datasets and mean file
    if args.synthetic:
        train = SyntheticDataset(insize=model.insize, length=1024)
        val = SyntheticDataset(insize=model.insize, length=1024)
    else:
        mean = np.load(args.mean)
        train = PreprocessedDataset(args.train, args.root, mean, model.insize)
        val = PreprocessedDataset(args.val, args.root, mean, model.insize, False)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    if args.iteration > 0:
        trainer = training.Trainer(updater, (args.iteration, 'iteration'), args.out)
    else:
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (1 if args.test else 100000), 'iteration'
    log_interval = (1 if args.test else 1000), 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu),
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
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if args.memprof:
        from chainer.function_hooks import CupyMemoryProfileHook
        hook = CupyMemoryProfileHook()
        with hook:
            trainer.run()
        hook.print_report()
    elif args.timer:
        from chainer.function_hooks import TimerHook
        hook = TimerHook()
        with hook:
            trainer.run()
        hook.print_report()
    elif args.printvar:
        from chainer.function_hooks import PrintHook
        hook = PrintHook()
        with hook:
            trainer.run()
    elif args.ooc:
        from chainer.function_hooks import OutOfCore
        logfile = sys.stdout if args.ooc > 1 else None
        hook = OutOfCore(logfile=logfile)
        with hook:
            trainer.run()
    else:
        trainer.run()
    etime = time.time()
    print('TIME=', etime - stime)

if __name__ == '__main__':
    args = parse_args()
    if args.usepool == 0:
        import cupy
        cupy.cuda.memory.set_allocator(cupy.cuda.memory._malloc)
    if args.memcallstack == 1:
        from cupy.cuda import memory_hooks
        with memory_hooks.CallStackHook(flush=False, full=True if args.memcallstack > 1 else False):
            main(args)
    elif args.memprint == 1:
        from cupy.cuda import memory_hooks
        with memory_hooks.DebugPrintHook():
            print("AFOAFO")
            main(args)
    else:
        main(args)

