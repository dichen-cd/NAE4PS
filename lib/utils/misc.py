# -*- coding: utf-8 -*-
# Reference:
# https://github.com/pytorch/vision/blob/fe3b4c8f2c/references/detection/utils.py

import argparse
import sys
import torch
import huepy as hue

from .serialization import read_json, write_json


class Nestedspace(argparse.Namespace):

    def __setattr__(self, name, value):
        if '.' in name:
            group, name = name.split('.', 1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        if '.' in name:
            group, name = name.split('.', 1)
            try:
                ns = self.__dict__[group]
            except KeyError:
                raise AttributeError
            return getattr(ns, name)
        else:
            raise AttributeError

    def to_dict(self, args=None, prefix=None):
        out = {}
        args = self if args is None else args
        for k, v in args.__dict__.items():
            if isinstance(v, Nestedspace):
                out.update(self.to_dict(v, prefix=k))
            else:
                if prefix is not None:
                    out.update({prefix + '.' + k: v})
                else:
                    out.update({k: v})
        return out

    def from_dict(self, dic):
        for k, v in dic.items():
            self.__setattr__(k, v)

    def export_to_json(self, file_path):
        write_json(self.to_dict(), file_path)

    def load_from_json(self, file_path):
        self.from_dict(read_json(file_path))


def lazy_arg_parse(parser):
    '''
    Only parse the given flags.
    '''
    def parse_known_args():
        args = sys.argv[1:]
        namespace = Nestedspace()

        try:
            namespace, args = parser._parse_known_args(args, namespace)
            if hasattr(namespace, '_unrecognized_args'):
                args.extend(getattr(namespace, '_unrecognized_args'))
                delattr(namespace, '_unrecognized_args')
            return namespace, args
        except argparse.ArgumentError:
            err = sys.exc_info()[1]
            parser.error(str(err))

    args, argv = parse_known_args()
    if argv:
        msg = _('unrecognized arguments: %s')
        parser.error(msg % ' '.join(argv))
    return args


def ship_data_to_cuda(batch, device):
    f = lambda sample: ship_data_to_cuda_singe_sample(
        sample[0], sample[1], device=device)
    return tuple(map(list, zip(*map(f, batch))))


def ship_data_to_cuda_singe_sample(img, target, device):
    img = img.to(device)
    if target is not None:
        target['boxes'] = target['boxes'].to(device)
        target['labels'] = target['labels'].to(device)
        if 'heatmaps' in target:
            target['heatmaps'] = target['heatmaps'].to(device)
    return img, target


def resume_from_checkpoint(args, model, optimizer=None, lr_scheduler=None):
    load_name = args.resume
    checkpoint = torch.load(load_name)
    args.train.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    print(hue.good('loaded checkpoint %s' % (load_name)))
    print(hue.info('model was trained for %s epochs' % (args.train.start_epoch)))
    return args, model, optimizer, lr_scheduler


def get_optimizer(args, model):
    lr = args.train.lr
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr':lr * (args.train.double_bias + 1),
                            'weight_decay': args.train.bias_decay and args.train.weight_decay or 0}]
            else:
                params += [{'params': [value], 'lr':lr,
                            'weight_decay': args.train.weight_decay}]

    optimizer = torch.optim.SGD(params, momentum=args.train.momentum)
    return optimizer


def get_lr_scheduler(args, optimizer):
    if args.train.lr_decay_milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.train.lr_decay_milestones,
            gamma=args.train.lr_decay_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.train.lr_decay_step,
            gamma=args.train.lr_decay_gamma)

    return scheduler


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def lucky_bunny(i):
    print('')
    print('|￣￣￣￣￣￣￣￣|')
    print('|    TRAINING    |')
    print('|     epoch      |')
    print('|       ' + hue.bold(hue.green(str(i))) + '        |')
    print('| ＿＿＿_＿＿＿＿|')
    print(' (\__/) ||')
    print(' (•ㅅ•) || ')
    print(' / 　 づ')
    print('')
