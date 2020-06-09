import torch
import torch.distributed as dist
from collections import defaultdict, deque
import datetime
import time
import huepy as hue

from .distributed import is_dist_avail_and_initialized


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=10, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def print_log(self, epoch, step, iters_per_epoch):
        print(hue.lightgreen('[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e'
                             % (epoch, step, iters_per_epoch,
                                self.meters['loss_value'].avg, self.meters['lr'].value)))
        if 'num_fg' in self.meters:       
            print('\tfg/bg: %d/%d, time cost: %.4f' %
                      (self.meters['num_fg'].avg, 
                       self.meters['num_bg'].avg, 
                       self.meters['batch_time'].avg))
        else:
            print('\ttime cost: %.4f' % (self.meters['batch_time'].avg))
        print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_box: %.4f'
                  % (self.meters['loss_objectness'].avg, 
                     self.meters['loss_rpn_box_reg'].avg,
                     self.meters['loss_box_reg'].avg))
        print('\tdet_cls: %.4f, reid_cls: %.4f'
                  % (self.meters['loss_detection'].avg, 
                     self.meters['loss_reid'].avg))