from collections import OrderedDict

import torch
from torch.nn import functional as F

from trainer.callback import Callback

default_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, opt, phases, callbacks=None, epochs=1, device=default_device, loss_fn=F.nll_loss):
    """
    A generic structure of training loop.
    """
    model.to(device)
    
    cb = callbacks
    
    cb.training_started(phases=phases, optimizer=opt)
    
    for epoch in range(1, epochs + 1):
        cb.epoch_started(epoch=epoch)

        for phase in phases:
            n = len(phase.loader)
            cb.phase_started(phase=phase, total_batches=n)
            is_training = phase.grad
            model.train(is_training)

            for batch in phase.loader:

                phase.batch_index += 1
                cb.batch_started(phase=phase, total_batches=n)
                x, y = place_and_unwrap(batch, device)

                with torch.set_grad_enabled(is_training):
                    cb.before_forward_pass()
                    out = model(x)
                    cb.after_forward_pass()
                    loss = loss_fn(out, y)

                if is_training:
                    opt.zero_grad()
                    cb.before_backward_pass()
                    loss.backward()
                    cb.after_backward_pass()
                    opt.step()

                phase.batch_loss = loss.item()
                cb.batch_ended(phase=phase, output=out, target=y)

            cb.phase_ended(phase=phase)

        cb.epoch_ended(phases=phases, epoch=epoch)

    cb.training_ended(phases=phases)

def train_accumulate_gradients(model, opt, phases, callbacks=None, epochs=1, device=default_device, loss_fn=F.nll_loss,
    accumulation_steps = 1):
    """
    A generic structure of training loop which utilizes gradient accumulation
    """
    model.to(device)
    
    cb = callbacks
    
    cb.training_started(phases=phases, optimizer=opt)
    
    for epoch in range(1, epochs + 1):
        cb.epoch_started(epoch=epoch)

        for phase in phases:
            n = len(phase.loader)
            cb.phase_started(phase=phase, total_batches=n)
            is_training = phase.grad
            model.train(is_training)
            if is_training:
                opt.zero_grad()
            for batch in phase.loader:

                phase.batch_index += 1
                cb.batch_started(phase=phase, total_batches=n)
                x, y = place_and_unwrap(batch, device)

                with torch.set_grad_enabled(is_training):
                    cb.before_forward_pass()
                    out = model(x)
                    cb.after_forward_pass()
                    loss = loss_fn(out, y)
                if is_training:
                    cb.before_backward_pass()
                    loss = loss/accumulation_steps
                    loss.backward()
                    loss = loss*accumulation_steps
                    cb.after_backward_pass()
                    if (phase.batch_index+1)%accumulation_steps == 0:
                        opt.step()
                        opt.zero_grad()

                phase.batch_loss = loss.item()
                cb.batch_ended(phase=phase, output=out, target=y)

            cb.phase_ended(phase=phase)

        cb.epoch_ended(phases=phases, epoch=epoch)

    cb.training_ended(phases=phases)
    
def place_and_unwrap(batch, dev):
    x, *y = batch
    x = x.to(dev)
    y = [tensor.to(dev) for tensor in y]
    if len(y) == 1:
        [y] = y
    return x, y

class RollingLoss(Callback):

    def __init__(self, smooth=0.98):
        self.smooth = smooth

    def batch_ended(self, phase, **kwargs):
        prev = phase.rolling_loss
        a = self.smooth
        avg_loss = a * prev + (1 - a) * phase.batch_loss
        debias_loss = avg_loss / (1 - a ** phase.batch_index)
        phase.rolling_loss = avg_loss
        phase.update(debias_loss)

    def epoch_ended(self, phases, **kwargs):
        for phase in phases:
            phase.update_metric('loss', phase.last_loss)
            


class Phase:
    """
    Model training loop phase.

    Each model's training loop iteration could be separated into (at least) two
    phases: training and validation. The instances of this class track
    metrics and counters, related to the specific phase, and keep the reference
    to subset of data, used during phase.
    """

    def __init__(self, name, loader, grad=True):
        self.name = name
        self.loader = loader
        self.grad = grad
        self.batch_loss = None
        self.batch_index = 0
        self.rolling_loss = 0
        self.losses = []
        self.metrics = OrderedDict()

    @property
    def last_loss(self):
        return self.losses[-1] if self.losses else None

    @property
    def last_metrics(self):
        metrics = OrderedDict()
        metrics[f'{self.name}_loss'] = self.last_loss
        for name, values in self.metrics.items():
            metrics[f'{self.name}_{name}'] = values[-1]
        return metrics

    @property
    def metrics_history(self):
        metrics = OrderedDict()
        for name, values in self.metrics.items():
            metrics[f'{self.name}_{name}'] = values
        return metrics

    def update(self, loss):
        self.losses.append(loss)

    def update_metric(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)



def save(model, opt, file_path = './model.tar'):

    state = {
            'model_state': model.state_dict(),
            'optimizer_state': opt.state_dict()
        }
    torch.save(state, file_path)

def load(model, opt, file_path):
    """Loads model and optimizer checkpoint
    """
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state'])
    opt.load_state_dict(checkpoint['optimizer_state'])