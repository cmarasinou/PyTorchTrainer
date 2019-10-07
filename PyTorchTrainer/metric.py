from collections import defaultdict
import torch

from PyTorchTrainer.callback import Callback

def accuracy(out, y_true):
    if out.dim() > 1:
        y_hat = out.argmax(dim=-1).view(y_true.size(0), -1)
        y_true = y_true.view(y_true.size(0), -1)
    else:
        y_hat = (out>0.5).to(torch.float)
    match = y_hat == y_true
    return match.float().mean()

class Accuracy(Callback):

    def epoch_started(self, **kwargs):
        self.values = defaultdict(int)
        self.counts = defaultdict(int)

    def batch_ended(self, phase, output, target, **kwargs):
        acc = accuracy(output, target).detach().item()
        self.counts[phase.name] += target.size(0)
        self.values[phase.name] += target.size(0) * acc

    def epoch_ended(self, phases, **kwargs):
        for phase in phases:
            metric = self.values[phase.name] / self.counts[phase.name]
            phase.update_metric('accuracy', metric)


class AverageMetric(Callback):
    '''
    Callback wrapper for metric functions
    Takes a function for the metric with arguments (output, target)
    Function must return average over batch
    '''

    def __init__(self, metric_fn = accuracy):
        self.metric_fn = metric_fn
        self.name = metric_fn.__name__

    def epoch_started(self, **kwargs):
        self.values = defaultdict(int)
        self.counts = defaultdict(int)

    def batch_ended(self, phase, output, target, **kwargs):
        value = self.metric_fn(output, target).detach().item()
        self.counts[phase.name] += target.size(0)
        self.values[phase.name] += target.size(0) * value

    def epoch_ended(self, phases, **kwargs):
        for phase in phases:
            metric = self.values[phase.name] / self.counts[phase.name]
            phase.update_metric(self.name, metric)


def IoU(y_pred, y_true):
    '''Returns IoUs between pairs of binary images.
    Compares images with same batch index within y_pred and y_true
    Args:
        y_pred (torch.Tensor (n,h,w)): batch of the predicted binary images
        y_true (torch.Tensor (n,h,w)): batch of the true binary images
    '''
    if y_pred.dim()==4: y_pred = y_pred.squeeze(1)
    if y_true.dim()==4: y_true = y_true.squeeze(1)
    assert(y_pred.dim()==3 and y_true.dim()==3)
    # thresholding
    y_pred = y_pred>0.2
    y_true = y_true>0.2

    y_pred, y_true = y_pred.to(torch.long), y_true.to(torch.long)

    intersection = y_pred.__and__(y_true)
    union = y_pred.__or__(y_true)
    # sum over spacial directions
    intersection = intersection.sum([1,2]).to(torch.float)
    union = union.sum([1,2]).to(torch.float)
    # Treat special empty case
    for i in range(0,union.size(0)):
        if union[i]==0:
            # In this case we want IoU = 1, so we can set
            union[i] = 1
            intersection[i] = 1 
    return (intersection/union).mean()