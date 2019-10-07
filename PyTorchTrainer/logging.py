def in_ipynb():
    '''
    checks if we are in a jupyter notebook
    '''
    try:
        get_ipython()
        return True
    except NameError:
        return False


from collections import OrderedDict
import sys
if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
import numpy as np
import os
from tensorboardX import SummaryWriter

from PyTorchTrainer.callback import Callback
from PyTorchTrainer.train import save

def merge_dicts(ds):
    merged = OrderedDict()
    for d in ds:
        for k, v in d.items():
            merged[k] = v
    return merged


class StreamLogger(Callback):
    """
    Writes performance metrics collected during the training process into list
    of streams.

    Parameters:
        streams: A list of file-like objects with `write()` method.

    """
    def __init__(self, streams=None, log_every=1):
        self.streams = streams or [sys.stdout]
        self.log_every = log_every

    def epoch_ended(self, phases, epoch, **kwargs):
        metrics = merge_dicts([phase.last_metrics for phase in phases])
        values = [f'{k}={v:.4f}' for k, v in metrics.items()]
        values_string = ', '.join(values)
        string = f'Epoch: {epoch:4d} | {values_string}\n'
        for stream in self.streams:
            stream.write(string)
            stream.flush()


class ProgressBar(Callback):

    def training_started(self, phases, **kwargs):
        bars = OrderedDict()
        for phase in phases:
            bars[phase.name] = tqdm(total=len(phase.loader), desc=phase.name)
        self.bars = bars

    def batch_ended(self, phase, **kwargs):
        bar = self.bars[phase.name]
        bar.set_postfix_str(f'loss: {phase.last_loss:.4f}')
        bar.update(1)
        bar.refresh()

    def epoch_ended(self, **kwargs):
        for bar in self.bars.values():
            bar.n = 0
            bar.refresh()

    def training_ended(self, **kwargs):
        for bar in self.bars.values():
            bar.n = bar.total
            bar.refresh()
            bar.close()



class SaveModel(Callback):
    """

    """
    def __init__(self, model, opt, file_path = "./best_model.tar",monitor="valid_loss", mode="min"):
        self.model = model
        self.opt = opt
        self.monitor = monitor
        self.mode = mode
        self.fp = file_path
        if self.mode == "min": self.criterion = np.min
        if self.mode == "max": self.criterion = np.max

    def epoch_ended(self, phases, epoch, **kwargs):
        
        #TODO: Make sure that epoch's last metrics were updated
        
        last_metrics = merge_dicts([phase.last_metrics for phase in phases])
        metrics_history = merge_dicts([phase.metrics_history for phase in phases])
        last_metric = last_metrics[self.monitor]
        metric_history = metrics_history[self.monitor]
        
        if self.criterion(metric_history) == last_metric:
            save(self.model, self.opt, self.fp)
            out = sys.stdout
            out.write(f'Best model saving for epoch {epoch:4d}\n')
            out.flush()


class TensorBoardLogger(Callback):
    """
    Writes performance metrics collected during the training process into list
    of streams.

    Parameters:
        streams: A list of file-like objects with `write()` method.

    """
    def __init__(self, logging_dir='.', experiment = None):
        self.dir = os.path.abspath(logging_dir)
        self.experiment = experiment
        self.writer = SummaryWriter(log_dir=os.path.join(self.dir,experiment))
    
    def batch_ended(self, phase, **kwargs):
        p = phase
        self.writer.add_scalar(f'{p.name}_loss/batch',p.last_loss,p.batch_index)

    def epoch_ended(self, phases, epoch, **kwargs):
        metrics = merge_dicts([phase.last_metrics for phase in phases])
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, epoch)