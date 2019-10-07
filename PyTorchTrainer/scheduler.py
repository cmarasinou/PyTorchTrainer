from collections import defaultdict
import math

from PyTorchTrainer.callback import Callback

class ParameterUpdater:

    def __init__(self, schedule, params, opt=None):
        self.schedule = schedule
        self.params = params
        self.opt = opt
        self.start_parameters = None

    def set_optimizer(self, opt):
        self.opt = opt

    def save_start_values(self):
        start = []
        for group in self.opt.param_groups:
            params = {}
            for item in self.params:
                name = item['name']
                if name in group:
                    params[name] = group[name]
            start.append(params)
        self.start_parameters = start

    def current_values(self):
        return [
            {conf['name']: group[conf['name']]
             for conf in self.params}
            for group in self.opt.param_groups]

    def step(self):
        mult = self.schedule.update()
        for i, group in enumerate(self.opt.param_groups):
            for item in self.params:
                name = item['name']
                if name in group:
                    params = self.start_parameters[i]
                    inverse = item.get('inverse', False)
                    start_value = params.get(name)
                    group[name] = start_value * ((1 - mult) if inverse else mult)



class Scheduler(Callback):
    default = [{'name': 'lr'}]

    def __init__(self, schedule, mode='epoch', params_conf=None):
        self.schedule = schedule
        self.params_conf = params_conf or self.default
        self.mode = mode
        self.history = []

    def training_started(self, optimizer, **kwargs):
        self.updater = ParameterUpdater(self.schedule, self.params_conf, optimizer)
        self.updater.save_start_values()

    def batch_ended(self, phase, **kwargs):
        if self.mode == 'batch' and phase.grad:
            self.update_parameters()

    def epoch_ended(self, epoch, **kwargs):
        if self.mode == 'epoch':
            self.update_parameters()

    def update_parameters(self):
        self.updater.step()
        self.history.append(self.updater.current_values())

    def parameter_history(self, name, *names, group_index=0):
        if not self.history:
            return {}
        curve = defaultdict(list)
        names = [name] + list(names)
        for record in self.history:
            group = record[group_index]
            for name in names:
                if name not in group:
                    raise ValueError(f'no history for parameter \'{name}\'')
                curve[name].append(group[name])
        return dict(curve)



class CosineAnnealingSchedule:
    """
    The schedule class that returns eta multiplier in range from 0.0 to 1.0.
    """
    def __init__(self, eta_min=0.0, eta_max=1.0, t_max=100, t_mult=2):
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.t_max = t_max
        self.t_mult = t_mult
        self.iter = 0

    def update(self, **kwargs):
        self.iter += 1

        eta_min, eta_max, t_max = self.eta_min, self.eta_max, self.t_max

        t = self.iter % t_max
        eta = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / t_max))
        if t == 0:
            self.iter = 0
            self.t_max *= self.t_mult

        return eta



class OneCycleSchedule:

    def __init__(self, t, linear_pct=0.2, eta_max=1.0, eta_min=None,
                 div_factor=100, decay_to_zero=True):

        if eta_min is None:
            eta_min = eta_max / div_factor

        self.t = t
        self.linear_pct = linear_pct
        self.eta_max = eta_max
        self.eta_min = eta_min

        self.t_cosine = int(math.ceil(t * (1 - linear_pct))) + 1
        self.t_linear = int(math.floor(t * linear_pct))

        self.cosine = CosineAnnealingSchedule(
            eta_min=0 if decay_to_zero else eta_min,
            eta_max=eta_max,
            t_max=self.t_cosine, t_mult=1)
        self.linear = lambda x: x * (eta_max - eta_min) / self.t_linear + eta_min

        self.iter = 0

    def update(self, **kwargs):
        self.iter += 1
        if self.iter <= self.t_linear:
            return self.linear(self.iter)
        else:
            return self.cosine.update()


class StepSchedule:

    def __init__(self, schedule):
        '''
        Args:
            schedule: list of tuples of this form (learning_rate, n_epochs)
        '''
        self.schedule = []
        for lr, n_epochs in schedule:
            self.schedule +=([lr]*n_epochs)
        self.iter = 0

    def update(self, **kwargs):
        
        if self.iter<len(self.schedule):
            lr = self.schedule[self.iter]
        else:
            lr = self.schedule[-1] #returning last lr
        self.iter += 1
        return lr


class PytorchScheduler(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler
    def phase_ended(self, phase, **kwargs):
        if phase.grad == False:
            self.scheduler.step(phase.last_loss)