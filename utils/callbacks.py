from fastai.torch_core import *
from fastai.callback import *
from fastai.callbacks.tracker import TrackerCallback
from fastai.basic_train import *
from fastai.basic_data import *

import time


class SaveModelCallback(TrackerCallback):
    """A `TrackerCallback` that saves the model when monitored quantity is best.

    Original SaveModelCallback saves the optimizer as well.

    Diff from origin:
    Track the best epoch and time to reach it."""
    def __init__(self, learn:Learner, monitor:str='valid_loss', mode:str='auto', every:str='improvement', name:str='bestmodel',
                 with_opt=True):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every,self.name,self.with_opt = every,name,with_opt
        self.best_epoch = 0
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'

    def jump_to_epoch(self, epoch:int)->None:
        try:
            self.learn.load(f'{self.name}_{epoch-1}', purge=False)
            print(f"Loaded {self.name}_{epoch-1}")
        except: print(f'Model {self.name}_{epoch-1} not found.')

    def on_train_begin(self, **kwargs:Any) ->None:
        super().on_train_begin(**kwargs)
        self._start_time = time.time()
        self._best_epoch_time = time.time()

    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.every=="epoch": self.learn.save(f'{self.name}_{epoch}', with_opt=self.with_opt)
        else: #every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                self.best = current
                self.best_epoch = epoch
                self._best_epoch_time = time.time()
                self.learn.save(f'{self.name}', with_opt=self.with_opt)

    def on_train_end(self, **kwargs):
        "Load the best model."
        self.time_to_best_epoch = self._best_epoch_time - self._start_time
        if self.every=="improvement":
            self.learn.load(f'{self.name}', purge=False, with_opt=self.with_opt)


class ReduceLROnPlateauCallback(TrackerCallback):
    "A `TrackerCallback` that reduces learning rate when a metric has stopped improving."
    def __init__(self, learn:Learner, monitor:str='valid_loss', mode:str='auto', patience:int=0, factor:float=0.2,
                 min_delta:int=0, min_lr:float=0.001):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.patience,self.factor,self.min_delta,self.min_lr = patience,factor,min_delta,min_lr
        if self.operator == np.less:  self.min_delta *= -1
        self.changed_lr_on_epochs = {}

    def on_train_begin(self, **kwargs:Any)->None:
        "Initialize inner arguments."
        self.wait, self.opt = 0, self.learn.opt
        super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        """Compare the value monitored to its best and maybe reduce lr.

        Diff from origin:
        Tracks the epochs when lr was changed."""
        current = self.get_monitor_value()
        if current is None: return
        if self.operator(current - self.min_delta, self.best): self.best,self.wait = current,0
        else:
            self.wait += 1
            if self.wait > self.patience and self.opt.lr > self.min_lr:
                self.opt.lr *= self.factor
                self.wait = 0
                print(f'Epoch {epoch}: reducing lr to {self.opt.lr}')
                self.changed_lr_on_epochs[epoch] = self.opt.lr


class MetricTracker(LearnerCallback):
    """Wrap a `func` in a callback for metrics computation.

        learn: Learner
        func: function (metric) to compute stats on. Should be an average metric
        train: whether to record stats on train or valid phase.

        https://forums.fast.ai/t/i-cant-save-via-add-metrics-in-custom-callback/43057
        https://docs.fast.ai/metrics.html#Creating-your-own-metric

    """
    _order = -20  # Needs to run before the recorder

    def __init__(self, learn, func, train, name=None):
        super().__init__(learn)
        # If func has a __name__ use this one else it should be a partial
        if name is None:
            name = func.__name__ if hasattr(func, '__name__') else func.func.__name__
        self.func, self.train, self.name = func, train, name

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names([self.name])

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0., 0

    def on_batch_end(self, last_output, last_target, train, **kwargs):
        """Update metric computation with `last_output` and `last_target`."""
        if self.train == train:
            if not is_listy(last_target): last_target=[last_target]
            self.count += first_el(last_target).size(0)
            val = self.func(last_output, *last_target)
            self.val += first_el(last_target).size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        """Set the final result in `last_metrics`."""
        return add_metrics(last_metrics, self.val/self.count)
