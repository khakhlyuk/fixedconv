"Provides convenient callbacks for Learners that write model images, metrics/losses, stats and histograms to Tensorboard"
from fastai.basic_train import Learner
from fastai.basic_data import DatasetType, DataBunch
from fastai.vision import Image
from fastai.basic_train import LearnerCallback
from fastai.core import *
from fastai.torch_core import *
from threading import Thread, Event
from time import sleep
from queue import Queue
import statistics
import torchvision.utils as vutils
from abc import ABC

from torch.utils.tensorboard import SummaryWriter


class LearnerTensorboardWriter(LearnerCallback):
    "Broadly useful callback for Learners that writes to Tensorboard.  Writes model histograms, losses/metrics, and gradient stats."
    def __init__(self, learn:Learner, base_dir:Path, name:str, stats_iters=None, hist_iters=None, save_graph=False):
        super().__init__(learn=learn)
        self.base_dir, self.name, self.stats_iters, self.hist_iters, self.save_graph = base_dir, name, stats_iters, hist_iters, save_graph
        log_dir = base_dir/name
        self.tbwriter = SummaryWriter(str(log_dir))
        self.stats_writer = ModelStatsTBWriter()
        self.hist_writer = HistogramTBWriter()
        self.data = None
        self.metrics_root = '/metrics/'
        self._update_batches_if_needed()

    def _get_new_batch(self, ds_type:DatasetType)->Collection[Tensor]:
        "Retrieves new batch of DatasetType, and detaches it."
        return self.learn.data.one_batch(ds_type=ds_type, detach=True, denorm=False, cpu=False)

    def _update_batches_if_needed(self)->None:
        "one_batch function is extremely slow with large datasets.  This is caching the result as an optimization."
        if self.learn.data.valid_dl is None: return # Running learning rate finder, so return
        update_batches = self.data is not self.learn.data
        if not update_batches: return
        self.data = self.learn.data
        self.trn_batch = self._get_new_batch(ds_type=DatasetType.Train)
        self.val_batch = self._get_new_batch(ds_type=DatasetType.Valid)

    def _write_model_stats(self, step:int)->None:
        "Writes gradient statistics to Tensorboard."
        self.stats_writer.write(model=self.learn.model, step=step, tbwriter=self.tbwriter)

    def _write_weight_histograms(self, step:int)->None:
        "Writes model weight histograms to Tensorboard."
        self.hist_writer.write(model=self.learn.model, step=step, tbwriter=self.tbwriter)

    def _write_graph(self)->None:
        "Writes model graph to Tensorboard (without asyncTBWriter)."
        self.tbwriter.add_graph(model=self.learn.model, input_to_model=next(
            iter(self.learn.data.dl(DatasetType.Single)))[0])

    def _write_scalar(self, name:str, scalar_value, step:int)->None:
        "Writes single scalar value to Tensorboard."
        tag = self.metrics_root + name
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=step)

    def _write_training_loss(self, step:int, train_loss:Tensor)->None:
        "Writes training loss to Tensorboard."
        scalar_value = to_np(train_loss)
        tag = self.metrics_root + 'train_loss'
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=step)

    #TODO:  Relying on a specific hardcoded start_idx here isn't great.  Is there a better solution?
    def _write_metrics(self, step:int, last_metrics:MetricsList, start_idx:int=2)->None:
        """Writes training metrics to Tensorboard (ommiting epoch number and time).

        How last_metrics and recorder.names are composed
        last_metrics: [valid_loss, accuracy, train_accu]
        recorder.names: ['epoch', 'train_loss', 'valid_loss', 'accuracy', 'train_accu', 'time']
        epoch, train_loss, valid_loss, time are standard
        accuracy comes from learner(metrics=[accuracy])
        train_accu is provided from custom class MetricTracker
        """
        recorder = self.learn.recorder
        for i, name in enumerate(recorder.names[start_idx:]):
            if last_metrics is None or len(last_metrics) < i+1: return
            scalar_value = last_metrics[i]
            self._write_scalar(name=name, scalar_value=scalar_value, step=step)

    def _write_embedding(self, step:int)->None:
        "Writes embedding to Tensorboard."
        for name, emb in self.learn.model.named_children():
            if isinstance(emb, nn.Embedding):
                self.tbwriter.add_embedding(list(emb.parameters())[0], global_step=step, tag=name)

    def on_train_begin(self, **kwargs: Any) -> None:
        if self.save_graph:
            self._write_graph()

    def on_batch_end(self, iteration:int, train:bool, **kwargs)->None:
        "Callback function that writes batch end appropriate data to Tensorboard."
        if iteration == 0 or not train: return
        self._update_batches_if_needed()
        if self.hist_iters is not None:
            if iteration % self.hist_iters == 0: self._write_weight_histograms(step=iteration)

    # Doing stuff here that requires gradient info, because they get zeroed out afterwards in training loop
    def on_backward_end(self, iteration:int, train:bool, **kwargs)->None:
        "Callback function that writes backward end appropriate data to Tensorboard."
        if iteration == 0 or not train: return
        self._update_batches_if_needed()
        if self.stats_iters is not None:
            if iteration % self.stats_iters == 0: self._write_model_stats(step=iteration)

    def on_epoch_end(self, last_metrics:MetricsList, smooth_loss:Tensor, epoch:int, **kwargs)->None:
        "Callback function that writes epoch end appropriate data to Tensorboard."
        self._write_training_loss(step=epoch, train_loss=smooth_loss)
        self._write_metrics(step=epoch, last_metrics=last_metrics)
        self._write_embedding(step=epoch)


class TBWriteRequest(ABC):
    "A request object for Tensorboard writes.  Useful for queuing up and executing asynchronous writes."
    def __init__(self, tbwriter: SummaryWriter, step:int):
        super().__init__()
        self.tbwriter = tbwriter
        self.step = step

    @abstractmethod
    def write(self)->None: pass


# SummaryWriter writes tend to block quite a bit.  This gets around that and greatly boosts performance.
# Not all tensorboard writes are using this- just the ones that take a long time.  Note that the
# SummaryWriter does actually use a threadsafe consumer/producer design ultimately to write to Tensorboard,
# so writes done outside of this async loop should be fine.
class AsyncTBWriter():
    "Callback for GANLearners that writes to Tensorboard.  Extends LearnerTensorboardWriter and adds output image writes."
    def __init__(self):
        super().__init__()
        self.stop_request = Event()
        self.queue = Queue()
        self.thread = Thread(target=self._queue_processor, daemon=True)
        self.thread.start()

    def request_write(self, request: TBWriteRequest)->None:
        "Queues up an asynchronous write request to Tensorboard."
        if self.stop_request.isSet(): return
        self.queue.put(request)

    def _queue_processor(self)->None:
        "Processes queued up write requests asynchronously to Tensorboard."
        while not self.stop_request.isSet():
            while not self.queue.empty():
                if self.stop_request.isSet(): return
                request = self.queue.get()
                request.write()
            sleep(0.2)

    #Provided this to stop thread explicitly or by context management (with statement) but thread should end on its own
    # upon program exit, due to being a daemon.  So using this is probably unecessary.
    def close(self)->None:
        "Stops asynchronous request queue processing thread."
        self.stop_request.set()
        self.thread.join()

    # Nothing to do, thread already started.  Could start thread here to enforce use of context manager
    # (but that sounds like a pain and a bit unweildy and unecessary for actual usage)
    def __enter__(self): pass

    def __exit__(self, exc_type, exc_value, traceback): self.close()


asyncTBWriter = AsyncTBWriter()


class ModelImageSet():
    "Convenience object that holds the original, real(target) and generated versions of a single image fed to a model."
    @staticmethod
    def get_list_from_model(learn:Learner, ds_type:DatasetType, batch:Tuple)->[]:
        "Factory method to convert a batch of model images to a list of ModelImageSet."
        image_sets = []
        x,y = batch[0],batch[1]
        preds = learn.pred_batch(ds_type=ds_type, batch=(x,y), reconstruct=True)
        for orig_px, real_px, gen in zip(x,y,preds):
            orig, real = Image(px=orig_px), Image(px=real_px)
            image_set = ModelImageSet(orig=orig, real=real, gen=gen)
            image_sets.append(image_set)
        return image_sets

    def __init__(self, orig:Image, real:Image, gen:Image): self.orig, self.real, self.gen = orig, real, gen


class HistogramTBRequest(TBWriteRequest):
    "Request object for model histogram writes to Tensorboard."
    def __init__(self, model:nn.Module, step:int, tbwriter:SummaryWriter, name:str):
        super().__init__(tbwriter=tbwriter, step=step)
        self.params = [(name, values.clone().detach().cpu()) for (name, values) in model.named_parameters()]
        self.name = name

    def _write_histogram(self, param_name:str, values)->None:
        "Writes single model histogram to Tensorboard."
        tag = self.name + '/weights/' + param_name
        self.tbwriter.add_histogram(tag=tag, values=values, global_step=self.step)

    def write(self)->None:
        "Writes model histograms to Tensorboard."
        for param_name, values in self.params: self._write_histogram(param_name=param_name, values=values)


#If this isn't done async then this is sloooooow
class HistogramTBWriter():
    "Writes model histograms to Tensorboard."
    def __init__(self): super().__init__()

    def write(self, model:nn.Module, step:int, tbwriter:SummaryWriter, name:str='model')->None:
        "Writes model histograms to Tensorboard."
        request = HistogramTBRequest(model=model, step=step, tbwriter=tbwriter, name=name)
        asyncTBWriter.request_write(request)


class ModelStatsTBRequest(TBWriteRequest):
    "Request object for model gradient statistics writes to Tensorboard."
    def __init__(self, model:nn.Module, step:int, tbwriter:SummaryWriter, name:str):
        super().__init__(tbwriter=tbwriter, step=step)
        self.gradients = [x.grad.clone().detach().cpu() for x in model.parameters() if x.grad is not None]
        self.name = name

    def _add_gradient_scalar(self, name:str, scalar_value)->None:
        "Writes a single scalar value for a gradient statistic to Tensorboard."
        tag = self.name + '/gradients/' + name
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=self.step)

    def _write_avg_norm(self, norms:[])->None:
        "Writes the average norm of the gradients to Tensorboard."
        avg_norm = sum(norms)/len(self.gradients)
        self._add_gradient_scalar('avg_norm', scalar_value=avg_norm)

    def _write_median_norm(self, norms:[])->None:
        "Writes the median norm of the gradients to Tensorboard."
        median_norm = statistics.median(norms)
        self._add_gradient_scalar('median_norm', scalar_value=median_norm)

    def _write_max_norm(self, norms:[])->None:
        "Writes the maximum norm of the gradients to Tensorboard."
        max_norm = max(norms)
        self._add_gradient_scalar('max_norm', scalar_value=max_norm)

    def _write_min_norm(self, norms:[])->None:
        "Writes the minimum norm of the gradients to Tensorboard."
        min_norm = min(norms)
        self._add_gradient_scalar('min_norm', scalar_value=min_norm)

    def _write_num_zeros(self)->None:
        "Writes the number of zeroes in the gradients to Tensorboard."
        gradient_nps = [to_np(x.data) for x in self.gradients]
        num_zeros = sum((np.asarray(x) == 0.0).sum() for x in gradient_nps)
        self._add_gradient_scalar('num_zeros', scalar_value=num_zeros)

    def _write_avg_gradient(self)->None:
        "Writes the average of the gradients to Tensorboard."
        avg_gradient = sum(x.data.mean() for x in self.gradients)/len(self.gradients)
        self._add_gradient_scalar('avg_gradient', scalar_value=avg_gradient)

    def _write_median_gradient(self)->None:
        "Writes the median of the gradients to Tensorboard."
        median_gradient = statistics.median(x.data.median() for x in self.gradients)
        self._add_gradient_scalar('median_gradient', scalar_value=median_gradient)

    def _write_max_gradient(self)->None:
        "Writes the maximum of the gradients to Tensorboard."
        max_gradient = max(x.data.max() for x in self.gradients)
        self._add_gradient_scalar('max_gradient', scalar_value=max_gradient)

    def _write_min_gradient(self)->None:
        "Writes the minimum of the gradients to Tensorboard."
        min_gradient = min(x.data.min() for x in self.gradients)
        self._add_gradient_scalar('min_gradient', scalar_value=min_gradient)

    def write(self)->None:
        "Writes model gradient statistics to Tensorboard."
        if len(self.gradients) == 0: return
        norms = [x.data.norm() for x in self.gradients]
        self._write_avg_norm(norms=norms)
        self._write_median_norm(norms=norms)
        self._write_max_norm(norms=norms)
        self._write_min_norm(norms=norms)
        self._write_num_zeros()
        self._write_avg_gradient()
        self._write_median_gradient()
        self._write_max_gradient()
        self._write_min_gradient()


class ModelStatsTBWriter():
    "Writes model gradient statistics to Tensorboard."
    def write(self, model:nn.Module, step:int, tbwriter:SummaryWriter, name:str='model_stats')->None:
        "Writes model gradient statistics to Tensorboard."
        request = ModelStatsTBRequest(model=model, step=step, tbwriter=tbwriter, name=name)
        asyncTBWriter.request_write(request)


class ImageTBRequest(TBWriteRequest):
    "Request object for model image output writes to Tensorboard."
    def __init__(self, learn:Learner, batch:Tuple, step:int, tbwriter:SummaryWriter, ds_type:DatasetType):
        super().__init__(tbwriter=tbwriter, step=step)
        self.image_sets = ModelImageSet.get_list_from_model(learn=learn, batch=batch, ds_type=ds_type)
        self.ds_type = ds_type

    def _write_images(self, name:str, images:[Tensor])->None:
        "Writes list of images as tensors to Tensorboard."
        tag = self.ds_type.name + ' ' + name
        self.tbwriter.add_image(tag=tag, img_tensor=vutils.make_grid(images, normalize=True), global_step=self.step)

    def _get_image_tensors(self)->([Tensor], [Tensor], [Tensor]):
        "Gets list of image tensors from lists of Image objects, as a tuple of original, generated and real(target) images."
        orig_images, gen_images, real_images = [], [], []
        for image_set in self.image_sets:
            orig_images.append(image_set.orig.px)
            gen_images.append(image_set.gen.px)
            real_images.append(image_set.real.px)
        return orig_images, gen_images, real_images

    def write(self)->None:
        "Writes original, generated and real(target) images to Tensorboard."
        orig_images, gen_images, real_images = self._get_image_tensors()
        self._write_images(name='orig images', images=orig_images)
        self._write_images(name='gen images',  images=gen_images)
        self._write_images(name='real images', images=real_images)


#If this isn't done async then this is noticeably slower
class ImageTBWriter():
    "Writes model image output to Tensorboard."
    def __init__(self): super().__init__()

    def write(self, learn:Learner, trn_batch:Tuple, val_batch:Tuple, step:int, tbwriter:SummaryWriter)->None:
        "Writes training and validation batch images to Tensorboard."
        self._write_for_dstype(learn=learn, batch=val_batch, step=step, tbwriter=tbwriter, ds_type=DatasetType.Valid)
        self._write_for_dstype(learn=learn, batch=trn_batch, step=step, tbwriter=tbwriter, ds_type=DatasetType.Train)

    def _write_for_dstype(self, learn:Learner, batch:Tuple, step:int, tbwriter:SummaryWriter, ds_type:DatasetType)->None:
        "Writes batch images of specified DatasetType to Tensorboard."
        request = ImageTBRequest(learn=learn, batch=batch, step=step, tbwriter=tbwriter, ds_type=ds_type)
        asyncTBWriter.request_write(request)
