import argparse

from utils.data_loader import get_train_valid_loader, get_test_loader
from utils.utils import num_params, save_summary, format_scientific

from fastai.vision import *
from fastai.vision.data import *
from fastai.callbacks import EarlyStoppingCallback, CSVLogger
from utils.callbacks import ReduceLROnPlateauCallback, SaveModelCallback, MetricTracker
from utils.tensorboard import LearnerTensorboardWriter

import modules.nets as nets


model_names = sorted(name for name in nets.__dict__
                     if name.islower() and not name.startswith("__")
                     and "resnet" in name
                     and callable(nets.__dict__[name]))
datasets = ['cifar10', 'cifar100', 'mnist', 'fmnist']

parser = argparse.ArgumentParser(description='Training resnets and fixed resnets')
parser.add_argument('--model', '-a', required=True, metavar='MODEL',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--dataset', required=True,
                    choices=datasets,
                    help='datasets: ' + ' | '.join(datasets))
parser.add_argument('-f', '--fixed',
                    action='store_true',
                    help='Trainable or fixed ResNet')
parser.add_argument('--ff', '--fully_fixed',
                    action='store_true',
                    help='If convolutions at stage 0 should be replaced by '
                         'fixed too.')
parser.add_argument('-k', default=1, type=int,
                    help='widening factor k (default: 1). Used for fixed resnets only')
parser.add_argument('--data_path', default='/root/data',
                    help='path to save downloaded data to')
parser.add_argument('--logs_path', default='./logs_resnet',
                    help='path to save downloaded data to')
parser.add_argument('-c', '--cuda', default=0, type=int,
                    help='cuda kernel to use (default: 0)')
# parser.add_argument('--epochs', default=200, type=int,
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
parser.add_argument('--bs', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--min_lr', default=1e-4, type=float,
                    metavar='LR', help='minimal learning rate (default: 1e-4)')
parser.add_argument('--mom', default=0.9, type=float, metavar='M',
                    help='momentum (default=0.9)')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--manualSeed', type=int, help='manual seed')


def main():
    args = parser.parse_args()

    if args.manualSeed is None:
        seed = random.randint(1, 10000)
    else:
        seed = args.manualSeed
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_name = args.model
    k = args.k
    fixed = args.fixed
    fully_fixed = args.ff
    cuda = args.cuda

    if args.dataset in ['cifar10', 'mnist', 'fmnist']:
        num_classes = 10
    elif args.dataset in ['cifar100']:
        num_classes = 100
    else:
        raise RuntimeError

    if not args.fixed:
        model_code = model_name + '(k={})'.format(k)
    else:
        model_code = 'fixed_' + model_name + '(k={}_ff={})'.format(
            k, fully_fixed)
    model = nets.__dict__[model_name](
        num_classes=num_classes, k=k, fixed=fixed, fully_fixed=fully_fixed)

    early_stopping = False
    save_model = True
    log = True
    write = True

    data_path = Path(args.data_path)
    logs_path = Path(args.logs_path)
    model_saves_dir = Path('model_saves')  # relative to logs_path
    csv_logs_dir = Path('csv_logs')  # relative to logs_path
    tb_dir = Path('tensorboard')  # relative to logs_path

    max_lr = args.lr
    min_lr = args.min_lr

    momentum = args.mom
    weight_decay = args.wd
    nesterov = False

    bs = args.bs
    num_workers = args.workers
    pin_memory = False  # no difference for the given machine

    device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
    defaults.device = device
    model.to(device)

    print("Running on", defaults.device)

    # Data
    train_loader, valid_loader = get_train_valid_loader(
        dataset=args.dataset,
        data_dir=data_path, valid_size=0.1, augment=True, random_seed=seed,
        batch_size=bs, num_workers=num_workers, shuffle=True,
        pin_memory=pin_memory, show_sample=False)

    test_loader = get_test_loader(
        dataset=args.dataset,
        data_dir=data_path,
        batch_size=bs, num_workers=num_workers, shuffle=False,
        pin_memory=pin_memory)

    train_epoch_len = len(train_loader)

    # Callbacks
    callback_fns = [
        partial(MetricTracker, func=accuracy, train=True, name='train_accu'),  # additionally track train accuracy
    ]
    if early_stopping:
        callback_fns.append(partial(
            ReduceLROnPlateauCallback, monitor='valid_loss', mode='auto',
            patience=10, factor=0.1, min_delta=0, min_lr=min_lr))
        callback_fns.append(partial(
            EarlyStoppingCallback, monitor='valid_loss', min_delta=0,
            patience=20))
    if save_model: callback_fns.append(partial(
        SaveModelCallback, every='improvement', monitor='accuracy',
        mode='max', name=model_code))
    if log: callback_fns.append(partial(
        CSVLogger, append=False, filename=csv_logs_dir/model_code))
    if write: callback_fns.append(partial(
        LearnerTensorboardWriter, base_dir=logs_path/tb_dir, name=model_code,
        stats_iters=10*train_epoch_len, hist_iters=10*train_epoch_len))

    # setting up fastai objects
    bunch = ImageDataBunch(train_loader, valid_loader, test_dl=test_loader,
                           device=device, path=data_path)
    # lr is set by fit
    sgd = partial(torch.optim.SGD, momentum=momentum,
                  weight_decay=weight_decay, nesterov=nesterov)

    learn = Learner(bunch, model, loss_func=nn.CrossEntropyLoss(),
                    opt_func=sgd, true_wd=False, wd=weight_decay,
                    metrics=[accuracy], callback_fns=callback_fns,
                    path=logs_path, model_dir=model_saves_dir)

    print('Training', model_code)

    # Training
    if early_stopping:
        # reduce on plateau + early stopping case
        learn.fit(200, lr=max_lr, wd=weight_decay)
    else:
        models_to_warm_up = ['110', '164', '1001', '1202']
        # regular step lr scheduler, as in the paper
        if len([True for x in models_to_warm_up if x in model_name]) > 0:
            # warmup for larger models
            learn.fit(1, lr=max_lr * 0.1, wd=weight_decay)
            learn.fit(99, lr=max_lr, wd=weight_decay)
        else:
            # no warmup
            learn.fit(100, lr=max_lr, wd=weight_decay)
        learn.fit(50, lr=max_lr * 0.1, wd=weight_decay)
        learn.fit(50, lr=max_lr * 0.01, wd=weight_decay)

    # Gathering stats and saving them
    best_epoch, best_value = learn.save_model_callback.best_epoch, learn.save_model_callback.best
    time_to_best_epoch = learn.save_model_callback.time_to_best_epoch
    changed_lr_on_epochs = learn.reduce_lr_on_plateau_callback.changed_lr_on_epochs

    print("Best model was found at epoch {} with accuracy value {:.4f} in {:.2f} seconds.".format(best_epoch, best_value, time_to_best_epoch))

    loss_train, accu_train = learn.validate(dl=learn.data.train_dl)
    loss_valid, accu_valid = learn.validate(dl=learn.data.valid_dl)
    loss_test,  accu_test  = learn.validate(dl=learn.data.test_dl)
    # accu_train, accu_valid, accu_test = accu_train.item(), accu_valid.item(), accu_test.item()

    n_params, n_layers = num_params(model)

    val_dict = {'name': model_code,
                'accu_test': accu_test * 100,
                'n_params': n_params,
                'epochs': best_epoch + 1,
                'time': time_to_best_epoch,
                'changed_lr_on': ','.join(map(format_scientific, changed_lr_on_epochs.keys())),
                'loss_train': loss_train,
                'loss_valid': loss_valid,
                'loss_test':  loss_test,
                'accu_train': accu_train * 100,
                'accu_valid': accu_valid * 100,
                'accu_test (again)': accu_test * 100,
                'other': '',
               }

    save_summary(logs_path/'models_summary.csv', val_dict)


if __name__ == '__main__':
    main()
