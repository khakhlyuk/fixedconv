import argparse

from utils.data_loader import get_train_valid_loader, get_test_loader
from utils.utils import num_params, save_summary, format_scientific
from modules.fixedconv import get_fixed_conv_params

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
conv_type_names = ['G', 'B']

parser = argparse.ArgumentParser(description='Training resnets and fixed resnets')
parser.add_argument('--model', '-a', metavar='MODEL',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--ff', '--fully_fixed',
                    choices=['y', 'n'],
                    help='If convolutions at stage 0 should be replaced by fixed too.'
                         'Used for fixed resnets only'
                         'Choices: "y", "n"')
parser.add_argument('-k', default=1, type=int,
                    help='widening factor k (default: 1). Used for fixed resnets only')
parser.add_argument('--conv_type', default='G',
                    choices=conv_type_names,
                    help='Conv types. Gaussian, Bilinear interpolation etc. '
                         'Used for fixed resnets only.'
                         'Choices: ' + ' | '.join(conv_type_names))
parser.add_argument('--sigma', type=float, default=0.8,
                    help='Parameter sigma for the gaussian kernel.')
parser.add_argument('--data_path', default='/root/data/cifar10',
                    help='path to save downloaded data to')
parser.add_argument('-c', '--cuda', default=0, type=int,
                    help='cuda kernel to use (default: 0)')
parser.add_argument('--epochs', default=150, type=int,
                    help='number of total epochs to run')
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


def main():
    args = parser.parse_args()

    cuda = args.cuda
    k = args.k
    model_name = args.model

    if model_name.startswith("resnet"):  # for original resnets
        model_code = model_name
        model = nets.__dict__[model_name]()
    else:  # for fixed resnets
        fully_fixed = True if args.ff == 'y' else False
        conv_type = args.conv_type
        fixed_conv_params = get_fixed_conv_params(
            args.conv_type, bilin_interpol=True, n=3, sigma=args.sigma)

        model_code = model_name + '(k={},type={},fully_fixed={},sigma={})'.format(
            k, conv_type, fully_fixed, args.sigma)
        model = nets.__dict__[model_name](k, fully_fixed, fixed_conv_params)

    save_model = True
    log = True
    write = True

    data_path = Path(args.data_path)
    logs_path = Path('logs')  # relative to project directory
    model_saves_dir = Path('model_saves')
    csv_logs_dir = Path('csv_logs')
    tb_dir = Path('tensorboard')

    max_lr = args.lr
    min_lr = args.min_lr
    epochs = args.epochs

    momentum = args.mom
    weight_decay = args.wd
    nesterov = False

    bs = args.bs  # as used in resnet paper. Takes 1.5 MB of RAM, so not an issue
    num_workers = args.workers  # optimal for the given machine. sometimes gives an error if num_workers>0
    pin_memory = False  # no difference for the given machine

    device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
    defaults.device = device
    model.to(device)

    print("Running on", defaults.device)

    # Data
    train_loader, valid_loader = get_train_valid_loader(
        data_dir=data_path, valid_size=0.1, augment=True, random_seed=42,
        batch_size=bs, num_workers=num_workers, shuffle=True,
        pin_memory=pin_memory, show_sample=False)

    test_loader = get_test_loader(
        data_dir=data_path,
        batch_size=bs, num_workers=num_workers, shuffle=False,
        pin_memory=pin_memory)

    train_epoch_len = len(train_loader)

    # Callbacks
    callback_fns = [
        partial(ReduceLROnPlateauCallback, monitor='valid_loss', mode='auto', patience=10, factor=0.1, min_delta=0, min_lr=min_lr),
        partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0, patience=20),
        partial(MetricTracker, func=accuracy, train=True, name='train_accu'),  # additionally track train accuracy
    ]
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
    sgd = partial(torch.optim.SGD, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

    learn = Learner(bunch, model, loss_func=nn.CrossEntropyLoss(), opt_func=sgd, true_wd=False, wd=weight_decay,
                    metrics=[accuracy], callback_fns=callback_fns,
                    path=logs_path, model_dir=model_saves_dir)

    # Training
    learn.fit(epochs, lr=max_lr, wd=weight_decay)

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
