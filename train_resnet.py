import argparse

from utils.data_loader import get_train_valid_loader, get_test_loader
from utils.utils import num_params, save_summary, format_scientific, format_number_km
from modules.fixedconv import get_fixed_conv_params

from fastai.vision import *
from fastai.vision.data import *
from fastai.callbacks import EarlyStoppingCallback, CSVLogger
from utils.callbacks import ReduceLROnPlateauCallback, SaveModelCallback, MetricTracker
from utils.tensorboard import LearnerTensorboardWriter
from fastai.callbacks.general_sched import GeneralScheduler, TrainingPhase

import modules.nets as nets


model_names = sorted(name for name in nets.__dict__
                     if name.islower() and not name.startswith("__")
                     and "resnet" in name
                     and callable(nets.__dict__[name]))
datasets = ['cifar10', 'cifar100', 'mnist', 'fmnist']
conv_type_names = ['R', 'G', 'B']

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
parser.add_argument('-k', default=1, type=float,
                    help='widening factor k (default: 1). Used for fixed resnets only')
parser.add_argument('--conv_type', default='R',
                    choices=conv_type_names,
                    help='Fixed kernel types. Random weights, Gaussian filter or Bilinear interpolation etc. '
                         'Used for fixed resnets only.'
                         'Choices: ' + ' | '.join(conv_type_names))
parser.add_argument('--sigma', type=float, default=0.8,
                    help='Parameter sigma for the gaussian kernel.')
parser.add_argument('--data_path', default='/root/data',
                    help='path to save downloaded data to')
parser.add_argument('--logs_path', default='./logs_resnet',
                    help='path to save logs to')
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

    if model_name.startswith("resnet"):  # for original resnets
        model_code = model_name
        model = nets.__dict__[model_name](num_classes=num_classes)
    else:  # for fixed resnets
        fully_fixed = True if args.ff == 'y' else False
        conv_type = args.conv_type
        fixed_conv_params = get_fixed_conv_params(
            args.conv_type, bilin_interpol=True, kernel_size=3, sigma=args.sigma)

        model_code = model_name + '(k={},type={},fully_fixed={},type={})'.format(
            k, conv_type, fully_fixed, conv_type)
        model = nets.__dict__[model_name](k, fully_fixed, fixed_conv_params,
                                          num_classes)

    reduce_on = False
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
    if save_model: callback_fns.append(partial(
        SaveModelCallback, every='improvement', monitor='accuracy',
        mode='max', name=model_code))
    if log: callback_fns.append(partial(
        CSVLogger, append=False, filename=csv_logs_dir/model_code))
    if write: callback_fns.append(partial(
        LearnerTensorboardWriter, base_dir=logs_path/tb_dir, name=model_code,
        stats_iters=10*train_epoch_len, hist_iters=10*train_epoch_len))

    # lr schedulers
    if reduce_on:
        # early stopping + reduce on plateau
        callback_fns.append(partial(
            ReduceLROnPlateauCallback, monitor='valid_loss', mode='auto',
            patience=10, factor=0.1, min_delta=0, min_lr=min_lr))
        callback_fns.append(partial(
            EarlyStoppingCallback, monitor='valid_loss', min_delta=0,
            patience=20))
    else:
        # regular step lr scheduler, as in the paper
        models_to_warm_up = ['110', '164', '1001', '1202']
        milestones = [100, 50, 50]

        # warmup for larger models
        if len([True for x in models_to_warm_up if x in model_name]) > 0:
            phases = [
                TrainingPhase(train_epoch_len * 1)
                    .schedule_hp('lr', max_lr * 0.1),
                TrainingPhase(train_epoch_len * (milestones[0] - 1))
                    .schedule_hp('lr', max_lr),
                TrainingPhase(train_epoch_len * milestones[1])
                    .schedule_hp('lr', max_lr * 0.1),
                TrainingPhase(train_epoch_len * milestones[2])
                    .schedule_hp('lr', max_lr * 0.01),
            ]
        # no warmup
        else:
            phases = [
                TrainingPhase(train_epoch_len * milestones[0])
                    .schedule_hp('lr', max_lr),
                TrainingPhase(train_epoch_len * milestones[1])
                    .schedule_hp('lr', max_lr * 0.1),
                TrainingPhase(train_epoch_len * milestones[2])
                    .schedule_hp('lr', max_lr * 0.01),
            ]
        callback_fns.append(partial(GeneralScheduler, phases=phases))

    # setting up fastai objects
    bunch = ImageDataBunch(train_loader, valid_loader, test_dl=test_loader,
                           device=device, path=data_path)
    # lr is set by fit and scheduler
    sgd = partial(torch.optim.SGD, momentum=momentum,
                  weight_decay=weight_decay, nesterov=nesterov)

    learn = Learner(bunch, model, loss_func=nn.CrossEntropyLoss(),
                    opt_func=sgd, true_wd=False, wd=weight_decay,
                    metrics=[accuracy], callback_fns=callback_fns,
                    path=logs_path, model_dir=model_saves_dir)


    print("Running on", device)
    print("-" * 50)
    print('Training', model_code)
    n_params, n_layers = num_params(model)
    n_total_params, _ = num_params(model, count_fixed=True)
    n_fixed = n_total_params - n_params
    print("Number of trainable parameters:", n_params)
    print("Number of fixed parameters:", n_fixed)

    # Training
    learn.fit(200, lr=max_lr, wd=weight_decay)

    # Gathering stats and saving them
    best_epoch, best_value = learn.save_model_callback.best_epoch, learn.save_model_callback.best
    time_to_best_epoch = learn.save_model_callback.time_to_best_epoch

    if reduce_on:
        changed_lr_on_epochs = learn.reduce_lr_on_plateau_callback\
            .changed_lr_on_epochs.keys()
    else:
        changed_lr_on_epochs = milestones

    print("Best model was found at epoch {} with accuracy value {:.4f} in {:.2f} seconds.".format(best_epoch, best_value, time_to_best_epoch))

    loss_train, accu_train = learn.validate(dl=learn.data.train_dl)
    loss_valid, accu_valid = learn.validate(dl=learn.data.valid_dl)
    loss_test,  accu_test  = learn.validate(dl=learn.data.test_dl)
    # accu_train, accu_valid, accu_test = accu_train.item(), accu_valid.item(), accu_test.item()


    val_dict = {'name': model_code,
                'accu_test': accu_test * 100,
                'n_params': format_number_km(n_params),
                'n_fixed': format_number_km(n_fixed),
                'n_total': format_number_km(n_total_params),
                'epochs': best_epoch + 1,
                'time': round(time_to_best_epoch / 3600, 2),
                'time_per_epoch': time_to_best_epoch / (best_epoch + 1),
                'changed_lr_on': changed_lr_on_epochs,
                'loss_train': loss_train,
                'loss_valid': loss_valid,
                'loss_test':  loss_test,
                'accu_train': accu_train * 100,
                'accu_valid': accu_valid * 100,
                'accu_test (again)': accu_test * 100,
                'other': ''}

    save_summary(logs_path/'models_summary.csv', val_dict)


if __name__ == '__main__':
    main()
