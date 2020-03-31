import argparse

from utils.data_loader import get_train_valid_loader, get_test_loader
from utils.utils import num_params, format_scientific

from fastai.vision import *
from fastai.vision.data import *

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
parser.add_argument('--bs', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
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

    data_path = Path(args.data_path)
    logs_path = Path(args.logs_path)
    model_saves_dir = Path('model_saves')  # relative to logs_path

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

    # setting up fastai objects
    bunch = ImageDataBunch(train_loader, valid_loader, test_dl=test_loader,
                           device=device, path=data_path)
    learn = Learner(bunch, model, loss_func=nn.CrossEntropyLoss(),
                    metrics=[accuracy],
                    path=logs_path, model_dir=model_saves_dir)

    # loading model weights from dict
    learn.load(model_code, with_opt=False)

    n_params, n_layers = num_params(model)
    n_total_params, _ = num_params(model, count_fixed=True)
    n_fixed = n_total_params - n_params

    loss_train, accu_train = learn.validate(dl=learn.data.train_dl)
    loss_valid, accu_valid = learn.validate(dl=learn.data.valid_dl)
    loss_test,  accu_test  = learn.validate(dl=learn.data.test_dl)
    # accu_train, accu_valid, accu_test = accu_train.item(), accu_valid.item(), accu_test.item()

    val_dict = {'name': model_code,
                'accu_test': accu_test * 100,
                'n_params': n_params,
                'n_fixed': n_fixed,
                'n_total': n_total_params,
                'loss_train': loss_train,
                'loss_valid': loss_valid,
                'loss_test':  loss_test,
                'accu_train': accu_train * 100,
                'accu_valid': accu_valid * 100,
                'accu_test (again)': accu_test * 100,
                'other': ''}
    print("-" * 50)
    print(val_dict)


if __name__ == '__main__':
    main()
