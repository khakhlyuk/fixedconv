"""

Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.

Taken from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils.plot import plot_images


def get_data_loaders(dataset, data_dir, batch_size, augment, random_seed,
                     valid_size=0.1, shuffle=True, show_sample=False,
                     num_workers=4, pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - dataset: dataset to use.
    - data_dir: path directory to the dataset. The dataset will be read from
      this folder. If dataset is not present, it will be downloaded.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # First, PIL transforms are applied, then Tensor transforms.
    if dataset == 'cifar10':
        dset = datasets.CIFAR10
        image_size = 32
        normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    elif dataset == 'cifar100':
        dset = datasets.CIFAR100
        image_size = 32
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    elif dataset == 'mnist':
        dset = datasets.MNIST
        image_size = 28
        normalize = transforms.Normalize(0.5, 0.5)
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    elif dataset == 'fmnist':
        dset = datasets.FashionMNIST
        image_size = 28
        normalize = transforms.Normalize(0.5, 0.5)
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    elif dataset == 'tiny-imagenet':
        dset = datasets.ImageFolder
        image_size = 64
        normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    assert dataset

    if augment == False:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

    if dset == datasets.ImageFolder:
        train_path = data_dir + '/test'
        val_path = data_dir + '/val'
        test_path = data_dir + '/test'
    else:
        train_path = data_dir
        val_path = data_dir
        test_path = data_dir

    # load the dataset
    train_dataset = dset(
        root=train_path, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = dset(
        root=val_path, train=True,
        download=True, transform=valid_transform,
    )
    test_dataset = dset(
        root=test_path, train=False,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    torch.manual_seed(random_seed)
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        plot_images(images, labels, unnormalize=True, interpolate=True)

    return train_loader, valid_loader, test_loader
