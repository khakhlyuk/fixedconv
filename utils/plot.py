import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from utils.data_stats import MEANS, STDS


def plot_images(images, cls_true=None, cls_pred=None, interpolate=False, unnormalize=False):
    """
    Adapted from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
    """
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        img = images[i]
        
        if unnormalize:
            means = torch.tensor(MEANS).view(3, 1, 1)
            stds = torch.tensor(STDS).view(3, 1, 1)
            img = img * stds + means
        
        if interpolate:
            interpolation = 'spline16'
        else: 
            interpolation = 'none'

        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation=interpolation)

        if cls_true is not None:
            # show true & predicted classes
            cls_true_name = LABEL_NAMES[cls_true[i]]
            if cls_pred is None:
                xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
            else:
                cls_pred_name = LABEL_NAMES[cls_pred[i]]
                xlabel = "True: {0}\nPred: {1}".format(
                    cls_true_name, cls_pred_name
                )
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()


def imshow(img, interpolate=False, unnormalize=False):

    if len(img.size()) == 4:
        imshow(torchvision.utils.make_grid(img), interpolate, unnormalize)
    else:
        if unnormalize:
            means = torch.tensor(MEANS).view(3, 1, 1)
            stds = torch.tensor(STDS).view(3, 1, 1)
            img = img * stds + means

        if interpolate:
            interpolation = 'spline16'
        else:
            interpolation = 'none'

        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation=interpolation)
        plt.show()
