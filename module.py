# -*- coding:utf-8 -*-
"""
Author: Yiyan Zhang
Date: September 22, 2022
"""
import numpy as np
import torch
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from operator import truediv

def weight_init(layer):
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)  #
    return np.round(each_acc, 4), average_acc

def colormap(num_class, p):
    palette = [
        '#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
        '#C86400', '#00C864', '#6400C8', '#C80064', '#64C800', '#0064C8',
        '#964B4B', '#4B964B', '#4B4B96', '#FF6464'
    ]
    max_required = num_class + 1 if not p else num_class + 1  # include background in bounds check
    if max_required > len(palette):
        raise ValueError(f"Requested {max_required} colors but palette only supports {len(palette)} entries.")

    if p:
        return colors.ListedColormap(palette[:num_class + 1], N=num_class + 1)
    return colors.ListedColormap(palette[:num_class + 1], N=num_class + 1)

def dis_groundtruth(dataset, num_class, gt, p):
    '''plt.figure(title)
    plt.title(title)'''
    plt.imshow(gt, cmap=colormap(num_class, p))
    # spectral.imshow(classes=gt)
    '''plt.colorbar()'''
    plt.xticks([])
    plt.yticks([])
    '''plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)'''
    if p:
        plt.savefig('./results/{}/{}.png'.format(dataset, dataset+'true'), dpi=1200, pad_inches=0.0)
    else:
        plt.savefig('./results/{}/{}.png'.format(dataset, dataset+'false'), dpi=1200, pad_inches=0.0)
    plt.show()

