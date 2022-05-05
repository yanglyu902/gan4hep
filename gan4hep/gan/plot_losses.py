import importlib
from operator import truth
import os
import time
import yaml

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib
matplotlib.style.use('paper.mplstyle')


LOG_DIR = '/global/homes/y/yanglyu/phys_290/gan4hep/gan4hep/logs/'

def plot_losses(model_name=None):
    LOSS_DIR = LOG_DIR + model_name + '/logs/losses.txt'
    FIG_DIR = LOG_DIR + model_name

    epoch, loss_D, loss_G, tot_wdis, best_wdis, best_epoch = np.loadtxt(LOSS_DIR, delimiter=',', unpack=True)

    plt.figure(figsize=(10,8))
    plt.plot(epoch, loss_G, '.-', label='Generator loss')
    plt.plot(epoch, 0.5 * loss_D, '.-', label='Averaged Discriminator loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(FIG_DIR + '/losses.png', dpi=150)

    plt.figure(figsize=(10,8))
    plt.plot(epoch, loss_G, '.-', label='Generator loss')
    plt.plot(epoch, 0.5 * loss_D, '.-', label='Averaged Discriminator loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    delta = 4*1e-3
    plt.ylim(0.693-delta, 0.693+delta)
    plt.savefig(FIG_DIR + '/losses_zoomin.png', dpi=150)

    plt.figure(figsize=(10,8))
    plt.plot(epoch, tot_wdis, '.-', label='Total Wasserstein distance')
    plt.plot(epoch, best_wdis, '.-', label='Best Wasserstein distance: ' + str(np.round(best_wdis[-1], 5)))
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Wasserstein distance')
    plt.savefig(FIG_DIR + '/wdis.png', dpi=150)


if __name__=='__main__':


    import argparse
    parser = argparse.ArgumentParser(description='Generate events and compare!')
    add_arg = parser.add_argument

    add_arg("--log-dir", help='name of log folder', default=None)

    args = parser.parse_args()

    plot_losses(args.log_dir)