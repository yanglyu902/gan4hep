import importlib
from operator import truth
import os
import time
import yaml

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# from gan4hep.utils_plot import compare
from gan4hep.graph import read_dataset, loop_dataset
from gan4hep import data_handler as DataHandler
from gan4hep.preprocess import read_geant4

LOG_DIR = '/global/homes/y/yanglyu/phys_290/gan4hep/gan4hep/logs/'

def generate(model_name=None, ngen=100):

    # NOTE: the full GAN input should be [4_vector, material, random noise]

    MODEL_DIR = LOG_DIR + model_name
    GEN_DIR = MODEL_DIR + '/generator'
    SAVE_DIR = MODEL_DIR + '/generated_sample.csv'

    model = tf.keras.models.load_model(GEN_DIR)

    gen_out = []

    dim_cond = 4 # particle 4 vector
    dim_material = 2 # FIXME: 1 and 0, not all ones.
    dim_noise = 4 # random noise

    # material
    x_material = np.zeros((ngen, dim_material))
    x_material[:,0] = 1 # pion

    # random noise
    x_noise = np.random.normal(loc=0., scale=1., size=(ngen, dim_noise))

    # particle
    PionMass = 139.570
    E_min = 0.5 # GeV
    E_max = 15.0 # GeV

    x_particle = np.random.rand(ngen, 4)
    KE = E_min + x_particle[:,0] * (E_max - E_min)
    x_particle[:,0] = KE
    x_particle[:,1] = np.sqrt(KE**2 + 2 * KE * PionMass) * 0.6
    x_particle[:,2] = np.sqrt(KE**2 + 2 * KE * PionMass) * 0.6
    x_particle[:,3] = np.sqrt(KE**2 + 2 * KE * PionMass) * 0.5291502622129182

    x_particle = x_particle / np.max(x_particle, axis=0)

    X_input = np.concatenate((x_particle, x_material), axis=1)
    X_input = np.concatenate((X_input, x_noise), axis=1)

    print('Example of generated features: ', X_input[:10])

    gen_out = np.squeeze(model.predict(X_input))

    np.savetxt(SAVE_DIR, gen_out)
    print(f'Generated {ngen} events and saved to {SAVE_DIR}')
    
    return SAVE_DIR


def compare_slices(model_name=None):
    
    CSV_DIR = '/global/homes/y/yanglyu/phys_290/MCGenerators/G4/HadronicInteractions/build/' + model_name + '.csv'
    FIG_DIR = LOG_DIR + model_name
    
    # gen = np.loadtxt(LOG_DIR + model_name + '/generated_sample.csv')

    # y = np.concatenate((y_train, y_test))

    """ N secondary space """

    gen = np.loadtxt(LOG_DIR + model_name + '/generated_sample.csv')
    y_train_orig = np.loadtxt(LOG_DIR + model_name + '/y_train_orig.csv')
    y_test_orig = np.loadtxt(LOG_DIR + model_name + '/y_test_orig.csv')
    y_orig = np.loadtxt(LOG_DIR + model_name + '/y_orig.csv')
    
    y_max, y_min = np.max(y_orig), np.min(y_orig)
    def transform(x):
        return 1/2 * (x + 1) * (y_max - y_min) + y_min
    gen = transform(gen).astype(int)


    plt.figure()
    bins = np.arange(0,76,1)

    h,b = np.histogram(y_orig, bins=bins, density=True)
    b = (b[1:] + b[:-1])/2
    sig_y = h/len(y_orig) * np.sqrt(1/h + 1/len(y_orig))
    plt.errorbar(b,h, yerr=sig_y, color='black', fmt='.', markersize=3, capsize=2, label='Truth, all', zorder=0)

    plt.hist(y_train_orig, density=True, histtype='step', bins=bins, label='training truth', zorder=1)
    plt.hist(y_test_orig, density=True, histtype='step', bins=bins, label='testing truth', zorder=1)
    plt.hist(gen, density=True, histtype='step', bins=bins, label='Generated', zorder=1)

    plt.xlabel('Number of secondary particles')
    plt.title(model_name)
    plt.legend()

    plt.savefig(FIG_DIR + '/comparison_nsec.png', dpi=150)
    print(f'Comparison plot saved to {FIG_DIR}')


def compare(model_name=None):
    
    CSV_DIR = '/global/homes/y/yanglyu/phys_290/MCGenerators/G4/HadronicInteractions/build/' + model_name + '.csv'
    FIG_DIR = LOG_DIR + model_name
    
    # gen = np.loadtxt(LOG_DIR + model_name + '/generated_sample.csv')

    # y = np.concatenate((y_train, y_test))

    """ N secondary space """

    gen = np.loadtxt(LOG_DIR + model_name + '/generated_sample.csv')
    y_train_orig = np.loadtxt(LOG_DIR + model_name + '/y_train_orig.csv')
    y_test_orig = np.loadtxt(LOG_DIR + model_name + '/y_test_orig.csv')
    y_orig = np.loadtxt(LOG_DIR + model_name + '/y_orig.csv')
    
    y_max, y_min = np.max(y_orig), np.min(y_orig)
    def transform(x):
        return 1/2 * (x + 1) * (y_max - y_min) + y_min
    gen = transform(gen).astype(int)


    plt.figure()
    bins = np.arange(0,76,1)

    h,b = np.histogram(y_orig, bins=bins, density=True)
    b = (b[1:] + b[:-1])/2
    sig_y = h/len(y_orig) * np.sqrt(1/h + 1/len(y_orig))
    plt.errorbar(b,h, yerr=sig_y, color='black', fmt='.', markersize=3, capsize=2, label='Truth, all', zorder=0)

    plt.hist(y_train_orig, density=True, histtype='step', bins=bins, label='training truth', zorder=1)
    plt.hist(y_test_orig, density=True, histtype='step', bins=bins, label='testing truth', zorder=1)
    plt.hist(gen, density=True, histtype='step', bins=bins, label='Generated', zorder=1)

    plt.xlabel('Number of secondary particles')
    plt.title(model_name)
    plt.legend()

    plt.savefig(FIG_DIR + '/comparison_nsec.png', dpi=150)
    print(f'Comparison plot saved to {FIG_DIR}')


if __name__=='__main__':


    import argparse
    parser = argparse.ArgumentParser(description='Generate events and compare!')
    add_arg = parser.add_argument

    add_arg("model_name", help='name of log folder', default=None)
    add_arg("--ngen", default=None, type=int, help='if not None, generate sample!')

    args = parser.parse_args()

    if args.ngen:
        generate(args.model_name, args.ngen)

    compare_slices(args.model_name) # NOTE: slices or just one comparison?
