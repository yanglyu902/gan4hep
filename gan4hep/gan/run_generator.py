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

LOG_DIR = '/global/homes/y/yanglyu/phys_290/gan4hep/gan4hep/logs/'


def inverse_transform(gen, y_orig):
    """
    Transform generated data (between -1, 1) back to original space.
    """
    y_max, y_min = np.max(y_orig), np.min(y_orig)
    output = 1/2 * (gen + 1) * (y_max - y_min) + y_min
    return output.astype(int)

def str_to_float(x):
    """
    Example: '01' -> 0.1, '05' -> 0.5, '1' -> 1.0, '10' -> 10.0
    """
    if len(x) == 2 and x[0] == '0':
        return float(x[1]) / 10
    return x

def create_X(ngen, elow, ehigh):

    dim_cond = 4 # particle info (KE, p1, p2, p3)
    dim_material = 2 # Fe, Cu
    dim_noise = 4 # random noise

    """ 1. generate particle """

    elow = str_to_float(elow)
    ehigh = str_to_float(ehigh)

    E_min = float(elow) * 1000 # MeV
    E_max = float(ehigh) * 1000 # MeV

    print(f'Generating events with E_min = {E_min} MeV, E_max = {E_max} MeV')

    x_particle = np.random.rand(ngen, 4)
    direction = [0.6, 0.6, 0.5291502622129182]
    KE = E_min + x_particle[:,0] * (E_max - E_min)
    x_particle[:,0] = KE
    x_particle[:,1] = np.sqrt(KE**2 + 2 * KE * PionMass) * direction[0]
    x_particle[:,2] = np.sqrt(KE**2 + 2 * KE * PionMass) * direction[1]
    x_particle[:,3] = np.sqrt(KE**2 + 2 * KE * PionMass) * direction[2]

    try: # feature normalization factor saved or not?
        KE_max = np.loadtxt(FIG_DIR + '/KE_max.csv')
    except: # hope statistics is high enough
        print(FIG_DIR + '/KE_max.csv NOT FOUND!!!')
        KE_max = np.max(x_particle[:,0])

    print('Normalizing features with KE_max = ', KE_max)
    x_particle /= (KE_max + PionMass)

    """ 2. generate material """

    x_material = np.zeros((ngen, dim_material))
    x_material[:,0] = 1 # pion

    """ 3. generate noise """

    x_noise = np.random.normal(loc=0., scale=1., size=(ngen, dim_noise))

    """ 4. combine inputs """

    X_gen = np.concatenate((x_particle, x_material, x_noise), axis=1)

    return X_gen


def generate(model_name=None, ngen=100, elow=None, ehigh=None):

    # NOTE: the full GAN input should be [4_vector, material, random noise]

    model = tf.keras.models.load_model(GEN_DIR)

    X_gen = create_X(ngen, elow, ehigh)

    print('Generated input shape:', X_gen.shape)
    print('Example of generated features:')
    print(X_gen[:5])

    gen = np.squeeze(model.predict(X_gen))
    y_orig = np.loadtxt(LOG_DIR + model_name + '/y_orig.csv')
    gen_orig = inverse_transform(gen, y_orig)
    
    return gen_orig


def generate_and_compare_slices(model_name=None, ngen=100000):

    model = tf.keras.models.load_model(GEN_DIR)
    y_orig = np.loadtxt(LOG_DIR + model_name + '/y_orig.csv')

    for KE_curr in ['01', '05', '1', '5', '10', '15', '20', '30']: # GeV

        X_gen = create_X(ngen, KE_curr, KE_curr)
        print('Example of generated features:')
        print(X_gen[:5])

        gen = np.squeeze(model.predict(X_gen))
        gen_orig = inverse_transform(gen, y_orig)

        " --- make plot --- "

        y_truth = np.loadtxt('/global/homes/y/yanglyu/phys_290/MCGenerators/G4/HadronicInteractions/build/pion_' + KE_curr + 'GeV_Fe_1M.csv', usecols=(5))

        plt.figure()
        bins = np.arange(0,76,1)

        plt.hist(y_truth, density=True, histtype='step', bins=bins, label='truth', zorder=1)
        plt.hist(gen_orig, density=True, histtype='step', bins=bins, label='Generated', zorder=1)

        plt.xlabel('Number of secondary particles')
        plt.title(model_name + ': ' + KE_curr + ' GeV')
        plt.legend()

        plt.savefig(FIG_DIR + '/comparison_' + str(KE_curr) + 'GeV.png', dpi=150)
        print(f'Comparison plot saved to {FIG_DIR}')



def compare(gen_orig, model_name=None, elow=None, ehigh=None):
    
    """ N secondary space """

    y_train_orig = np.loadtxt(LOG_DIR + model_name + '/y_train_orig.csv')
    y_test_orig = np.loadtxt(LOG_DIR + model_name + '/y_test_orig.csv')
    y_orig = np.loadtxt(LOG_DIR + model_name + '/y_orig.csv')
    

    """ make plot """
    
    plt.figure()
    bins = np.arange(0,76,1)

    h_orig,b = np.histogram(y_orig, bins=bins)
    h,b = np.histogram(y_orig, bins=bins, density=True)

    # print(h)
    b = (b[1:] + b[:-1])/2
    sig_y = h_orig/np.sum(h_orig) * np.sqrt(1/h_orig + 1/np.sum(h_orig))
    plt.errorbar(b,h, yerr=sig_y, color='black', fmt='.', markersize=3, capsize=2, label='Truth, all')

    plt.hist(y_train_orig, density=True, histtype='step', bins=bins, label='training truth', alpha=0.5,lw=2)
    plt.hist(y_test_orig, density=True, histtype='step', bins=bins, label='testing truth', alpha=0.5,lw=2)
    plt.hist(gen_orig, density=True, histtype='step', bins=bins, label='Generated', lw=2)

    plt.xlabel('Number of secondary particles')
    plt.ylabel('Fraction of events')
    plt.title(model_name)
    plt.legend()

    plt.savefig(FIG_DIR + '/comparison_total.png', dpi=150)
    print(f'Comparison plot saved to {FIG_DIR}')



if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Generate events and compare!')
    add_arg = parser.add_argument

    add_arg("--log-dir", help='name of log folder', default=None)
    add_arg("--ngen", default=1000000, type=int, help='if not None, generate sample!')
    add_arg("--elow", default='05', type=str)
    add_arg("--ehigh", default='15', type=str)

    args = parser.parse_args()

    GEN_DIR = LOG_DIR + args.log_dir + '/generator'
    FIG_DIR = LOG_DIR + args.log_dir
    PionMass = 139.570 # MeV

    # gen_orig = generate(args.log_dir, args.ngen, args.elow, args.ehigh)
    # compare(gen_orig, args.log_dir, args.elow, args.ehigh)

    generate_and_compare_slices(args.log_dir, args.ngen)
