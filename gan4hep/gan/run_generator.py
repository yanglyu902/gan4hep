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

def generate(model_name=None, ngen=100, elow=None, ehigh=None):

    # NOTE: the full GAN input should be [4_vector, material, random noise]

    GEN_DIR = LOG_DIR + model_name + '/generator'
    SAVE_DIR = LOG_DIR + model_name + '/generated_sample_' + str(elow) + '-' + str(ehigh) + 'GeV.csv'

    model = tf.keras.models.load_model(GEN_DIR)

    dim_cond = 4 # particle 4 vector
    dim_material = 2 # FIXME: 1 and 0, not all ones.
    dim_noise = 4 # random noise

    """ 1. generate particle """

    PionMass = 139.570 # MeV
    E_min = float(elow) * 1000 # MeV
    E_max = float(ehigh) * 1000 # MeV

    x_particle = np.random.rand(ngen, 4)
    direction = [0.6, 0.6, 0.5291502622129182]
    KE = E_min + x_particle[:,0] * (E_max - E_min)
    x_particle[:,0] = KE
    x_particle[:,1] = np.sqrt(KE**2 + 2 * KE * PionMass) * direction[0]
    x_particle[:,2] = np.sqrt(KE**2 + 2 * KE * PionMass) * direction[1]
    x_particle[:,3] = np.sqrt(KE**2 + 2 * KE * PionMass) * direction[2]

    # x_particle = x_particle / np.max(x_particle, axis=0) # ALERT: careful!!!! could be wrong!!!
    # x_particle = (x_particle.T / (x_particle[:,0] + PionMass)).T # NOTE: 139 is the pion mass
    x_particle /= (np.max(x_particle[:,0]) + PionMass)


    """ 2. generate material """

    x_material = np.zeros((ngen, dim_material))
    x_material[:,0] = 1 # pion


    """ 3. generate noise """
    x_noise = np.random.normal(loc=0., scale=1., size=(ngen, dim_noise))

    """ combine inputs """
    X_gen = np.concatenate((x_particle, x_material, x_noise), axis=1)

    print('Generated input shape:', X_gen.shape)
    print('Example of generated features:')
    print(X_gen[:5])

    gen = np.squeeze(model.predict(X_gen))
    y_orig = np.loadtxt(LOG_DIR + model_name + '/y_orig.csv')
    gen_orig = inverse_transform(gen, y_orig)

    np.savetxt(SAVE_DIR, gen_orig)
    print(f'Generated {ngen} events and saved to {SAVE_DIR}')
    
    return gen_orig

def generate_and_compare_slices(model_name=None, ngen=10000):

    MODEL_DIR = LOG_DIR + model_name
    GEN_DIR = MODEL_DIR + '/generator'
    # SAVE_DIR = MODEL_DIR + '/generated_sample_slices.csv'
    FIG_DIR = LOG_DIR + model_name

    model = tf.keras.models.load_model(GEN_DIR)

    dim_cond = 4 # particle 4 vector
    dim_material = 2 # FIXME: 1 and 0, not all ones.
    dim_noise = 4 # random noise

    # material
    x_material = np.zeros((ngen, dim_material))
    x_material[:,0] = 1 # pion

    for KE_curr in ['05', '1', '10', '15']: # GeV
    
        " --- generate --- "
        # random noise
        x_noise = np.random.normal(loc=0., scale=1., size=(ngen, dim_noise))

        # particle
        PionMass = 139#.570 # MeV

        x_particle = np.random.rand(ngen, 4)
        
        if KE_curr == '05':
            KE = 500.0
        else:
            KE = float(KE_curr) * 1000.0

        x_particle[:,0] = KE
        x_particle[:,1] = np.sqrt(KE**2 + 2 * KE * PionMass) * 0.6
        x_particle[:,2] = np.sqrt(KE**2 + 2 * KE * PionMass) * 0.6
        x_particle[:,3] = np.sqrt(KE**2 + 2 * KE * PionMass) * 0.5291502622129182
        # print(x_particle[:5])
        # x_particle = x_particle / np.max(x_particle, axis=0) # ALERT: careful!!!! could be wrong!!!
        x_particle /= (np.max(x_particle[:,0]) + PionMass)

        X_input = np.concatenate((x_particle, x_material), axis=1)
        X_input = np.concatenate((X_input, x_noise), axis=1)

        print('Example of generated features: ', X_input[:3])

        gen_out = np.squeeze(model.predict(X_input))
        print('gen_out:', gen_out[:10])
        # np.savetxt(SAVE_DIR, gen_out)
        # print(f'Generated {ngen} events and saved to {SAVE_DIR}')

        " --- make plot --- "

        gen = gen_out
        y_train_orig = np.loadtxt(LOG_DIR + model_name + '/y_train_orig.csv')
        y_test_orig = np.loadtxt(LOG_DIR + model_name + '/y_test_orig.csv')
        y_orig = np.loadtxt(LOG_DIR + model_name + '/y_orig.csv')
        
        y_max, y_min = np.max(y_orig), np.min(y_orig)
        def inverse_transform(x):
            return 1/2 * (x + 1) * (y_max - y_min) + y_min
        gen = inverse_transform(gen).astype(int)

        y_truth = np.loadtxt('/global/homes/y/yanglyu/phys_290/MCGenerators/G4/HadronicInteractions/build/pion_' + KE_curr + 'GeV_Fe_1M.csv', usecols=(5))

        plt.figure()
        bins = np.arange(0,76,1)

        plt.hist(y_truth, density=True, histtype='step', bins=bins, label='truth', zorder=1)
        plt.hist(gen, density=True, histtype='step', bins=bins, label='Generated', zorder=1)

        plt.xlabel('Number of secondary particles')
        plt.title(model_name)
        plt.legend()

        plt.savefig(FIG_DIR + '/comparison_nsec_' + str(KE_curr) + '.png', dpi=150)
        print(f'Comparison plot saved to {FIG_DIR}')

def inverse_transform(gen, y_orig):
    """
    Transform generated data (between -1, 1) back to original space.
    """
    y_max, y_min = np.max(y_orig), np.min(y_orig)
    output = 1/2 * (gen + 1) * (y_max - y_min) + y_min
    return output.astype(int)


def compare(model_name=None, elow=None, ehigh=None):
    
    FIG_DIR = LOG_DIR + model_name
    
    """ N secondary space """

    gen_orig = np.loadtxt(LOG_DIR + model_name + '/generated_sample_' + str(elow) + '-' + str(ehigh) + 'GeV.csv')
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

    plt.savefig(FIG_DIR + '/comparison_direct.png', dpi=150)
    print(f'Comparison plot saved to {FIG_DIR}')

if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Generate events and compare!')
    add_arg = parser.add_argument

    add_arg("--log-dir", help='name of log folder', default=None)
    add_arg("--ngen", default=None, type=int, help='if not None, generate sample!')
    add_arg("--elow", default=None, type=str)
    add_arg("--ehigh", default=None, type=str)

    args = parser.parse_args()

    generate(args.log_dir, args.ngen, args.elow, args.ehigh)
    compare(args.log_dir, args.elow, args.ehigh)

    # generate_and_compare_slices(args.log_dir, args.ngen)
