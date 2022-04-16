import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def shuffle(array: np.ndarray):
    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    np_rs = RandomState(MT19937(SeedSequence(123456789)))
    np_rs.shuffle(array)


def read_dataframe(filename, sep=",", engine=None):
    if type(filename) == list:
        print(filename)
        df_list = [
            pd.read_csv(f, sep=sep, header=None, names=None, engine=engine)
                for f in filename
        ]
        df = pd.concat(df_list, ignore_index=True)
        filename = filename[0]
    else:
        df = pd.read_csv(filename, sep=sep, 
                    header=None, names=None, engine=engine)
    return df
   

def read_geant4(filename, log_dir):
    PionMass = 139.570 # MeV

    filename = "/global/homes/y/yanglyu/phys_290/MCGenerators/G4/HadronicInteractions/build/" + filename + '.csv'
    df = pd.read_csv(filename, sep=' ', usecols=[0,1,2,3,4,5], header=None) # 4 vector and num_secondary

    data = df.to_numpy().astype(np.float32)
    curr_material = data[:,0]
    X = data[:, 1:-1] # NOTE: which columns to use?
    y = data[:, -1]
    y_orig = y.copy()

    # NOTE: normalize and standardize: scale features to [0, 1], and scale labels to [-1, 1]
    # X /= np.max(X, axis=0) # input in [0, 1] # ALERT: really?? correlation between cols are missing!
    # X = (X.T / (X[:,0] + 139)).T
    X /= (np.max(X[:,0]) + PionMass)
    y = 2 * (y - np.min(y))/(np.max(y) - np.min(y)) - 1  # label in [-1, 1]

    # shuffle and split data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    train_ind, test_ind = train_test_split(indices, train_size=0.8)
    
    X_train, y_train = X[train_ind], y[train_ind, None]
    X_test, y_test = X[test_ind], y[test_ind, None]

    # one-hot encoding material info: 2 materials for now. 
    material = np.zeros((X.shape[0], 2), dtype=np.float32)
    material[:,0] = (curr_material == -211).astype(np.float32)
    material[:,1] = (curr_material == -321).astype(np.float32)

    X_train = np.concatenate((X_train, material[train_ind]), axis=1)
    X_test = np.concatenate((X_test, material[test_ind]), axis=1)
    
    # NOTE: the full GAN input should be [4_vector, material, random noise]

    print('train/test shapes:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print('example of training data:', X_train[:10], y_train[:10])

    xlabels = ['num_secondary']

    # output to csv for later comparison plot
    os.mkdir(log_dir)
    np.savetxt(log_dir + '/y_train.csv', y[train_ind])
    np.savetxt(log_dir + '/y_test.csv', y[test_ind])
    np.savetxt(log_dir + '/y_orig.csv', y_orig)
    np.savetxt(log_dir + '/y_train_orig.csv', y_orig[train_ind])
    np.savetxt(log_dir + '/y_test_orig.csv', y_orig[test_ind])

    return (X_train, X_test, y_train, y_test, xlabels)

    

def herwig_angles(filename,
        max_evts=None, testing_frac=0.1):
    """
    This reads the Herwig dataset where one cluster decays
    into two particles.
    In this case, we ask the GAN to predict the theta and phi
    angle of one of the particles
    """
    df = read_dataframe(filename, engine='python')

    event = None
    with open(filename, 'r') as f:
        for line in f:
            event = line
            break
    particles = event[:-2].split(';')

    input_4vec = df[0].str.split(",", expand=True)[[4, 5, 6, 7]].to_numpy().astype(np.float32)
    out_particles = []
    for idx in range(1, len(particles)):
        out_4vec = df[idx].str.split(",", expand=True).to_numpy()[:, -4:].astype(np.float32)
        out_particles.append(out_4vec)

    # ======================================
    # Calculate the theta and phi angle 
    # of the first outgoing particle
    # ======================================
    out_4vec = out_particles[0]
    px = out_4vec[:, 1].astype(np.float32)
    py = out_4vec[:, 2].astype(np.float32)
    pz = out_4vec[:, 3].astype(np.float32)
    pT = np.sqrt(px**2 + py**2)
    phi = np.arctan(px/py)
    theta = np.arctan(pT/pz)

    # <NOTE, inputs and outputs are scaled to be [-1, 1]>
    max_phi = np.max(np.abs(phi))
    max_theta = np.max(np.abs(theta))
    scales = np.array([max_phi, max_theta], np.float32)

    truth_in = np.stack([phi, theta], axis=1) / scales

    shuffle(truth_in)
    shuffle(input_4vec)


    # Split the data into training and testing
    # <HACK, FIXME, NOTE>
    # <HACK, For now a maximum of 10,000 events are used for testing, xju>
    num_test_evts = int(input_4vec.shape[0]*testing_frac)
    if num_test_evts < 10_000: num_test_evts = 10_000

    # <NOTE, https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html>



    test_in, train_in = input_4vec[:num_test_evts], input_4vec[num_test_evts:max_evts]
    test_truth, train_truth = truth_in[:num_test_evts], truth_in[num_test_evts:max_evts]

    xlabels = ['phi', 'theta']

    return (train_in, train_truth, test_in, test_truth, xlabels)

def herwig_angles2(filename,
        max_evts=None, testing_frac=0.1, mode=2):
    """
    This Herwig dataset is for the "ClusterDecayer" study.
    Each event has q1, q1, cluster, h1, h2.
    I define 3 modes:
    0) both q1, q2 are with Pert=1
    1) only one of q1 and q2 is with Pert=1
    2) neither q1 nor q2 are with Pert=1
    3) at least one quark with Pert=1
    """
    if type(filename) == list:
        filename = filename[0]
    arrays = np.load(filename)
    truth_in = arrays['out_truth']
    input_4vec = arrays['input_4vec']

    shuffle(truth_in)
    shuffle(input_4vec)
    print(truth_in.shape, input_4vec.shape)


    # Split the data into training and testing
    # <HACK, FIXME, NOTE>
    # <HACK, For now a maximum of 10,000 events are used for testing, xju>
    num_test_evts = int(input_4vec.shape[0]*testing_frac)
    if num_test_evts < 10_000: num_test_evts = 10_000

    test_in, train_in = input_4vec[:num_test_evts], input_4vec[num_test_evts:max_evts]
    test_truth, train_truth = truth_in[:num_test_evts], truth_in[num_test_evts:max_evts]
    xlabels = ['phi', 'theta']

    return (train_in, train_truth, test_in, test_truth, xlabels)


def dimuon_inclusive(filename, max_evts=None, testing_frac=0.1):
    
    df = read_dataframe(filename, " ", None)
    truth_data = df.to_numpy().astype(np.float32)
    print(f"reading dimuon {df.shape[0]} events from file {filename}")

    scaler = MinMaxScaler(feature_range=(-1,1))
    truth_data = scaler.fit_transform(truth_data)
    # scales = np.array([10, 1, 1, 10, 1, 1], np.float32)
    # truth_data = truth_data / scales

    shuffle(truth_data)

    num_test_evts = int(truth_data.shape[0]*testing_frac)
    if num_test_evts > 10_000: num_test_evts = 10_000


    test_truth, train_truth = truth_data[:num_test_evts], truth_data[num_test_evts:max_evts]

    xlabels = ['leading Muon {}'.format(name) for name in ['pT', 'eta', 'phi']] +\
              ['subleading Muon {}'.format(name) for name in ['pT', 'eta', 'phi']]
    
    return (None, train_truth, None, test_truth, xlabels)


if __name__== '__main__':
    read_geant4('pion_25GeV_Fe_100K.csv')
    # read_geant4('pion_25GeV_Fe_1M.csv')