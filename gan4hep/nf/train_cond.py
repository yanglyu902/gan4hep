"""Trainer for Conditional Normalizing Flow
"""
import os

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from utils import train_density_estimation_cond

# %%
def compare(predictions, truths, img_dir, idx=0):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axs = axs.flatten()

    config = dict(histtype='step', lw=2, density=True)
    # phi
    idx=0
    ax = axs[idx]
    x_range = [-1, 1]
    yvals, _, _ = ax.hist(truths[:, idx], bins=40, range=x_range, label='Truth', **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator', **config)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylim(0, max_y)
    ax.legend()
    
    # theta
    idx=1
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40, range=x_range, label='Truth', **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator', **config)
    ax.set_xlabel(r"$theta$")
    ax.set_ylim(0, max_y)
    ax.legend()

    plt.savefig(os.path.join(img_dir, f'image_at_epoch_{idx}.png'))
    plt.close('all')


def evaluate(flow_model, test_in, testing_data, layers):
    num_samples, num_dims = testing_data.shape
    cond_kwargs = dict([(f"b{idx}", {"conditional_input": test_in}) for idx in range(layers)])

    samples = flow_model.sample((num_samples,), 
        **cond_kwargs).numpy()
    distances = [
        stats.wasserstein_distance(samples[:, idx], testing_data[:, idx]) \
            for idx in range(num_dims)
    ]

    return sum(distances), samples


def train(train_in, train_truth, test_in, testing_truth, flow_model, layers, lr, batch_size, max_epochs, outdir):
    base_lr = lr
    end_lr = 1e-5
    learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(
        base_lr, max_epochs, end_lr, power=0.5)


    # initialize checkpoints
    checkpoint_directory = "{}/checkpoints".format(outdir)
    os.makedirs(checkpoint_directory, exist_ok=True)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=flow_model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=None)
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()


    AUTO = tf.data.experimental.AUTOTUNE
    training_data = tf.data.Dataset.from_tensor_slices(
        (train_in, train_truth)).batch(batch_size).prefetch(AUTO)

    img_dir = os.path.join(outdir, "imgs")
    os.makedirs(img_dir, exist_ok=True)

    # start training
    print("idx, train loss, distance, minimum distance, minimum epoch")
    min_wdis, min_iepoch = 9999, -1
    delta_stop = 1000

    for i in range(max_epochs):
        for condition,batch in training_data:
            print(condition.shape, batch.shape, layers, flow_model)
            train_loss = train_density_estimation_cond(
                flow_model, opt, batch, condition, layers)

        wdis, predictions = evaluate(flow_model, test_in, testing_truth)
        if wdis < min_wdis:
            min_wdis = wdis
            min_iepoch = i
            compare(predictions, testing_truth, img_dir, i)
            ckpt_manager.save()
        elif i - min_iepoch > delta_stop:
            break

        print(f"{i}, {train_loss}, {wdis}, {min_wdis}, {min_iepoch}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Normalizing Flow')
    add_arg = parser.add_argument
    add_arg("filename", help='input filename (csv file)', default=None)
    add_arg("--log-dir", help='output directory')
    add_arg("--max-evts", default=-1, type=int, help="maximum number of events")
    add_arg("--batch-size", type=int, default=512, help="batch size") # default: 512
    
    args = parser.parse_args()

    log_dir = '/global/homes/y/yanglyu/phys_290/gan4hep/gan4hep/nf/logs/' + args.log_dir

    from made import create_conditional_flow
    from gan4hep.gan.utils import generate_and_save_images
    from gan4hep.preprocess import read_geant4
    
    # format: (X_train, X_test, y_train, y_test, xlabels)
    train_in, test_in, train_truth, test_truth, xlabels = read_geant4(
                                                        args.filename, log_dir)



    outdir = args.log_dir
    hidden_shape = [128]*2
    layers = 10
    lr = 1e-3
    batch_size = args.batch_size
    max_epochs = 1000
    conditional_event_shape=(6,)

    maf =  create_conditional_flow(hidden_shape, 
                                    layers, 
                                    conditional_event_shape, 
                                    out_dim=1)

    train(train_in, train_truth, 
            test_in, test_truth, 
            maf, layers, 
            lr, batch_size, max_epochs, outdir)