"""
This is a simple MLP-base conditional GAN.
Same as gan.py except that the conditional input is
given to the discriminator.
"""
import numpy as np
import os

import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))
gpus = tf.config.experimental.list_physical_devices("GPU")
logging.info("found {} GPUs".format(len(gpus)))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from tensorflow import keras
from tensorflow.keras import layers

import tqdm


cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return tf.reduce_mean(total_loss)

def generator_loss(fake_output):
    return tf.reduce_mean(cross_entropy(tf.ones_like(fake_output), fake_output))

class CGAN():
    def __init__(self,
        noise_dim: int = 4, gen_output_dim: int = 1, # TODO: is noise dimension 1 or 4?
        cond_dim: int = 4 + 2, disable_tqdm=False, lr=0.0001): # TODO: n material is 2 for now
        """
        noise_dim: dimension of the noises
        gen_output_dim: output dimension
        cond_dim: in case of conditional GAN, 
                  it is the dimension of the condition
        """
        self.noise_dim = noise_dim
        self.gen_output_dim = gen_output_dim
        self.cond_dim = cond_dim
        self.disable_tqdm = disable_tqdm

        self.gen_input_dim = self.noise_dim + self.cond_dim

        # ============
        # Optimizers
        # ============

        # NOTE: large gen_lr / small disc_lr lead to slower convergence (w-disc slowly descend to min). 
        end_lr = 1e-6
        gen_lr = 1e-5
        disc_lr = 1e-2
        max_epochs = 100

        # gen_lr = keras.optimizers.schedules.PolynomialDecay(gen_lr, max_epochs, end_lr, power=4)
        # disc_lr = keras.optimizers.schedules.PolynomialDecay(disc_lr, max_epochs, end_lr, power=1.0)

        gen_lr = lr
        disc_lr = lr

        self.generator_optimizer = keras.optimizers.Adam(gen_lr) # default: lr
        self.discriminator_optimizer = keras.optimizers.Adam(gen_lr) # default: lr

        # Build the critic
        self.discriminator = self.build_critic()
        self.discriminator.summary()

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()


    def build_generator(self):
        gen_input_dim = self.gen_input_dim

        model = keras.Sequential([
            keras.Input(shape=(gen_input_dim,)),
            layers.Dense(64), # TODO: original N_neural is 256 for all layers.
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Dense(64), # NOTE: generator too strong?
            layers.BatchNormalization(),
            
            layers.Dense(self.gen_output_dim),
            layers.Activation("tanh"),
        ], name='Generator')
        return model

    def build_critic(self):
        gen_output_dim = self.gen_output_dim + self.cond_dim

        model = keras.Sequential([
            keras.Input(shape=(gen_output_dim,)),
            layers.Dense(32),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Dense(32),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Dense(1, activation='sigmoid'),
        ], name='Discriminator')
        return model


    def train(self, train_truth, epochs, batch_size, test_truth, log_dir, evaluate_samples_fn, xlabels,
        train_in=None, test_in=None):
        # ======================================
        # construct testing data once for all
        # ======================================
        AUTO = tf.data.experimental.AUTOTUNE
        noise = np.random.normal(loc=0., scale=1., size=(test_truth.shape[0], self.noise_dim))

        # print(test_in.shape, noise.shape)

        test_in = np.concatenate(
            [test_in, noise], axis=1).astype(np.float32) if test_in is not None else noise


        testing_data = tf.data.Dataset.from_tensor_slices(
            (test_in, test_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO)

        # ====================================
        # Checkpoints and model summary
        # ====================================
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)
        logging.info("Loading latest checkpoint from: {}".format(checkpoint_dir))
        _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

        summary_dir = os.path.join(log_dir, "logs")
        summary_writer = tf.summary.create_file_writer(summary_dir)

        img_dir = os.path.join(log_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)

        @tf.function
        def train_step(gen_in_4vec, cond_in, truth_4vec):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_out_4vec = self.generator(gen_in_4vec, training=True)

                # =============================================================    
                # add the conditional inputs to generated and truth information
                # =============================================================
                # print(cond_in.shape, gen_out_4vec.shape, truth_4vec.shape)
                gen_out_4vec = tf.concat([cond_in, gen_out_4vec], axis=-1)
                truth_4vec = tf.concat([cond_in, truth_4vec], axis=-1)

                # apply discriminator
                real_output = self.discriminator(truth_4vec, training=True)
                fake_output = self.discriminator(gen_out_4vec, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

            return disc_loss, gen_loss

        @tf.function # NOTE: myown
        def train_step_generator(gen_in_4vec, cond_in, truth_4vec): # train generator and discriminator separately
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_out_4vec = self.generator(gen_in_4vec, training=True)
                gen_out_4vec = tf.concat([cond_in, gen_out_4vec], axis=-1)
                truth_4vec = tf.concat([cond_in, truth_4vec], axis=-1)

                real_output = self.discriminator(truth_4vec, training=False)
                fake_output = self.discriminator(gen_out_4vec, training=False)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

            return disc_loss, gen_loss

        @tf.function # NOTE: myown
        def train_step_discriminator(gen_in_4vec, cond_in, truth_4vec): # train generator and discriminator separately
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_out_4vec = self.generator(gen_in_4vec, training=False)
                gen_out_4vec = tf.concat([cond_in, gen_out_4vec], axis=-1)
                truth_4vec = tf.concat([cond_in, truth_4vec], axis=-1)

                real_output = self.discriminator(truth_4vec, training=True)
                fake_output = self.discriminator(gen_out_4vec, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

            return disc_loss, gen_loss

        best_wdis = 9999
        best_epoch = -1

        plaeto_threashold = 100000
        num_no_improve = 0
        do_disc_only = False
        num_disc_only_epochs = 0
        i_disc_only = 0
        losses_logfile = os.path.join(summary_dir, 'losses.txt')

        with tqdm.trange(epochs, disable=self.disable_tqdm) as t0:
            for epoch in t0:

                # compose the training dataset by generating different noises for each epochs
                noise = np.random.normal(loc=0., scale=1., size=(train_truth.shape[0], self.noise_dim))
                train_inputs = np.concatenate(
                    [train_in, noise], axis=1).astype(np.float32) if train_in is not None else noise

                dataset = tf.data.Dataset.from_tensor_slices(
                    (train_inputs, train_in, train_truth)
                    ).shuffle(2*batch_size).batch(batch_size, drop_remainder=True).prefetch(AUTO)
                
                # NOTE: original training method 
                # ''' 
                tot_loss = []
                for data_batch in dataset:
                    tot_loss.append(list(train_step(*data_batch)))
                # '''

                # NOTE: myown training method: train more discriminator, less generator
                '''
                tot_loss = []
                n_cnt = 0
                for data_batch in dataset:

                    tot_loss.append(list(train_step_discriminator(*data_batch)))

                    n_cnt += 1
                    
                    if n_cnt < 8: # train discriminator N times
                        continue

                    tot_loss.append(list(train_step(*data_batch)))
                    n_cnt = 0
                '''

                # NOTE: another method:
                '''
                tot_loss = []
                train_fn = train_step_discriminator if do_disc_only else train_step
                # train_fn = train_step
                i_disc_only += do_disc_only
                for data_batch in dataset:
                    tot_loss.append(list(train_fn(*data_batch)))
                '''

                tot_loss = np.array(tot_loss)
                avg_loss = np.sum(tot_loss, axis=0)/tot_loss.shape[0]
                loss_dict = dict(D_loss=avg_loss[0], G_loss=avg_loss[1])

                tot_wdis = evaluate_samples_fn(self.generator, epoch, testing_data, summary_writer, img_dir, xlabels, **loss_dict)
                if tot_wdis < best_wdis:
                    ckpt_manager.save()
                    self.generator.save(log_dir + "/generator")
                    best_wdis = tot_wdis
                    best_epoch = epoch
                else:
                    num_no_improve += 1

                t0.set_postfix(**loss_dict, BestD=best_wdis, BestE=best_epoch)
                with open(losses_logfile, 'a') as f:
                    f.write("{},{},{},{},{},{}\n".format(epoch, avg_loss[0], avg_loss[1], tot_wdis, best_wdis, best_epoch))

                
                # so long since last improvement
                # train discriminator only
                if num_no_improve > plaeto_threashold:
                    num_no_improve = 0
                    num_disc_only_epochs += 1
                    do_disc_only = True
                else:
                    if i_disc_only == num_disc_only_epochs:
                        do_disc_only = False
                        i_disc_only = 0





        tmp_res = "Best Model in {} Epoch with a Wasserstein distance {:.4f}".format(best_epoch, best_wdis)
        logging.info(tmp_res)
        summary_logfile = os.path.join(summary_dir, 'results.txt')

        with open(summary_logfile, 'a') as f:
            f.write(tmp_res + "\n")
        



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train The GAN')
    add_arg = parser.add_argument
    add_arg("filename", help='input filename (csv file)', default=None)
    add_arg("--epochs", help='number of maximum epochs', default=100, type=int) # TODO: before was 100
    add_arg("--log-dir", help='log directory') # TODO: must specify; use naming similar to the filename

    add_arg("--num-test-evts", help='number of testing events', default=10000, type=int)
    add_arg("-v", '--verbose', help='tf logging verbosity', default='INFO',
        choices=['WARN', 'INFO', "ERROR", "FATAL", 'DEBUG'])
    add_arg("--batch-size", help='Batch size', type=int, default=128)
    args = parser.parse_args()

    logging.set_verbosity(args.verbose)

    log_dir = '/global/homes/y/yanglyu/phys_290/gan4hep/gan4hep/logs/' + args.log_dir

    from gan4hep.gan.utils import generate_and_save_images
    from gan4hep.preprocess import read_geant4
    
    # format: (X_train, X_test, y_train, y_test, xlabels)
    train_in, test_in, train_truth, test_truth, xlabels = read_geant4(args.filename, log_dir)

    batch_size = args.batch_size
    gan = CGAN()
    gan.train(
        train_truth, args.epochs, batch_size,
        test_truth, log_dir,
        generate_and_save_images,
        xlabels,
        train_in, test_in
    )


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description='Train The GAN')
#     add_arg = parser.add_argument
#     add_arg("filename", help='input filename', default=None)
#     add_arg("--epochs", help='number of maximum epochs', default=100, type=int)
#     add_arg("--log-dir", help='log directory', default='log_training')
#     add_arg("--num-test-evts", help='number of testing events', default=10000, type=int)
#     add_arg("--inference", help='perform inference only', action='store_true')
#     add_arg("-v", '--verbose', help='tf logging verbosity', default='INFO',
#         choices=['WARN', 'INFO', "ERROR", "FATAL", 'DEBUG'])
#     add_arg("--max-evts", help='Maximum number of events', type=int, default=None)
#     add_arg("--batch-size", help='Batch size', type=int, default=512)
#     args = parser.parse_args()

#     logging.set_verbosity(args.verbose)


#     from gan4hep.utils_gan import generate_and_save_images
#     from gan4hep.preprocess import herwig_angles

#     train_in, train_truth, test_in, test_truth = herwig_angles(args.filename, args.max_evts)

#     batch_size = args.batch_size
#     gan = CGAN()
#     gan.train(
#         train_truth, args.epochs, batch_size,
#         test_truth, args.log_dir,
#         generate_and_save_images,
#         train_in, test_in
#     )