import tensorflow as tf
print(tf.__version__)
import os
print(os.system('which python'))
import tensorflow_addons as tfa
from tensorflow.train import latest_checkpoint
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import sys
import numpy as np
from architectures.blocks import upblock, downblock, conv2d_normal_reg
from architectures.blocks import StretchLayer
from losses import weighted_MSE_clumps_only, weighted_MSE_with_gals
from test_utils import find_peaks


class UNet:

    """UNet

    Attributes
    ----------
    input_shape : int
        The input shape of the model

    block_size : int
        Number of convolution layers in each block

    channels : list of int
        List of channels for the successive downblocks

    """
    def __init__(self,
                 input_shape, channels,
                 n_classes,
                 block_size,
                 last_activation,
                 training_path,
                 optimizers,
                 experiment
                 ):

        self.input_shape = input_shape
        self.optimizers = optimizers
        self.last_activation = last_activation
        self.channels = channels
        self.n_classes = n_classes
        if n_classes == 1:
            self.loss_fn = weighted_MSE_clumps_only
        else:
            self.loss_fn = weighted_MSE_with_gals

        self.block_size = block_size
        self.optimizers = optimizers
        self.training_path = training_path
        self.experiment = experiment
        self._setup_model()

    def _setup_model(self):

        self.model = self._init_Unet()
        optimizers_and_layers = [(self.optimizers[0],
                                  self.model.layers[0]),
                                 (self.optimizers[1],
                                  self.model.layers[1:])]

        self.optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

    def _init_Unet(self):

        """
        Unet
        The unet is a submodel of the probabilistic Unet.
        I don't think there is a need of making the Encoder
        and the decoder sub-submodels.
        The Encoder use the downblock function,
        the Decoder the upblock function.

        The Input is a Tensor((None, 128, 128, 1), dtype=float32)

        The output is a Tensor((None, 128, 128, n), dtype=float32)), with $n>1$
        """

        unet_input = tfk.layers.Input(shape=self.input_shape, name='Image')
        stretched = StretchLayer(units=1, name='stretching')(unet_input)
        features = [stretched]
        for i, n_channels in enumerate(self.channels):
            if i == 0:
                downsample = False
            else:
                downsample = True

            features.append(downblock(features[-1],
                                      n_channels,
                                      block_size=self.block_size,
                                      downsample=downsample,
                                      name=f'Unet_downblock_{i}'))
        encoder_output = features[1:]

        n = len(encoder_output) - 2

        lower_reso_features = encoder_output[-1]
        for i in range(n, -1, -1):
            same_reso_features = encoder_output[i]
            n_channels = self.channels[i]
            lower_reso_features = upblock(lower_reso_features,
                                          same_reso_features,
                                          output_channels=n_channels,
                                          block_size=self.block_size,
                                          name=f'Unet_upblock_{n-i}')

        unet_output = conv2d_normal_reg(lower_reso_features,
                                        self.n_classes,
                                        kernel_size=3,
                                        activation=self.last_activation,
                                        name='last_unet_deconv')

        return tfk.Model(unet_input, unet_output, name='unet')

    def eval_step(self, features, labels):
        unet_output = self.model([features])
        losses = self.loss_fn(labels, unet_output)
        return losses

    def train_step(self, features, labels):

        """
        Run the unet in training mode (training_model),
             compute the loss and update the weights of all the submodels
             for one batch.
        Return the total loss, the reconstruction loss and the KL. """

        with tf.GradientTape() as tape:
            unet_output = self.model([features])
            losses = self.loss_fn(labels, unet_output)
            total_loss = losses[-1]
            if tf.math.is_nan(total_loss):
                sys.exit('Stopping training, Loss has diverged (NaN)')
            # print(self.model.trainable_variables[0])

            grads = tape.gradient(total_loss,
                                  self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_variables))
        return losses

    def train(self, train_data, epochs,
              log_metric_frequency, plot_frequency,
              save_weights_epochs,
              ):

        global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)

        for epoch in range(epochs):
            print(f'epoch Number : {epoch}')
            for (image, label) in train_data:
                if global_steps % 200 == 0:
                    print(global_steps)
                losses = self.train_step(image, label)

                if (global_steps % log_metric_frequency == 0):
                    "Log metrics and images to comet"

                    lr_stretch = self.optimizers[0].lr.numpy()
                    self.experiment.log_metric('stretching learning rate',
                                               lr_stretch,
                                               global_steps.numpy())

                    lr_model = self.optimizers[1].lr.numpy()
                    self.experiment.log_metric('model learning rate',
                                               lr_model,
                                               global_steps.numpy())

                    with self.experiment.train():
                        self.experiment.log_metric('total_loss',
                                                   losses[0],
                                                   global_steps.numpy())
                        stretching = self.model.layers[1].get_weights()[0]
                        self.experiment.log_metric('stretching',
                                                   stretching,
                                                   global_steps.numpy())
                        if self.n_classes == 2:
                            self.experiment.log_metric('clumps loss',
                                                       losses[0],
                                                       global_steps.numpy())
                            self.experiment.log_metric('galaxy loss',
                                                       losses[1],
                                                       global_steps.numpy())
                            

                            # self.experiment.log_metric('stretching',
                            #                            losses[1],
                            #                            global_steps.numpy())

                if (global_steps % plot_frequency == 0):
                    predictions = self.model([image])
                    if self.n_classes == 1:
                        n_cols = 3
                    else:
                        n_cols = 5
                    fig, ax = plt.subplots(10, n_cols, figsize=(5, 15))
                    # fig, ax = plt.subplots(2, n_cols, figsize=(15, 5))
                    for i in range(10):
                    # for i in range(1):
                        galaxy_positions = find_peaks(predictions[i, :, :, 1],
                                                      0.1)
                        clumps_positions = find_peaks(predictions[i, :, :, 0],
                                                      0.1)
                        [axe.set_xticks([]) for axe in ax[i, :]]
                        [axe.set_yticks([]) for axe in ax[i, :]]
                        ax[i, 0].set_title(f'input, {np.max(image[i])}',
                                           fontsize=4)
                        ax[i, 0].imshow(image[i], cmap='bone')
                        ax[i, 1].imshow(label[i, :, :, 0],
                                        cmap='bone', vmin=0, vmax=1)

                        ax[i, 1].set_title(f'label, {np.max(label[i, :, :, 0])}',
                                           fontsize=4)
                        ax[i, 2].imshow(predictions[i, :, :, 0],
                                        cmap='bone', vmin=0, vmax=1)
                        
                        ax[i, 2].scatter(clumps_positions[:, 1],
                                         clumps_positions[:, 0],
                                         marker='.',
                                         s=0.3,
                                         c='red')

                        nans = np.count_nonzero(np.isnan(predictions[i, :, :, 1]))
                        if nans > 0:
                            sys.exit('Nan in output')

                        ax[i, 2].set_title(f'Pred clumps {nans}', fontsize=4)
                        if self.n_classes == 2:
                            ax[i, 3].imshow(label[i, :, :, 1],
                                            cmap='bone', vmin=0, vmax=1)
                            ax[i, 4].imshow(predictions[i, :, :, 1],
                                            cmap='bone', vmin=0, vmax=1)

                            ax[i, 4].scatter(galaxy_positions[:, 1],
                                             galaxy_positions[:, 0],
                                             marker='.',
                                             s=0.3,
                                             c='red')
                    ax[0, 0].set_title('Input', fontsize=4)
                    ax[0, 1].set_title('Gt glumps', fontsize=4)

                    if self.n_classes == 2:
                        ax[0, 3].set_title('Gtalaxy g', fontsize=4)
                        ax[0, 4].set_title('Pred galaxy', fontsize=4)
                    self.experiment.log_figure('Image training',
                                               fig,
                                               global_steps.numpy())
                    plt.close()
                if (epoch % save_weights_epochs == 0):
                    self.save_weights(epoch)

                global_steps.assign_add(1)

    def save_weights(self, epoch):
        """ Save the weights of all the submodels, with a name corresponding
         to the current training step """
        self.model.save_weights(f'{self.training_path}unet_weights' +
                                f'epoch_{epoch+1}')
        return ()

    def load_weights(self):
        """ Load the weights into all the submodels of the PUnet,
         according to the epoch written in the `checkpoint`
         file inside the checkpoint_path"""
        self.model.load_weights(latest_checkpoint(self.training_path +
                                                  'unet_weights/'))
        return ()

    def print_models(self, path, show_shape=True):
        tfk.utils.plot_model(self.model, to_file=path+'unet.png',
                             show_shapes=True)
        return 0
