import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.backend as tkb  # pylint: disable=E0401
import tensorflow.keras.layers as tkl  # pylint: disable=E0401

from data.generator import DataGenerator
from learning.image_generators.image_generator import ImageGenerator
from learning.utils.layers import one_hot_gen, sampling


class VAE(ImageGenerator):
    def __init__(
            self,
            training_generator: DataGenerator,
            validation_generator: DataGenerator,
            result_path: Path,
            model_path: Path,
        ):
        self.training_generator = training_generator
        self.validation_generator = validation_generator

        self.result_path = result_path
        self.model_path = model_path

        self.result_path.mkdir(exist_ok=True)
        self.model_path.parent.mkdir(exist_ok=True)


        self.image_shape = (64, 64, 1)
        self.one_hot = one_hot_gen(num_classes=4)

        self.nf = 64  # Number of filters in firts layer, scales accordingly
        self.latent_dimension = self.nf * 8


        # tf.logging.set_verbosity(tf.logging.ERROR)
        optimizer = tk.optimizers.Adam(1e-4)

        # Build and compile the discriminator
        self.model = self.build_model()
        self.model.compile(optimizer=optimizer, loss=self.vae_loss)
        self.model.summary()

    def build_model(self):
        def conv2d(x, filters, kernel_size=(4, 4), dropout_rate=0.0, batch_normalization=True, activation=True):
            x = tkl.Conv2D(filters, kernel_size=kernel_size, strides=(2, 2), padding='same')(x)
            if activation:
                x = tkl.LeakyReLU(alpha=0.2)(x)
            if dropout_rate:
                x = tkl.Dropout(dropout_rate)(x, training=True)
            if batch_normalization:
                x = tkl.BatchNormalization(momentum=0.8)(x)
            return x

        def deconv2d(x, x_skip, filters, kernel_size=(4, 4), dropout_rate=0.0, batch_normalization=True):
            x = tkl.UpSampling2D(size=(2, 2))(x)
            x = tkl.Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(x)
            if dropout_rate:
                x = tkl.Dropout(dropout_rate)(x, training=True)
            if batch_normalization:
                x = tkl.BatchNormalization(momentum=0.8)(x)
            x = tkl.Concatenate()([x, x_skip])
            return x


        image = tkl.Input(shape=self.image_shape)  # Think for (64, 64, 1)
        reward = tkl.Input(shape=(1,))
        action_type = tkl.Input(shape=(1,))

        r = tkl.Reshape((1, 1, 1))(reward)
        r = tkl.UpSampling2D(size=self.image_shape[:-1])(r)

        a = tkl.Lambda(self.one_hot)(action_type)
        a = tkl.Reshape((1, 1, 4))(a)
        a = tkl.UpSampling2D(size=self.image_shape[:-1])(a)

        h = tkl.Concatenate()([image, r, a])

        # Downsampling
        d1 = conv2d(h, self.nf, batch_normalization=False)
        d2 = conv2d(d1, self.nf*2)
        d3 = conv2d(d2, self.nf*4)
        d4 = conv2d(d3, self.nf*4)
        d5 = conv2d(d4, self.nf*8, dropout_rate=0.1)

        z_mean = conv2d(d5, self.latent_dimension, activation=False)
        z_logvar = conv2d(d5, self.latent_dimension, activation=False)

        assert z_mean.shape[1:] == (1, 1, self.latent_dimension)
        assert z_logvar.shape[1:] == (1, 1, self.latent_dimension)

        eps = tkl.Lambda(sampling)([z_mean, z_logvar])

        u1 = deconv2d(eps, d5, self.nf*8, dropout_rate=0.1)
        u2 = deconv2d(u1, d4, self.nf*8, dropout_rate=0.1)
        u3 = deconv2d(u2, d3, self.nf*4)
        u4 = deconv2d(u3, d2, self.nf*2)
        u5 = deconv2d(u4, d1, self.nf)

        u6 = tkl.UpSampling2D(size=2)(u5)
        result = tkl.Conv2D(self.image_shape[2], kernel_size=4, strides=1, padding='same', activation='tanh')(u6)
        assert result.shape[1:] == self.image_shape

        return tk.models.Model(
            inputs=[image, reward, action_type],
            outputs=[result, z_mean, z_logvar],
        )

    def vae_loss(self, y_true, y_pred):
        result = y_pred[0]
        z_mean = y_pred[1]
        z_logvar = y_pred[2]
        image_real = y_true[0]

        reconstruction_loss = tkb.sum(tk.losses.mse(result, image_real))
        k1_loss = 1 + z_logvar - tkb.square(z_mean) - tkb.exp(z_logvar)
        k1_loss = -0.5 * tkb.sum(k1_loss, axis=-1)
        return tkb.mean(reconstruction_loss + k1_loss)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            for batch_i, batch_data in enumerate(self.training_generator.load_batch(batch_size)):
                images_A, images_B, rewards, action_types = self.split_batch(batch_data)

                z_zero = np.zeros((batch_size, 1, 1, self.latent_dimension))
                loss = self.model.train_on_batch([images_B, rewards, action_types], [images_A, z_zero, z_zero])

                # Plot the progress
                elapsed_time = datetime.datetime.now() - start_time
                print(f'[Epoch {epoch}/{epochs}] [Batch {batch_i}/{self.training_generator.n_batches}] [loss: {loss[0]}] time: {elapsed_time}')

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, expand=False)

            # Save generator
            self.model.save(str(self.model_path))
