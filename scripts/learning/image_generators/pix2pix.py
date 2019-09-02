import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.layers as tkl  # pylint: disable=E0401

from data.generator import DataGenerator
from learning.image_generators.image_generator import ImageGenerator
from learning.utils.layers import one_hot_gen


class Pix2Pix(ImageGenerator):
    def __init__(
            self,
            training_generator: DataGenerator,
            validation_generator: DataGenerator,
            result_path: Path,
            generator_path: Path,
            discriminator_path: Path
        ):
        self.training_generator = training_generator
        self.validation_generator = validation_generator

        self.result_path = result_path
        self.generator_path = generator_path
        self.discriminator_path = discriminator_path

        self.result_path.mkdir(exist_ok=True)
        self.generator_path.parent.mkdir(exist_ok=True)
        self.discriminator_path.parent.mkdir(exist_ok=True)


        self.image_shape = (64, 64, 1)
        self.one_hot = one_hot_gen(num_classes=4)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.image_shape[0] / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64


        tf.logging.set_verbosity(tf.logging.ERROR)
        optimizer = tk.optimizers.Adam(2e-4, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.discriminator.summary()

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        image_A = tkl.Input(shape=self.image_shape)
        image_B = tkl.Input(shape=self.image_shape)
        reward = tkl.Input(shape=(1,))
        action_type = tkl.Input(shape=(1,), dtype='uint8')

        # By conditioning on B generate a fake version of A
        fake_A = self.generator([image_B, reward, action_type])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, image_B, reward, action_type])

        self.combined = tk.models.Model(inputs=[image_A, image_B, reward, action_type], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=optimizer)
        self.combined.summary()

    def build_generator(self):
        def conv2d(x, filters, kernel_size=(4, 4), dropout_rate=0.0, batch_normalization=True):
            x = tkl.Conv2D(filters, kernel_size=kernel_size, strides=(2, 2), padding='same')(x)
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

        # Image input
        image = tkl.Input(shape=self.image_shape)
        reward = tkl.Input(shape=(1,))
        action_type = tkl.Input(shape=(1,))

        r = tkl.Reshape((1, 1, 1))(reward)
        r = tkl.UpSampling2D(size=self.image_shape[:-1])(r)

        a = tkl.Lambda(self.one_hot)(action_type)
        a = tkl.Reshape((1, 1, 4))(a)
        a = tkl.UpSampling2D(size=self.image_shape[:-1])(a)

        h = tkl.Concatenate()([image, r, a])

        # Downsampling
        d1 = conv2d(h, self.gf, batch_normalization=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8, dropout_rate=0.2)
        d6 = conv2d(d5, self.gf*8, dropout_rate=0.4)

        if self.image_shape[0] == 128:
            d7 = conv2d(d6, self.gf*8, dropout_rate=0.4)

            # Upsampling
            u1 = deconv2d(d7, d6, self.gf*8, dropout_rate=0.4)
            u2 = deconv2d(u1, d5, self.gf*8, dropout_rate=0.4)
        else:
            u2 = deconv2d(d6, d5, self.gf*8, dropout_rate=0.4)

        u3 = deconv2d(u2, d4, self.gf*8, dropout_rate=0.2)
        u4 = deconv2d(u3, d3, self.gf*4, dropout_rate=0.2)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = tkl.UpSampling2D(size=2)(u6)
        result = tkl.Conv2D(self.image_shape[2], kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
        assert result.shape[1:] == self.image_shape

        return tk.models.Model([image, reward, action_type], result, name='gen')

    def build_discriminator(self):
        def d_layer(x, filters, kernel_size=(4, 4), batch_normalization=True):
            x = tkl.Conv2D(filters, kernel_size=kernel_size, strides=(2, 2), padding='same')(x)
            x = tkl.LeakyReLU(alpha=0.2)(x)
            if batch_normalization:
                x = tkl.BatchNormalization(momentum=0.8)(x)
            return x

        image_A = tkl.Input(shape=self.image_shape)
        image_B = tkl.Input(shape=self.image_shape)
        reward = tkl.Input(shape=(1,))
        action_type = tkl.Input(shape=(1,), dtype='uint8')

        # Concatenate image and conditioning image by channels to produce input
        combined_images = tkl.Concatenate(axis=-1)([image_A, image_B])

        h = d_layer(combined_images, self.df, batch_normalization=False)
        h = d_layer(h, self.df*2)
        h = d_layer(h, self.df*4)
        h = d_layer(h, self.df*8)

        r = tkl.Reshape((1, 1, 1))(reward)
        r = tkl.UpSampling2D(size=self.disc_patch[:-1])(r)

        a = tkl.Lambda(self.one_hot)(action_type)
        a = tkl.Reshape((1, 1, 4))(a)
        a = tkl.UpSampling2D(size=self.disc_patch[:-1])(a)

        h = tkl.Concatenate()([h, r, a])

        validity = tkl.Conv2D(1, kernel_size=4, strides=1, padding='same')(h)  # Validity shape for PatchGAN
        assert validity.shape[1:] == self.disc_patch

        return tk.models.Model([image_A, image_B, reward, action_type], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        d_loss = [0.0, 0.0]

        for epoch in range(epochs):
            for batch_i, batch_data in enumerate(self.training_generator.load_batch(batch_size)):
                images_A, images_B, rewards, action_types = self.split_batch(batch_data)

                fake_A = self.generator.predict([images_B, rewards, action_types])

                if d_loss[1] < 0.75:
                    d_loss_real = self.discriminator.train_on_batch([images_A, images_B, rewards, action_types], valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, images_B, rewards, action_types], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                else:
                    d_loss = [0.0, 0.0]

                g_loss = self.combined.train_on_batch([images_A, images_B, rewards, action_types], [valid, images_A])

                elapsed_time = datetime.datetime.now() - start_time
                print(f'[Epoch {epoch}/{epochs}] [Batch {batch_i}/{self.training_generator.n_batches}] [D loss: {d_loss[0]}, acc: {100*d_loss[1]}] [G loss: {g_loss[0]}] time: {elapsed_time}')

                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, expand=False)

            # Save generator
            self.generator.save(str(self.generator_path))
            self.discriminator.save(str(self.discriminator_path))
