import datetime
from pathlib import Path

import numpy as np

import tensorflow as tf
import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.backend as tkb  # pylint: disable=E0401
import tensorflow.keras.layers as tkl  # pylint: disable=E0401

from data.generator import DataGenerator
from learning.image_generators.image_generator import ImageGenerator
from learning.utils.layers import conv_block_gen, one_hot_gen, sampling


class Bicycle(ImageGenerator):
    def __init__(
            self,
            training_generator: DataGenerator,
            validation_generator: DataGenerator,
            result_path: Path,
            generator_path: Path,
            discriminator_path: Path,
            encoder_path: Path,
        ):
        self.training_generator = training_generator
        self.validation_generator = validation_generator

        self.result_path = result_path
        self.generator_path = generator_path
        self.discriminator_path = discriminator_path
        self.encoder_path = encoder_path

        self.result_path.mkdir(exist_ok=True)
        self.generator_path.parent.mkdir(exist_ok=True)
        self.discriminator_path.parent.mkdir(exist_ok=True)
        self.encoder_path.parent.mkdir(exist_ok=True)

        self.image_shape = (64, 64, 1)
        self.reward_shape = (1, 1, 1)
        self.action_type_shape = (1, 1,)
        self.one_hot = one_hot_gen(num_classes=4)

        self.latent_dimension = 8

        # Calculate output shape of D (PatchGAN)
        patch = int(self.image_shape[0] / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64


        tf.logging.set_verbosity(tf.logging.ERROR)
        d_optimizer = tk.optimizers.Adam(5e-5, 0.5)
        g1_optimizer = tk.optimizers.Adam(2e-4)
        g2_optimizer = tk.optimizers.Adam(2e-4)


        load_models = False
        if load_models:
            self.discriminator = tk.models.load_model(str(self.discriminator_path), compile=False)
            self.generator = tk.models.load_model(str(self.generator_path), compile=False)
            self.encoder = tk.models.load_model(str(self.encoder_path), compile=False)

        else:
            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()
            self.discriminator.summary()

            # Build the generator and encoder
            self.generator = self.build_generator()
            self.encoder = self.build_encoder()

        self.discriminator.compile(loss='mse', optimizer=d_optimizer, metrics=['accuracy'])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Inspired from https://github.com/gitlimlab/BicycleGAN-Tensorflow/blob/master/model.py
        image_after = tkl.Input(shape=self.image_shape, name='image_after')
        image_before = tkl.Input(shape=self.image_shape, name='image_before')
        reward = tkl.Input(shape=self.reward_shape, name='reward')
        action_type = tkl.Input(shape=self.action_type_shape, dtype='uint8', name='action_type')
        z = tkl.Input(shape=(1, 1, self.latent_dimension), name='z')

        # Conditional VAE-GAN training step: B -> z -> B'
        z_encoded, z_mean_encoded, z_logvar_encoded = self.encoder([image_after, reward, action_type])
        fake_after_encoded = self.generator([image_before, reward, action_type, z_encoded])

        # Conditional Latent Regressor-GAN: z -> B' -> z'
        fake_after_noise = self.generator([image_before, reward, action_type, z])
        z_recon, _, _ = self.encoder([fake_after_noise, reward, action_type])

        gan_valid = self.discriminator([fake_after_noise, image_before, reward, action_type])
        vae_gan_valid = self.discriminator([fake_after_encoded, image_before, reward, action_type])

        lambda_k1 = 0.1
        k1_loss = -0.5 * tkb.mean(1 + 2 * z_logvar_encoded - z_mean_encoded**2 - tkb.exp(2 * z_logvar_encoded))

        self.vae_gan = tk.models.Model(
            inputs=[image_after, image_before, reward, action_type, z],
            outputs=[vae_gan_valid, fake_after_encoded]
        )
        self.vae_gan.add_loss(lambda_k1 * k1_loss)
        self.vae_gan.compile(loss=['mse', 'mae'], loss_weights=[1.0, 60.0], optimizer=g1_optimizer)
        self.vae_gan.summary()

        self.gan = tk.models.Model(
            inputs=[image_after, image_before, reward, action_type, z],
            outputs=[gan_valid, fake_after_noise, z_recon]
        )
        self.gan.compile(loss=['mse', 'mae', 'mae'], loss_weights=[1.0, 0.6, 0.1], optimizer=g2_optimizer)
        self.gan.summary()

    def conv_block(self, x, filters, kernel_size=(4, 4), padding='same', dropout_rate=0.0, batch_normalization=True, activation='lrelu'):
        x = tkl.Conv2D(filters, kernel_size=kernel_size, strides=(2, 2), padding=padding)(x)
        if activation == 'lrelu':
            x = tkl.LeakyReLU(alpha=0.2)(x)
        if dropout_rate:
            x = tkl.Dropout(dropout_rate)(x)
        if batch_normalization:
            x = tkl.BatchNormalization(momentum=0.8)(x)
        return x

    def deconv_block(self, x, x_skip, filters, kernel_size=(4, 4), padding='same', dropout_rate=0.0, batch_normalization=True, activation='lrelu'):
        x = tkl.UpSampling2D(size=(2, 2))(x)
        x = tkl.Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding=padding)(x)
        if activation == 'lrelu':
            x = tkl.LeakyReLU(alpha=0.2)(x)
        if dropout_rate:
            x = tkl.Dropout(dropout_rate)(x)
        if batch_normalization:
            x = tkl.BatchNormalization(momentum=0.8)(x)
        x = tkl.Concatenate()([x, x_skip])
        return x

    def build_generator(self):
        image = tkl.Input(shape=self.image_shape)
        reward = tkl.Input(shape=self.reward_shape)
        action_type = tkl.Input(shape=self.action_type_shape, dtype='uint8')
        eps = tkl.Input(shape=(1, 1, self.latent_dimension))

        a = tkl.Lambda(self.one_hot)(action_type)

        r = tkl.UpSampling2D(size=self.image_shape[:-1])(reward)
        a = tkl.UpSampling2D(size=self.image_shape[:-1])(a)
        z = tkl.UpSampling2D(size=self.image_shape[:-1])(eps)

        h = tkl.Concatenate()([image, r, a, z])

        # Downsampling
        d1 = self.conv_block(h, self.gf*2, batch_normalization=False)
        d2 = self.conv_block(d1, self.gf*2)
        d3 = self.conv_block(d2, self.gf*4)
        d4 = self.conv_block(d3, self.gf*8)
        d5 = self.conv_block(d4, self.gf*8)
        d6 = self.conv_block(d5, self.gf*8)

        u2 = self.deconv_block(d6, d5, self.gf*8)
        u3 = self.deconv_block(u2, d4, self.gf*8)
        u4 = self.deconv_block(u3, d3, self.gf*8)
        u5 = self.deconv_block(u4, d2, self.gf*4)
        u6 = self.deconv_block(u5, d1, self.gf*2)

        u7 = tkl.UpSampling2D(size=2)(u6)
        result = tkl.Conv2D(self.image_shape[2], kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
        assert result.shape[1:] == self.image_shape

        return tk.models.Model([image, reward, action_type, eps], result)

    def build_discriminator(self):
        image_after = tkl.Input(shape=self.image_shape)
        image_before = tkl.Input(shape=self.image_shape)
        reward = tkl.Input(shape=self.reward_shape)
        action_type = tkl.Input(shape=self.action_type_shape, dtype='uint8')

        combined_images = tkl.Concatenate(axis=-1)([image_after, image_before])

        h = self.conv_block(combined_images, self.df, batch_normalization=False)
        h = self.conv_block(h, self.df, dropout_rate=0.2)
        h = self.conv_block(h, self.df*2, dropout_rate=0.3)
        h = self.conv_block(h, self.df*2, dropout_rate=0.4)

        a = tkl.Lambda(self.one_hot)(action_type)

        r = tkl.UpSampling2D(size=self.disc_patch[:-1])(reward)
        a = tkl.UpSampling2D(size=self.disc_patch[:-1])(a)

        h = tkl.Concatenate()([h, r, a])

        validity = tkl.Conv2D(1, kernel_size=4, strides=1, padding='same')(h)  # Validity shape for PatchGAN
        assert validity.shape[1:] == self.disc_patch

        return tk.models.Model([image_after, image_before, reward, action_type], validity)

    def build_encoder(self):
        image = tkl.Input(shape=self.image_shape)
        reward = tkl.Input(shape=self.reward_shape)
        action_type = tkl.Input(shape=self.action_type_shape, dtype='uint8')

        h = self.conv_block(image, self.df, batch_normalization=False)
        h = self.conv_block(h, self.df, dropout_rate=0.1)
        h = self.conv_block(h, self.df*2, dropout_rate=0.2)
        h = self.conv_block(h, self.df*4, dropout_rate=0.3)

        a = tkl.Lambda(self.one_hot)(action_type)

        r = tkl.UpSampling2D(size=self.disc_patch[:-1])(reward)
        a = tkl.UpSampling2D(size=self.disc_patch[:-1])(a)

        h = tkl.Concatenate()([h, r, a])
        h = self.conv_block(h, self.df*2, padding='valid')

        z_mean = tkl.Conv2D(self.latent_dimension, kernel_size=1, strides=1, padding='same')(h)
        z_logvar = tkl.Conv2D(self.latent_dimension, kernel_size=1, strides=1, padding='same')(h)

        assert z_mean.shape[1:] == (1, 1, self.latent_dimension)
        assert z_logvar.shape[1:] == (1, 1, self.latent_dimension)

        eps = tkl.Lambda(sampling)([z_mean, z_logvar])

        return tk.models.Model(
            inputs=[image, reward, action_type],
            outputs=[eps, z_mean, z_logvar],
        )

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.zeros((batch_size,) + self.disc_patch) + 0.1  # Label smoothing
        fake = np.ones((batch_size,) + self.disc_patch) - 0.1  # Label smoothing

        d_loss = [0.0, 0.0]

        for epoch in range(epochs):
            for batch_i, batch_data in enumerate(self.training_generator.load_batch(batch_size)):
                images_after, images_before, rewards, action_types = self.split_batch(batch_data)
                rewards = np.expand_dims(np.expand_dims(np.expand_dims(rewards, axis=1), axis=1), axis=1).astype(np.float32)
                action_types = np.expand_dims(np.expand_dims(action_types, axis=1), axis=1)

                z_noise = np.random.normal(size=(batch_size, 1, 1, self.latent_dimension))
                z_zero = np.zeros(shape=(batch_size, 1, 1, self.latent_dimension))
                z_encoded, _, _ = self.encoder.predict([images_after, rewards, action_types])

                fake_vae_gan = self.generator.predict([images_before, rewards, action_types, z_encoded])
                fake_gan = self.generator.predict([images_before, rewards, action_types, z_noise])

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([images_after, images_before, rewards, action_types], valid)
                d_loss_fake_vae_gan = self.discriminator.train_on_batch([fake_vae_gan, images_before, rewards, action_types], fake)

                # Cycle 1
                vae_gan_loss = self.vae_gan.train_on_batch([images_after, images_before, rewards, action_types, z_zero], [valid, images_after])
                gan_loss = self.gan.train_on_batch([images_after, images_before, rewards, action_types, z_noise], [valid, images_after, z_noise])

                d_loss_real = self.discriminator.train_on_batch([images_after, images_before, rewards, action_types], valid)
                d_loss_fake_gan = self.discriminator.train_on_batch([fake_gan, images_before, rewards, action_types], fake)
                d_loss = 0.5 * np.array(d_loss_real) + 0.25 * np.array(d_loss_fake_vae_gan) + 0.25 * np.array(d_loss_fake_gan)

                # Cycle 2
                vae_gan_loss = self.vae_gan.train_on_batch([images_after, images_before, rewards, action_types, z_zero], [valid, images_after])
                gan_loss = self.gan.train_on_batch([images_after, images_before, rewards, action_types, z_noise], [valid, images_after, z_noise])
                g_loss = 0.5 * np.array(vae_gan_loss) + 0.5 * np.array(gan_loss[:3])
                e_loss = gan_loss[-1]

                # Plot the progress
                elapsed_time = datetime.datetime.now() - start_time
                print(f'[Epoch {epoch}/{epochs}] [Batch {batch_i}/{self.training_generator.n_batches}] [D loss: {d_loss[0]:0.3f}, acc: {d_loss[1]:0.3f}] [G loss: {g_loss[0]:0.3f}] [E loss: {e_loss:0.3f}] time: {elapsed_time}')

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, latent=True)

            # Save generator
            self.generator.save(str(self.generator_path))
            self.discriminator.save(str(self.discriminator_path))
            self.encoder.save(str(self.encoder_path))
