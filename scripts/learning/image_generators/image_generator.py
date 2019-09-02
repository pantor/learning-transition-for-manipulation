import matplotlib.pyplot as plt
import numpy as np


class ImageGenerator:
    @classmethod
    def split_batch(cls, batch):
        return batch[0][0], batch[0][1], batch[1][0][:, 0], batch[1][0][:, 1]

    def sample_images(self, epoch, batch_i, expand=True, latent=False):
        images_after, images_before, rewards, action_types = self.split_batch(self.validation_generator.load_data(batch_size=8, is_testing=True))

        if expand:
            rewards = np.expand_dims(np.expand_dims(np.expand_dims(rewards, axis=1), axis=1), axis=1).astype(np.float32)
            action_types = np.expand_dims(np.expand_dims(action_types, axis=1), axis=1)

        input_data = [images_before, rewards, action_types]

        if latent:
            latent_data = np.zeros((8, 1, 1, self.latent_dimension))
            # latent_data = np.random.normal(size=(32, 1, 1, self.latent_dimension)) i
            input_data += [latent_data]

        fake_after = self.generator.predict(input_data)

        gen_images = np.concatenate([images_before, fake_after, images_after])
        gen_images = (gen_images + 1) / 2  # Rescale images 0 - 1

        r, c = 3, 8
        title = f'epoch-{epoch}-{batch_i}'
        fig, axs = plt.subplots(r, c)
        fig.suptitle(title)

        if expand:
            rewards = rewards[:, 0, 0, 0]
            action_types = action_types[:, 0, 0]

        for ax, col in zip(axs[0], [f'r={r:0.2f}\na={a}' for r, a in zip(rewards, action_types)]):
            ax.set_title(col)

        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_images[cnt, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig(str(self.result_path / f'{title}.png'))
        plt.close()
