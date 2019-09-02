import numpy as np

import tensorflow.keras as tk  # pylint: disable=E0401


class Generator(tk.utils.Sequence):
    def __init__(self, x, y, batch_size=32, shuffle=True, save_to_dir=None, validation=False):
        assert len(x[0]) == len(y[0])

        self.x = x
        self.y = y

        self.batch_size = batch_size
        self.length = len(x[0])
        self.n_batches = self.length // batch_size + 1

        self.shuffle = shuffle
        self.save_to_dir = save_to_dir
        self.validation = validation

        self.flip_ud = None
        self.flip_lr = None
        self.height = None

        self.on_epoch_end()

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        idx_start, idx_end = self.batch_size * idx, self.batch_size * (idx + 1)
        batch_x = [e[idx_start:idx_end] for e in self.x]
        batch_y = [e[idx_start:idx_end] for e in self.y]

        if self.flip_ud or self.flip_lr or self.height:
            map(self.__data__augmentation(
                flip_ud=self.flip_ud,
                flip_lr=self.flip_lr,
                height=self.height
            ), zip(batch_x, batch_y))
        return batch_x, batch_y

    @classmethod
    def __data__augmentation(cls, flip_ud=None, flip_lr=None, height=None):
        def function(x, y):
            # Flip image
            if flip_ud and np.random.random() < flip_ud:
                map(np.flipud, x)

            if flip_lr and np.random.random() < flip_lr:
                map(np.fliplr, x)

            # Height augmentation
            if height:
                x[x != 0] += np.random.uniform(height[0], height[1])
        return function
