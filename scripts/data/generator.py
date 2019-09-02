import numpy as np


class DataGenerator:
    """ A generator class for GANs"""

    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.length = len(dataset[1][0])  # Length of validation set
        self.n_batches = -1

        # self.augment = self.__data__augmentation(flip_ud=None, flip_lr=None, height=None)

        if shuffle:
            permutation = np.random.permutation(self.length)

            for i in range(len(self.dataset)):
                for j in range(len(self.dataset[i])):
                    self.dataset[i][j] = self.dataset[i][j][permutation]

    # @classmethod
    # def __data__augmentation(cls, flip_ud=None, flip_lr=None, height=None):
    #     def function(x, y):
    #         # Flip image
    #         if flip_ud and np.random.random() < flip_ud:
    #             map(np.flipud, x)

    #         if flip_lr and np.random.random() < flip_lr:
    #             map(np.fliplr, x)

    #         # Height augmentation
    #         if height:
    #             x[x != 0] += np.random.uniform(height[0], height[1])
    #     return function

    # def data_for_index(self, idx):
    #     x = [self.dataset[0][i][idx] for i in len(self.dataset[0])]
    #     y = [self.dataset[1][i][idx] for i in len(self.dataset[0])]
    #     self.augment(x, y)
    #     return x, y

    def data_for_indices(self, indices, is_testing):
        result = []
        for i in range(len(self.dataset)):  # features, labels
            mid_result = []
            for j in range(len(self.dataset[i])):
                end_result = []
                for idx in indices:
                    element = self.dataset[i][j][idx]

                    if not is_testing and len(element.shape) > 2 and np.random.random() < 0.5:
                        # element = transforms[idx](element)
                        pass

                    end_result.append(element)
                mid_result.append(np.array(end_result))
            result.append(mid_result)
        return tuple(result)

    def load_data(self, batch_size=1, is_testing=False):
        batch_images_idx = np.random.randint(self.length, size=batch_size)
        return self.data_for_indices(batch_images_idx, is_testing)

    def load_batch(self, batch_size=1, is_testing=False):
        self.n_batches = int(self.length / batch_size)

        for i in range(self.n_batches):
            yield self.data_for_indices(range(i*batch_size, (i+1)*batch_size), is_testing)
