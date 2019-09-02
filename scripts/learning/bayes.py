#%%
import os

import tensorflow_probability as tfp  # pylint: disable=E0401
import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.layers as tkl  # pylint: disable=E0401

from data.dataset import Dataset
from learning.utils.layers import bayes_conv_block_gen
from learning.utils.trainer import Trainer

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#%%
data = Dataset([
    'cylinder-cube-1',
    'cylinder-1',
    'cylinder-2',
    'cube-1',
])

train_set, validation_set = data.load_data(
    force_images=False,
    suffixes=[['ed-v']],
    # suffixes=['ed-side_b-0.400'],
)

#%%
def define_model(number_actions):
    image = tk.Input(shape=(None, None, 1), name='image')

    bayes_conv_block = bayes_conv_block_gen()

    x = bayes_conv_block(image, 32, kernel_size=(5, 5), strides=(2, 2))
    x = bayes_conv_block(x, 32)

    x = bayes_conv_block(x, 48)
    x = bayes_conv_block(x, 48)

    x = bayes_conv_block(x, 64)
    x = bayes_conv_block(x, 64)

    x = bayes_conv_block(x, 96)
    x = bayes_conv_block(x, 128, kernel_size=(2, 2))

    x = bayes_conv_block(x, 128, kernel_size=(1, 1))

    reward = tfp.layers.Convolution2DFlipout(number_actions, kernel_size=(1, 1), activation='sigmoid', name='prob')(x)

    assert (str(reward.shape[1]) == '?') or ((reward.shape[1] == 1) and (reward.shape[2] == 1))

    reward_training = tkl.Reshape((number_actions,))(reward)
    return tk.Model(inputs=image, outputs=reward_training)

#%%
trainer = Trainer(train_set, validation_set)

tk.backend.clear_session()

model = define_model(number_actions=len(data.indexer))
model.summary()

history, best_metrics = trainer.train(
    model,
    load_model=False,
    train_model=True,
    path=data.model_path / 'model-8-bayes.h5',
    lr=5e-4,
    verbose=1,
)
