#%%
import os

import tensorflow as tf
import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.layers as tkl  # pylint: disable=E0401

from data.dataset import Dataset
from learning.utils.layers import conv_block_gen
from learning.utils.trainer import Trainer

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#%%
data = Dataset([
    'shifting',
])

train_set, validation_set = data.load_data(
    force_images=True,
    suffixes=[
        ['ed-v'],
    ],
)


def normalize_labels(data_set):
    data_set[1][0][:, 0] = 0.5 * data_set[1][0][:, 0] + 0.5

normalize_labels(train_set)
normalize_labels(validation_set)

# print(abs(train_set['labels'][:, 0] - 0.5).mean())
# print(abs(test_set['labels'][:, 0] - 0.5).mean())


#%%
def define_model(number_actions: float):
    image = tk.Input(shape=(None, None, 1), name='image')

    conv_block = conv_block_gen(l2_reg=0.02, dropout_rate=0.5)

    x = conv_block(image, 32, kernel_size=(5, 5), strides=(2, 2))

    x = conv_block(x, 32)
    x = conv_block(x, 32)

    x = conv_block(x, 48)
    x = conv_block(x, 48)
    x = conv_block(x, 48)

    x = conv_block(x, 64)
    x = conv_block(x, 96, kernel_size=(2, 2))

    prob = tkl.Conv2D(number_actions, kernel_size=(1, 1), activation='sigmoid', name='prob')(x)
    assert (str(prob.shape[1]) == '?') or ((prob.shape[1] == 1) and (prob.shape[2] == 1))

    prob_training = tkl.Reshape((number_actions,))(prob)
    return tk.Model(inputs=image, outputs=prob_training)


#%%

trainer = Trainer(train_set, validation_set)

tf.reset_default_graph()
tk.backend.clear_session()

model = define_model(number_actions=1)
model.summary()

history, best_metrics = trainer.train(
    model,
    load_model=False,
    train_model=True,
    path=data.model_path / 'model-1.h5',
    loss_name='mean_square_error',
    lr=1e-4,
    batch_size=32,
    learning_duration=2,
    verbose=0,
)
