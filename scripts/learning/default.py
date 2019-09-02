#%%
import os

import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.layers as tkl  # pylint: disable=E0401

from actions.indexer import GraspIndexer
from config import Config
from data.dataset import Dataset
from learning.utils.layers import conv_block_gen
from learning.utils.trainer import Trainer


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#%%
data = Dataset([
    'small-cubes-2',
    'cylinder-cube-mc-1',
    'cylinder-cube-1',
    # 'cylinder-1',
    'cylinder-2',
    'cube-1',
    'cube-3',
],
    indexer=GraspIndexer(gripper_classes=Config.gripper_classes)
)

train_set, validation_set = data.load_data(
    force_images=False,
    suffixes=[
        ['ed-v'],
    ],
)

#%%
def define_model(number_actions: float):
    image = tk.Input(shape=(None, None, 1), name='image')

    conv_block = conv_block_gen(l2_reg=0.02, dropout_rate=0.5)

    x = conv_block(image, 32, kernel_size=(5, 5))
    x = conv_block(x, 32)

    x = conv_block(x, 48)
    x = conv_block(x, 48)
    x = conv_block(x, 48, strides=(2, 2))

    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x = conv_block(x, 64)

    x = conv_block(x, 96)
    x = conv_block(x, 96, kernel_size=(2, 2))

    x = conv_block(x, 64, kernel_size=(1, 1))

    print(x.shape)

    reward = tkl.Conv2D(number_actions, kernel_size=(1, 1), activation='sigmoid', name='prob')(x)
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
    path=data.model_path / 'model-2.h5',
    lr=1e-4,
    batch_size=32,
    learning_duration=1.3,
    verbose=1,
)
