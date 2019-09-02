#%%
import os

from actions.indexer import GraspShiftIndexer
from data.dataset import Dataset
from data.generator import DataGenerator
from learning.image_generators.bicycle import Bicycle
from learning.image_generators.pix2pix import Pix2Pix
from learning.image_generators.vae import VAE

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#%%
data = Dataset([
        'cube-1',
        'cube-3',
        'cylinder-2',
        'cylinder-cube-1',
        'cylinder-cube-2',
        'cylinder-cube-mc-1',
        'shifting',
        'small-cubes-2',
    ],
    indexer=GraspShiftIndexer(gripper_classes=[0.05, 0.07, 0.086], shift_distance=0.03),
)


train_set, validation_set = data.load_data(
    # max_number=5000,
    force_images=False,
    scale_around_zero=True,
    size_cropped=(256, 256),
    size_output=(64, 64),
    suffixes=[
        ['ed-after', 'ed-v'],
    ],
)

#%%

image_generator_type = 'Bicycle'

if image_generator_type == 'Pix2Pix':
    image_generator = Pix2Pix(
        training_generator=DataGenerator(train_set, shuffle=True),
        validation_generator=DataGenerator(validation_set, shuffle=True),
        result_path=data.result_path,
        generator_path=data.model_path / 'predict-gen-6.h5',
        discriminator_path=data.model_path / 'predict-dis-6.h5',
    )

elif image_generator_type == 'VAE':
    image_generator = VAE(
        training_generator=DataGenerator(train_set, shuffle=True),
        validation_generator=DataGenerator(validation_set, shuffle=True),
        result_path=data.result_path,
        model_path=data.model_path / 'predict-vae-2.h5',
    )

elif image_generator_type == 'Bicycle':
    image_generator = Bicycle(
        training_generator=DataGenerator(train_set, shuffle=True),
        validation_generator=DataGenerator(validation_set, shuffle=True),
        result_path=data.result_path,
        generator_path=data.model_path / 'predict-bi-gen-8.h5',
        discriminator_path=data.model_path / 'predict-bi-dis-8.h5',
        encoder_path=data.model_path / 'predict-bi-enc-8.h5',
    )

image_generator.train(epochs=350, batch_size=32, sample_interval=200)
