from pathlib import Path
import unittest

import imageio
import numpy as np

import picking.path_fix_ros  # pylint: disable=W0611

from actions.converter import Converter
from config import Config
from data.dataset import Dataset
from data.loader import Loader
from picking.image import draw_around_box, image_difference
from picking.param import SelectionMethod


class FunctionTestHelper(unittest.TestCase):
    file_path = Path(__file__).parent / 'documents'

    def test_dataset(self):
        ds = Dataset('all-1')
        ds.load_data(
            force_images=True,
            suffixes=[['ed-v']],
        )

    def test_loader(self):
        for suffix in ['ed-v', 'ed-side_b-0_400']:
            action, image = Loader.get_action('cylinder-cube-1', '2019-06-25-15-49-13-551', suffix)

            draw_around_box(image, box=Config.box)
            imageio.imwrite(str(self.file_path / f'gen-draw-around-box-{suffix}.png'), image.mat)

            self.assertEqual(image.mat.dtype, np.uint16)
            self.assertEqual(image.pixel_size, 2000.0)
            self.assertEqual(action.method, SelectionMethod.Prob)

    def test_inside_box(self):
        action = Loader.get_action('cylinder-cube-1', '2019-06-25-14-59-51-451')

        conv = Converter(box=Config.box)
        self.assertFalse(conv.is_pose_inside_box(action.pose))

    def test_difference(self):
        _, image1, image2 = Loader.get_action('cube-1', '2018-10-22-23-42-52-096', ['ed-v', 'ed-after'])

        diff = image_difference(image1, image2)
        imageio.imwrite(str(self.file_path / f'gen-diff.png'), diff.mat)


if __name__ == '__main__':
    unittest.main(verbosity=3)
