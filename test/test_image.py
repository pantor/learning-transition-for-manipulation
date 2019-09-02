from pathlib import Path
import unittest

import imageio
import numpy as np

import picking.path_fix_ros  # pylint: disable=W0611

from actions.action import Action, RobotPose
from actions.converter import Converter
from actions.indexer import GraspIndexer
from orthographical import OrthographicImage
from cfrankr import Affine
from config import Config
from inference.inference import Inference
from picking.image import crop, draw_around_box, draw_pose, get_area_of_interest, sinkhorn
from picking.param import SelectionMethod


class UnitTestHelper(unittest.TestCase):
    file_path = Path(__file__).parent / 'documents'
    image_name = 'image-2018-11-07-18-31-09-249-ed-v.png'

    @classmethod
    def read_image(cls, path: Path):
        return imageio.imread(str(path), 'PNG-PIL').astype(np.uint16) * 255

    def assertAlmostEqualIter(self, a, b):
        map(lambda c: self.assertAlmostEqual(c[0], c[1]), zip(a, b))

    def test_crop(self):
        small_image = np.array([
            [1, 1, 1, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 1, 1, 1]
        ])
        correct_small_image_cropped = np.array([
            [2, 2],
            [2, 2]
        ])
        small_image_cropped = crop(small_image, (2, 2))
        self.assertTupleEqual(small_image_cropped.shape, (2, 2))
        self.assertAlmostEqualIter(small_image_cropped.flatten(), correct_small_image_cropped.flatten())

        large_image = self.read_image(self.file_path / self.image_name)
        large_image_cropped = crop(large_image, (317, 563), (1, 1))
        self.assertTupleEqual(large_image_cropped.shape, (317, 563))

    def test_image_transformation(self):
        image = OrthographicImage(self.read_image(self.file_path / self.image_name), 2000.0, 0.22, 0.4, '', Config.default_image_pose)

        pose = RobotPose(affine=Affine(0.02, 0.0, 0.0))
        area_image = get_area_of_interest(image, pose, border_color=(0))
        imageio.imwrite(str(self.file_path / 'gen-x20-b.png'), area_image.mat)

        pose = RobotPose(affine=Affine(0.03, 0.03, 0.0))
        area_image = get_area_of_interest(image, pose, border_color=(0))
        imageio.imwrite(str(self.file_path / 'gen-x30-y30-b.png'), area_image.mat)

        pose = RobotPose(affine=Affine(0.01, 0.04, 0.0, 0.4))
        area_image = get_area_of_interest(image, pose, border_color=(0))
        imageio.imwrite(str(self.file_path / 'gen-x10-y40-a04-b.png'), area_image.mat)

        image = image.translate([0.0, 0.0, 0.05])
        image = image.rotate_x(-0.3, [0.0, 0.25])
        imageio.imwrite(str(self.file_path / 'gen-rot0_3.png'), image.mat)

    def test_filter(self):
        data = np.array([0.0, 0.3, 0.0, 0.9, 0.5, 0.4, 0.1, 0.2, 0.1, 0.0])

        def get_element(method: SelectionMethod):
            return data[Inference.get_filter(method)(data)]

        self.assertEqual(get_element(SelectionMethod.Max), 0.9)
        self.assertEqual(get_element(SelectionMethod.Uncertain), 0.5)
        self.assertGreater(get_element(SelectionMethod.Top5), 0.1)
        self.assertGreater(get_element(SelectionMethod.NotZero), 0.0)

    def test_drawing(self):
        image = OrthographicImage(self.read_image(self.file_path / self.image_name), 2000.0, 0.22, 0.4, '', Config.default_image_pose)

        self.assertEqual(image.mat.dtype, np.uint16)

        draw_around_box(image, box=Config.box)
        imageio.imwrite(str(self.file_path / 'gen-draw-around-bin-unit.png'), image.mat)

    def test_sinkhorn(self):
        def load_image(filename):
            return self.read_image(self.file_path / filename).astype(np.float32) / 255.0

        orig = load_image('image-blob.png')

        r4 = load_image('image-blob-r2.png') + load_image('image-blob-r3.png')

        comparisons = [
            load_image('image-blob-r1.png'),
            load_image('image-blob-r2.png'),
            load_image('image-blob-r3.png'),
            r4,
        ]

        for comp in comparisons:
            print(sinkhorn(orig, comp, lambda_new=5e3))



if __name__ == '__main__':
    unittest.main(verbosity=3)
