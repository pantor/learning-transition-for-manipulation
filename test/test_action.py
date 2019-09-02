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
from picking.episode import Episode, EpisodeHistory
from picking.image import draw_around_box, draw_pose
from picking.param import Bin


class UnitTestHelper(unittest.TestCase):
    file_path = Path(__file__).parent / 'documents'
    image_name = 'image-2018-11-07-18-31-09-249-ed-v.png'

    box = {'center': [-0.001, -0.0065, 0.372], 'size': [0.172, 0.281, 0.068]}

    @classmethod
    def read_image(cls, path: Path):
        return imageio.imread(str(path), 'PNG-PIL').astype(np.uint16) * 255

    def assertAlmostEqualIter(self, a, b):
        map(lambda c: self.assertAlmostEqual(c[0], c[1]), zip(a, b))

    def test_affine(self):
        a = Affine(1.2, 0.4, 0.6, -0.8, 0.0, 0.4)
        self.assertAlmostEqual(a.x, 1.2)
        self.assertAlmostEqual(a.y, 0.4)
        self.assertAlmostEqual(a.z, 0.6)
        self.assertAlmostEqual(a.a, -0.8)
        self.assertAlmostEqual(a.b, 0.0)
        self.assertAlmostEqual(a.c, 0.4)

        a.x = 1.5
        a.a = 0.3
        a.c = 0.11
        self.assertAlmostEqual(a.x, 1.5)
        self.assertAlmostEqual(a.y, 0.4)
        self.assertAlmostEqual(a.z, 0.6)
        self.assertAlmostEqual(a.a, 0.3)
        self.assertAlmostEqual(a.b, 0.0)
        self.assertAlmostEqual(a.c, 0.11)

        b = Affine(0.1, 0.2, c=0.3)
        self.assertAlmostEqual(b.x, 0.1)
        self.assertAlmostEqual(b.y, 0.2)
        self.assertAlmostEqual(b.z, 0.0)
        self.assertAlmostEqual(b.a, 0.0)
        self.assertAlmostEqual(b.b, 0.0)
        self.assertAlmostEqual(b.c, 0.3)

    def test_action(self):
        p = RobotPose()
        self.assertEqual(p.d, 0.0)

        a = Action()
        a.index = 1
        self.assertEqual(a.index, 1)

        a_data = a.__dict__
        self.assertEqual(a_data['index'], 1)

    def test_indexer(self):
        indexer = GraspIndexer(gripper_classes=[0.04, 0.06, 0.08])

        def define_action(d=0.0, index=0):
            a = Action()
            a.pose.d = d
            a.index = index
            return a

        self.assertEqual(indexer.from_action(define_action(d=0.04)), 0)
        self.assertEqual(indexer.from_action(define_action(d=0.059)), 1)
        self.assertEqual(indexer.from_action(define_action(d=0.06)), 1)
        self.assertEqual(indexer.from_action(define_action(d=0.07)), 2)
        self.assertEqual(indexer.from_action(define_action(d=0.08)), 2)
        self.assertEqual(indexer.from_action(define_action(d=0.1)), 2)

        a = define_action(index=2)
        indexer.to_action(a)
        self.assertEqual(indexer.from_action(a), 2)

    def test_safe_lower_x(self):
        lower_x_for_y = {
            0.15: 0.33,
            0.1: 0.36,
            0.05: 0.37,
            0.0: 0.36,
            -0.05: 0.32,
            -0.10: 0.30,
            -0.15: 0.25,
        }

        def find_nearest(obj, k):
            return list(obj.values())[np.argmin(np.abs(np.array(list(obj.keys())) - k))]

        print(find_nearest(lower_x_for_y, 0.0249))

    def test_grasp_conversion(self):
        conv = Converter(grasp_z_offset=0.0, box=self.box)

        image = OrthographicImage(self.read_image(self.file_path / self.image_name), 2000.0, 0.22, 0.4, '', Config.default_image_pose)

        action = Action()
        action.type = 'grasp'
        action.pose.x = -0.06
        action.pose.y = 0.099
        action.pose.a = 0.523
        action.pose.d = 0.078
        action.index = 1

        draw_pose(image, action.pose, convert_to_rgb=True)
        draw_around_box(image, box=self.box, draw_lines=True)
        imageio.imwrite(str(self.file_path / 'gen-grasp-draw-pose.png'), image.mat)

        self.assertTrue(conv.grasp_check_safety(action, [image]))

        conv.calculate_pose(action, [image])
        self.assertLess(action.pose.z, 0.0)

    def test_shift_conversion(self):
        conv = Converter(shift_z_offset=0.0, box=self.box)

        image = OrthographicImage(self.read_image(self.file_path / self.image_name), 2000.0, 0.22, 0.4, '', Config.default_image_pose)

        action = Action()
        action.type = 'shift'
        action.pose = RobotPose()
        action.pose.x = -0.02
        action.pose.y = 0.105
        action.pose.a = 1.52
        action.pose.d = 0.078
        action.index = 1
        action.shift_motion = [0.03, 0.0]

        draw_pose(image, action.pose, convert_to_rgb=True)
        draw_around_box(image, box=self.box, draw_lines=True)
        imageio.imwrite(str(self.file_path / 'gen-shift-draw-pose.png'), image.mat)

        self.assertTrue(conv.shift_check_safety(action, [image]))

        conv.calculate_pose(action, [image])
        self.assertLess(action.pose.z, 0.0)

    def test_episode_history(self):
        eh = EpisodeHistory()

        def append_episode(reward: float, bin_enum: Bin) -> None:
            e = Episode()
            a = Action(action_type='grasp')
            a.reward = reward
            a.bin = bin_enum
            e.actions.append(a)
            eh.append(e)

        append_episode(reward=1.0, bin_enum=Bin.Right)
        append_episode(reward=0.0, bin_enum=Bin.Right)
        append_episode(reward=1.0, bin_enum=Bin.Left)
        append_episode(reward=0.0, bin_enum=Bin.Left)
        append_episode(reward=0.0, bin_enum=Bin.Left)
        append_episode(reward=1.0, bin_enum=Bin.Left)
        append_episode(reward=1.0, bin_enum=Bin.Left)
        append_episode(reward=1.0, bin_enum=Bin.Left)
        append_episode(reward=0.0, bin_enum=Bin.Left)
        append_episode(reward=0.0, bin_enum=Bin.Left)

        self.assertEqual(eh.total(), 10)
        self.assertEqual(eh.total_reward(), 5.0)
        self.assertEqual(eh.failed_grasps_since_last_success_in_bin(Bin.Left), 2)
        self.assertEqual(eh.successful_grasps_in_bin(Bin.Left), 4)


if __name__ == '__main__':
    unittest.main(verbosity=3)
