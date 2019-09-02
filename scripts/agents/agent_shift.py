from typing import List

import numpy as np

from actions.action import Action
from actions.converter import Converter
from actions.indexer import GraspIndexer, ShiftIndexer
from orthographical import OrthographicImage
from config import Config
from data.loader import Loader
from inference.planar import InferencePlanarPose
from picking.epoch import Epoch
from picking.param import SelectionMethod


class Agent:
    def __init__(self):
        self.grasp_inference = InferencePlanarPose(
            model=Loader.get_model(Config.grasp_model, output_layer='prob'),
            box=Config.box,
            lower_random_pose=Config.lower_random_pose,
            upper_random_pose=Config.upper_random_pose,
        )
        self.grasp_indexer = GraspIndexer(gripper_classes=Config.gripper_classes)

        self.converter = Converter(grasp_z_offset=Config.grasp_z_offset, shift_z_offset=0.007, box=Config.box)  # [m]

        if Config.shift_objects:
            self.shift_inference = InferencePlanarPose(
                model=Loader.get_model(Config.shift_model, output_layer='prob'),
                box=Config.box,
                lower_random_pose=Config.lower_random_pose,
                upper_random_pose=Config.upper_random_pose,
            )
            self.shift_inference.a_space = np.linspace(-3.0, 3.0, 26)  # [rad] # Don't use a=0.0
            self.shift_inference.size_original_cropped = (240, 240)
            self.shift_indexer = ShiftIndexer(shift_distance=Config.shift_distance)

        self.reinfer_next_time = True  # Always true in contrast to AgentPredict

    def infer(self, images: List[OrthographicImage], method: SelectionMethod) -> Action:
        if len(images) == 3:
            images[2].mat = images[2].mat[:, :, ::-1]  # BGR to RGB

        grasp = self.grasp_inference.infer(images, method)
        self.grasp_indexer.to_action(grasp)

        estimated_reward_lower_than_threshold = grasp.estimated_reward < Config.bin_empty_at_max_probability
        bin_empty = estimated_reward_lower_than_threshold and Epoch.selection_method_should_be_high(method)

        if Config.shift_objects and grasp.estimated_reward < Config.grasp_shift_threshold:
            shift = self.shift_inference.infer(images, method)
            self.shift_indexer.to_action(shift)

            if shift.estimated_reward > Config.shift_empty_threshold:
                self.converter.calculate_pose(shift, images)
                return shift
            return Action('bin_empty', safe=1)

        if bin_empty:
            return Action('bin_empty', safe=1)

        self.converter.calculate_pose(grasp, images)
        return grasp

    def infer_shift(self, images: List[OrthographicImage], method: SelectionMethod) -> Action:
        shift = self.shift_inference.infer(images, method)
        self.shift_indexer.to_action(shift)
        return shift

    def infer_max_grasp_reward(self, images: List[OrthographicImage]) -> float:
        return self.grasp_inference.infer(images, SelectionMethod.Max).estimated_reward

    def infer_max_grasp_reward_around_action(
                self,
                images: List[OrthographicImage],
                action: Action,
                window=(0.13, 0.13)
        ) -> float:  # [m]
        input_images = [self.grasp_inference.get_images(d) for d in images]
        estimated_reward = self.grasp_inference.model.predict(input_images)

        for index_raveled in range(estimated_reward.size):
            index = np.unravel_index(index_raveled, estimated_reward.shape)
            pose = self.grasp_inference.pose_from_index(index, estimated_reward.shape, images[0])

            if not (
                    (action.pose.x - window[0] / 2 < pose.x < action.pose.x + window[0] / 2) and
                    (action.pose.y - window[1] / 2 < pose.y < action.pose.y + window[1] / 2)
                ):
                estimated_reward[index] = 0.0

        return np.max(estimated_reward)
