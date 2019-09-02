from pathlib import Path
import time
from typing import List

import cv2
from loguru import logger
import numpy as np

from actions.action import Action
from actions.converter import Converter
from actions.indexer import GraspIndexer, ShiftIndexer, GraspShiftIndexer
from data.loader import Loader
from config import Config
from inference.planar import InferencePlanarPose
from picking.epoch import Epoch
from picking.image import clone, draw_around_box, draw_pose, get_area_of_interest, patch_image_at
from picking.param import SelectionMethod
from picking.planning_tree import PlanningTree
from orthographical import OrthographicImage


class Agent:
    def __init__(self, prediction_model):
        self.grasp_model = Config.grasp_model
        self.shift_model = Config.shift_model

        self.with_types = 'types' in self.grasp_model[1]

        self.output_layer = 'prob' if not self.with_types else ['prob', 'type']
        self.grasp_inference = InferencePlanarPose(
            Loader.get_model(self.grasp_model, output_layer=self.output_layer),
            box=Config.box,
            lower_random_pose=Config.lower_random_pose,
            upper_random_pose=Config.upper_random_pose,
            with_types=self.with_types,
            input_uncertainty=True,
        )
        self.grasp_inference.keep_indixes = [0, 1]
        self.grasp_indexer = GraspIndexer(gripper_classes=Config.gripper_classes)

        self.shift_inference = InferencePlanarPose(
            Loader.get_model(self.shift_model, output_layer='prob'),
            box=Config.box,
            lower_random_pose=Config.lower_random_pose,
            upper_random_pose=Config.upper_random_pose,
            with_types=False,
        )
        self.shift_inference.a_space = np.linspace(-3.0, 3.0, 26)  # [rad] # Don't use a=0.0
        self.shift_inference.size_original_cropped = (240, 240)
        self.shift_indexer = ShiftIndexer(shift_distance=Config.shift_distance)

        self.grasp_shift_indexer = GraspShiftIndexer(
            gripper_classes=Config.gripper_classes,
            shift_distance=Config.shift_distance,
        )

        self.converter = Converter(grasp_z_offset=Config.grasp_z_offset, shift_z_offset=0.007, box=Config.box)  # [m]

        self.prediction_model = prediction_model
        self.monte_carlo = 20

        self.actions_since_last_inference = 0
        self.actions: List[Action] = []

        self.output_path = Path.home() / 'Desktop'

        self.reinfer_next_time = True

        # First inference is slower
        self.prediction_model.predict([np.zeros((1, 64, 64, 1)), np.zeros((1, 1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((1, 1, 1, 8))])

    def predict_images_after_action(
            self,
            images: List[OrthographicImage],
            action: Action,
            reward: float,
            action_type: int,
            uncertainty_images=None,
        ) -> List[OrthographicImage]:
        image = images[0]
        uncertainty_image = uncertainty_images[0]

        start = time.time()
        draw_around_box(image, box=Config.box)
        area = get_area_of_interest(image, action.pose, size_cropped=(256, 256), size_result=(64, 64))

        area_input = np.expand_dims(area.mat, axis=2).astype(np.float32) / np.iinfo(np.uint16).max * 2 - 1
        reward = np.expand_dims(np.expand_dims(np.expand_dims(reward, axis=1), axis=1), axis=1).astype(np.float32)
        action_type = np.expand_dims(np.expand_dims(action_type, axis=1), axis=1)

        use_monte_carlo = self.monte_carlo and self.monte_carlo > 1

        if not use_monte_carlo:
            area_result = self.prediction_model.predict([[area_input], [reward], [action_type], np.zeros((1, 1, 1, 8))])[0]
            area_result = np.array(np.iinfo(np.uint16).max * (area_result + 1) / 2, dtype=np.uint16)

        else:
            latent = np.random.normal(scale=0.05, size=(self.monte_carlo, 1, 1, 8))
            if self.monte_carlo > 3:
                latent[0, :, :, :] = 0.0

            predictions = self.prediction_model.predict([
                [area_input for _ in range(self.monte_carlo)],
                [reward for _ in range(self.monte_carlo)],
                [action_type for _ in range(self.monte_carlo)],
                latent,
            ])
            predictions = (predictions + 1) / 2

            predictions = np.array(predictions, dtype=np.float32)
            area_result = predictions[0]
            area_result = np.array(np.iinfo(np.uint16).max * area_result, dtype=np.uint16)

            predictions[predictions < 0.1] = np.nan
            area_uncertainty = np.nanvar(predictions, axis=0)
            area_uncertainty *= 7e3
            area_uncertainty[area_uncertainty > 1] = 1
            area_uncertainty = np.array(np.iinfo(np.uint16).max * area_uncertainty, dtype=np.uint16)

            uncertainty_image = patch_image_at(
                uncertainty_image,
                area_uncertainty,
                action.pose,
                size_cropped=(256, 256),
                operation='add',
            )

        result = patch_image_at(image, area_result, action.pose, size_cropped=(256, 256))

        logger.info(f'Predicted image [s]: {time.time() - start:0.3f}')

        if use_monte_carlo:
            return [result], [uncertainty_image]
        return [result]

    def plan_actions(
            self,
            images: List[OrthographicImage],
            method: SelectionMethod,
            depth=1,
            leaves=1,
            verbose=False,
        ) -> List[Action]:

        uncertainty_images = [OrthographicImage(
            np.zeros(i.mat.shape, dtype=np.uint16),
            i.pixel_size,
            i.min_depth,
            i.max_depth,
            i.camera,
            i.pose
        ) for i in images]

        tree = PlanningTree(images, uncertainty_images)

        for node, i in tree.fill_nodes(leaves=leaves, depth=depth):
            # Visited actions: node.actions

            for image in node.images:
                draw_around_box(image, box=Config.box)

            grasp = self.grasp_inference.infer(node.images, method, uncertainty_images=node.uncertainty_images)
            self.grasp_indexer.to_action(grasp)

            # Shift actions
            if Config.shift_objects and grasp.estimated_reward < Config.grasp_shift_threshold:
                shift = self.shift_inference.infer(node.images, method)
                self.shift_indexer.to_action(shift)

                bin_empty = shift.estimated_reward < Config.shift_empty_threshold

                if bin_empty:
                    action = Action('bin_empty', safe=1)
                else:
                    self.converter.calculate_pose(shift, node.images)
                    action = shift

            # Grasp actions
            else:
                estimated_reward_lower_than_threshold = grasp.estimated_reward < Config.bin_empty_at_max_probability
                bin_empty = estimated_reward_lower_than_threshold and Epoch.selection_method_should_be_high(method)
                new_image = False

                if bin_empty:
                    action = Action('bin_empty', safe=1)
                elif grasp.estimated_reward_std > 0.9:  # default=0.25
                    action = Action('new_image', safe=1)
                else:
                    self.converter.calculate_pose(grasp, node.images)
                    action = grasp
            logger.info(f'{i}: {action}')

            if verbose:
                image_copy = clone(images[0])
                uncertainty_image_copy = clone(uncertainty_images[0])

                draw_pose(image_copy, action.pose, convert_to_rgb=True)
                draw_pose(uncertainty_image_copy, action.pose, convert_to_rgb=True)

                cv2.imwrite(str(self.output_path / f'result-{i}.png'), image_copy.mat)
                cv2.imwrite(str(self.output_path / f'uncertainty-{i}.png'), uncertainty_image_copy.mat)

            if action.type == 'bin_empty' or action.type == 'new_image':
                break

            # Predict next image
            reward = action.estimated_reward > Config.bin_empty_at_max_probability if action.type == 'grasp' else action.estimated_reward
            action_type = self.grasp_shift_indexer.from_action(action)
            images = self.predict_images_after_action(
                node.images,
                action,
                reward=reward,
                action_type=action_type,
                uncertainty_images=node.uncertainty_images,
            )

            if isinstance(images, tuple):
                images, uncertainty_images = images
            else:
                uncertainty_images = None

            node.add_action(action, images, uncertainty_images)

        if verbose:
            cv2.imwrite(str(self.output_path / f'result-{i+1}.png'), node.images[0].mat)
            cv2.imwrite(str(self.output_path / f'uncertainty-{i+1}.png'), node.uncertainty_images[0].mat)

        actions, max_reward, mean_reward = tree.get_actions_maximize_reward(max_depth=depth)
        print(f'Max reward: {max_reward:0.3f}, Mean reward: {mean_reward:0.3f}, Length: {len(actions)}')

        # actions, max_steps, mean_steps = tree.get_actions_minimize_steps()
        return actions

    def predict_actions(
            self,
            images: List[OrthographicImage],
            method: SelectionMethod,
            N=1,
            verbose=True,
        ) -> List[Action]:

        actions: List[Action] = []

        uncertainty_images = [OrthographicImage(
            np.zeros(i.mat.shape, dtype=np.uint16),
            i.pixel_size,
            i.min_depth,
            i.max_depth,
            i.camera,
            i.pose
        ) for i in images]

        for i in range(N):
            for image in images:
                draw_around_box(image, box=Config.box)

            grasp = self.grasp_inference.infer(images, method, uncertainty_images=uncertainty_images)
            self.grasp_indexer.to_action(grasp)

            # Shift actions
            if Config.shift_objects and grasp.estimated_reward < Config.grasp_shift_threshold:
                shift = self.shift_inference.infer(images, method)
                self.shift_indexer.to_action(shift)

                bin_empty = shift.estimated_reward < Config.shift_empty_threshold

                if bin_empty:
                    actions.append(Action('bin_empty', safe=1))
                else:
                    self.converter.calculate_pose(shift, images)
                    actions.append(shift)

            # Grasp actions
            else:
                estimated_reward_lower_than_threshold = grasp.estimated_reward < Config.bin_empty_at_max_probability
                bin_empty = estimated_reward_lower_than_threshold and Epoch.selection_method_should_be_high(method)
                new_image = False

                if bin_empty:
                    actions.append(Action('bin_empty', safe=1))
                elif grasp.estimated_reward_std > 0.9:  # default=0.25
                    actions.append(Action('new_image', safe=1))
                else:
                    self.converter.calculate_pose(grasp, images)
                    actions.append(grasp)

            actions[-1].step = i
            action = actions[-1]
            logger.info(f'{i}: {action}')

            if verbose:
                image_copy = clone(images[0])
                uncertainty_image_copy = clone(uncertainty_images[0])

                draw_pose(image_copy, action.pose, convert_to_rgb=True)
                draw_pose(uncertainty_image_copy, action.pose, convert_to_rgb=True)

                cv2.imwrite(str(self.output_path / f'result-{i}.png'), image_copy.mat)
                cv2.imwrite(str(self.output_path / f'uncertainty-{i}.png'), uncertainty_image_copy.mat)

            if action.type == 'bin_empty' or action.type == 'new_image':
                break

            # Predict next image
            reward = action.estimated_reward > Config.bin_empty_at_max_probability if action.type == 'grasp' else action.estimated_reward
            action_type = self.grasp_shift_indexer.from_action(action)
            images = self.predict_images_after_action(
                images,
                action,
                reward=reward,
                action_type=action_type,
                uncertainty_images=uncertainty_images,
            )

            if isinstance(images, tuple):
                images, uncertainty_images = images
            else:
                uncertainty_images = None

        if verbose:
            cv2.imwrite(str(self.output_path / f'result-{i+1}.png'), images[0].mat)
            cv2.imwrite(str(self.output_path / f'uncertainty-{i+1}.png'), uncertainty_images[0].mat)

        return actions

    def infer(self, images: List[OrthographicImage], method: SelectionMethod, N=5, reinfer=False):
        if self.actions_since_last_inference == 0 or self.actions_since_last_inference >= N or self.reinfer_next_time or reinfer:
            logger.warning(f'Calculate {N} predictions.')

            # self.actions = self.predict_actions(images, method, N=(N+1))
            self.actions = self.plan_actions(images, method, depth=N, leaves=1)
            self.actions_since_last_inference = 0
            self.reinfer_next_time = False
        else:
            logger.warning(f'Saved action, last inference {self.actions_since_last_inference} actions ago.')
            if self.actions_since_last_inference == len(self.actions) - 2:
                self.reinfer_next_time = True

        self.actions_since_last_inference += 1
        return self.actions[self.actions_since_last_inference - 1]

