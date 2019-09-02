import time
from typing import List

from loguru import logger
import numpy as np
import tensorflow.keras.backend as tkb  # pylint: disable=E0401

from actions.action import Action, RobotPose
from orthographical import OrthographicImage
from cfrankr import Affine
from inference.inference import Inference
from picking.image import clone, crop, draw_around_box, get_area_of_interest, get_distance_to_box
from picking.param import SelectionMethod

import picking.path_fix_ros  # pylint: disable=W0611
import cv2  # pylint: disable=C0411


class InferencePlanarPose(Inference):
    a_space = np.linspace(-1.484, 1.484, 16)  # [rad] # Don't use a=0.0
    keep_indixes = None

    def get_images(self, orig_image: OrthographicImage):
        image = clone(orig_image)

        draw_around_box(image, box=self.box)
        background_color = image.value_from_depth(get_distance_to_box(image, self.box))

        mat_image_resized = cv2.resize(image.mat, self.size_resized)

        mat_images = []
        for a in self.a_space:
            rot_mat = cv2.getRotationMatrix2D(
                (self.size_resized[0] / 2, self.size_resized[1] / 2),
                a * 180.0 / np.pi,
                1.0
            )
            rot_mat[:, 2] += [
                (self.size_rotated[0] - self.size_resized[0]) / 2,
                (self.size_rotated[1] - self.size_resized[1]) / 2
            ]
            dst_depth = cv2.warpAffine(mat_image_resized, rot_mat, self.size_rotated, borderValue=background_color)
            mat_images.append(crop(dst_depth, self.size_cropped))

        mat_images = np.array(mat_images) / np.iinfo(image.mat.dtype).max
        if len(mat_images.shape) == 3:
            mat_images = np.expand_dims(mat_images, axis=-1)

        # mat_images = 2 * mat_images - 1.0
        return mat_images

    def infer(
            self,
            images: List[OrthographicImage],
            method: SelectionMethod,
            verbose=1,
            uncertainty_images: List[OrthographicImage] = None,
        ) -> Action:

        start = time.time()

        if method == SelectionMethod.Random:
            action = Action()
            action.index = np.random.choice(range(3))
            action.pose = RobotPose(affine=Affine(
                x=np.random.uniform(self.lower_random_pose[0], self.upper_random_pose[0]),  # [m]
                y=np.random.uniform(self.lower_random_pose[1], self.upper_random_pose[1]),  # [m]
                a=np.random.uniform(self.lower_random_pose[3], self.upper_random_pose[3]),  # [rad]
            ))
            action.estimated_reward = -1
            action.estimated_reward_std = 0.0
            action.method = method
            action.step = 0
            return action

        input_images = [self.get_images(i) for i in images]
        result = self.model.predict(input_images)

        if self.with_types:
            estimated_reward = result[0]
            types = result[1]
        else:
            estimated_reward = result

        estimated_reward_std = np.zeros(estimated_reward.shape)

        filter_method = method
        filter_measure = estimated_reward


        # Calculate model uncertainty
        if self.monte_carlo:
            rewards_sampling = [self.model.predict(input_images) for i in range(self.monte_carlo)]
            estimated_reward = np.mean(rewards_sampling, axis=0)
            estimated_reward_std += self.mutual_information(rewards_sampling)

            if verbose:
                logger.info(f'Current monte carlo s: {self.current_s}')


        # Calculate input uncertainty
        if self.input_uncertainty:
            input_uncertainty_images = [self.get_images(i) for i in uncertainty_images]

            array_estimated_unc = tkb.get_session().run(
                self.propagation_input_uncertainty,
                feed_dict={self.model.input: input_images[0], self.uncertainty_placeholder: input_uncertainty_images[0]}
            )
            estimated_input_uncertainty = np.concatenate(array_estimated_unc, axis=3)
            estimated_reward_std += 0.0025 * estimated_input_uncertainty


        # Adapt filter measure for more fancy SelectionMethods
        if method == SelectionMethod.Top5LowerBound:
            filter_measure = estimated_reward - estimated_reward_std
            filter_method = SelectionMethod.Top5
        elif method == SelectionMethod.BayesTop:
            filter_measure = self.probability_in_policy(estimated_reward, s=self.current_s) * estimated_reward_std
            filter_method = SelectionMethod.Top5
        elif method == SelectionMethod.BayesProb:
            filter_measure = self.probability_in_policy(estimated_reward, s=self.current_s) * estimated_reward_std
            filter_method = SelectionMethod.Prob

        filter_lambda = self.get_filter(filter_method)


        # Set some action (indices) to zero
        if self.keep_indixes:
            self.keep_array_at_last_indixes(filter_measure, self.keep_indixes)


        # Grasp specific types
        if self.with_types and self.current_type > -1:
            alpha = 0.7
            factor_current_type = np.tile(np.expand_dims(types[:, :, :, self.current_type], axis=-1), reps=(1, 1, 1, estimated_reward.shape[-1]))
            factor_alt_type = np.tile(np.expand_dims(types[:, :, :, 1], axis=-1), reps=(1, 1, 1, estimated_reward.shape[-1]))

            filter_measure = estimated_reward * (alpha * factor_current_type + (1 - alpha) * factor_alt_type)


        # Find the index and corresponding action using the selection method
        index_raveled = filter_lambda(filter_measure)
        index = np.unravel_index(index_raveled, filter_measure.shape)

        action = Action()
        action.index = index[3]
        action.pose = self.pose_from_index(index, filter_measure.shape, images[0])
        action.estimated_reward = estimated_reward[index]
        action.estimated_reward_std = estimated_reward_std[index]
        action.method = method
        action.step = 0  # Default value

        if verbose:
            logger.info(f'NN inference time [s]: {time.time() - start:.3}')
        return action

    def infer_at_pose(self, images: List[OrthographicImage], pose: Affine):
        input_images = []
        for image in images:
            draw_around_box(image, box=self.box)

            area_mat = get_area_of_interest(
                image,
                pose,
                size_cropped=(200, 200),
                size_result=(32, 32),
                border_color=image.value_from_depth(get_distance_to_box(image, self.box)),
            ).mat
            area_mat = np.array(area_mat, dtype=np.float32) / np.iinfo(area_mat.dtype).max # 2 * x - 1.0
            area_mat_exp = np.expand_dims(area_mat, axis=-1)
            input_images.append(area_mat_exp)

        if self.monte_carlo:
            input_images_mc = [np.array([image for i in range(self.monte_carlo)]) for image in input_images]
            estimated_rewards_sampling = self.model.predict(input_images_mc)
            estimated_reward = np.mean(estimated_rewards_sampling, axis=0)
            estimated_reward_std = self.mutual_information(estimated_rewards_sampling)
            return estimated_reward, estimated_reward_std

        return self.model.predict([input_images])[0]
