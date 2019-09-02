import random
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.backend as tkb  # pylint: disable=E0401

from actions.action import RobotPose
from orthographical import OrthographicImage
from picking.param import SelectionMethod


class Inference:
    size_input = (752, 480)
    size_original_cropped = (200, 200)
    size_rotated = (160, 160)
    size_output = (32, 32)
    size_cropped = (110, 110)
    size_resized = (
        int(round(size_input[0] * size_output[0] / size_original_cropped[0])),
        int(round(size_input[1] * size_output[1] / size_original_cropped[1]))
    )
    scale_factors = (
        float(size_original_cropped[0]) / size_output[0],
        float(size_original_cropped[1]) / size_output[1]
    )
    resolution_factor = 2.0

    def __init__(
            self,
            model,
            box,
            lower_random_pose=None,
            upper_random_pose=None,
            monte_carlo=None,
            input_uncertainty=False,
            with_types=False,
        ):
        self.model = model
        self.box = box

        self.lower_random_pose = lower_random_pose
        self.upper_random_pose = upper_random_pose

        self.monte_carlo = monte_carlo
        self.current_s = 0.0

        self.with_types = with_types
        self.current_type = -1

        self.input_uncertainty = input_uncertainty
        if self.input_uncertainty:
            self.uncertainty_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 1))

            estimated_reward_output = self.model.output if not self.with_types else self.model.output[0]

            self.gradients = [tkb.gradients(estimated_reward_output[:, :, :, i], self.model.input)[0] for i in range(3)]
            self.propagation_input_uncertainty = [tk.layers.Conv2D(
                filters=1,
                strides=2,
                kernel_size=32,
                use_bias=False,
                kernel_initializer=tk.initializers.constant(1.0),
                trainable=False
            )(tkb.abs(g) * self.uncertainty_placeholder) for g in self.gradients]

        if True:
            self.model.predict(np.zeros((20, 40, 40, 1)))

    def pose_from_index(self, index, index_shape, example_image: OrthographicImage) -> RobotPose:
        vec = self.rotate_vector((
            example_image.position_from_index(index[1], index_shape[1]),
            example_image.position_from_index(index[2], index_shape[2])
        ), self.a_space[index[0]])

        pose = RobotPose()
        pose.x = -vec[0] * self.resolution_factor * self.scale_factors[0]  # [m]
        pose.y = -vec[1] * self.resolution_factor * self.scale_factors[1]  # [m]
        pose.a = -self.a_space[index[0]]  # [rad]
        return pose

    @classmethod
    def rotate_vector(cls, vec, a: float) -> Tuple[float, float]:  # [rad]
        return (vec[0] * np.cos(a) + vec[1] * np.sin(a), -vec[0] * np.sin(a) + vec[1] * np.cos(a))

    @classmethod
    def method_for_monte_carlo(cls, method: SelectionMethod) -> bool:
        return method in [SelectionMethod.BayesTop, SelectionMethod.BayesProb]

    @classmethod
    def keep_array_at_last_indixes(cls, array, indixes) -> None:
        mask = np.zeros(array.shape)
        mask[:, :, :, indixes] = 1
        array *= mask

    @classmethod
    def get_filter(cls, method: SelectionMethod):
        if method == SelectionMethod.Top5:
            return cls.top_n(5)  # For planning 12
        if method == SelectionMethod.Uncertain:
            return lambda x: np.argmin(np.abs(x - 0.5))
        if method == SelectionMethod.RandomInference:
            return lambda x: random.choice(range(x.size))
        if method == SelectionMethod.NotZero:
            return lambda x: random.choice(np.flatnonzero(x >= min(0.05, np.amax(x))))
        if method == SelectionMethod.Prob:
            return lambda x: np.random.choice(range(x.size), p=(np.ravel(x) / np.sum(np.ravel(x))))
        if method == SelectionMethod.Max:
            return lambda x: x.argmax()
        raise Exception(f'Selection method not implemented: {method}')

    @classmethod
    def top_n(cls, n: int):
        return lambda x: random.choice(np.argpartition(x, -n, axis=None)[-n:])

    @classmethod
    def binary_entropy(cls, p):
        p[p == 0.0] += 1e-6
        p[p == 1.0] -= 1e-6
        return - p * np.log(p) - (1 - p) * np.log(1 - p)

    @classmethod
    def mutual_information(cls, p):
        return cls.binary_entropy(np.mean(p, axis=0)) - np.mean(list(map(cls.binary_entropy, p)), axis=0)

    @classmethod
    def probability_in_policy(cls, p, s: float):
        if s == 0:
            return np.ones(p.shape) / p.size

        p_max = np.max(p)

        def f(p):
            return (p / p_max) ** (s / (1 - s + 1e-9))

        if s == 1:
            p[p != p_max] = 0.0
            p[p == p_max] = 1.0
            return p

        result = f(p)
        return result / np.sum(result)
