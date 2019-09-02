import cv2
import numpy as np

from actions.action import Affine
from orthographical import OrthographicImage
from picking.image import crop, draw_line


class Heatmap:
    def __init__(self, inference, model, box):
        self.inf = inference(model, box=box)

    def calculate_heat(self, prob):
        size_input = (480, 752)
        size_first = (520, 520)

        prob_mean = np.mean(prob, axis=3)  # prob[:,:,:,2]
        heat_values = np.zeros(size_first, dtype=np.float)
        for a in self.inf.a_space:  # or self.inf.a_space:[9:11]
            a_idx = list(self.inf.a_space).index(a)
            prob_a = prob_mean[a_idx]
            rot_mat = cv2.getRotationMatrix2D((20, 20), -a * 180.0 / np.pi, 1.0)

            prob_rot = cv2.warpAffine(prob_a, rot_mat, (40, 40), borderValue=0)
            heat_values += cv2.resize(prob_rot, size_first)
        heat_values *= 255.0 / (heat_values.max() * len(self.inf.a_space))
        heat_values = crop(heat_values, (size_input[0], size_first[1]))

        border_size = int((size_input[1] - size_first[1]) / 2)
        return cv2.copyMakeBorder(heat_values, 0, 0, border_size, border_size, cv2.BORDER_CONSTANT)

    def render(self, image: OrthographicImage, alpha=0.5, draw_directions=False):
        prob = self.inf.model.predict(self.inf.get_images(image))
        prob = np.maximum(prob, 0)

        heat_values = self.calculate_heat(prob)

        heatmap = cv2.applyColorMap(heat_values.astype(np.uint8), cv2.COLORMAP_JET)
        result = cv2.cvtColor(image.mat, cv2.COLOR_GRAY2RGB) / 255 + alpha * cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        result = OrthographicImage(result, image.pixel_size, image.min_depth, image.max_depth)

        if draw_directions:
            for _ in range(10):
                self.draw_arrow(result, prob, np.unravel_index(prob.argmax(), prob.shape))
                prob[np.unravel_index(prob.argmax(), prob.shape)] = 0
        return result.mat

    def draw_arrow(self, image: OrthographicImage, prob, index):
        pose = self.inf.pose_from_index(index, prob.shape, image)

        arrow_color = (255, 255, 255)
        draw_line(image, pose, Affine(0, 0), Affine(0.036, 0), color=arrow_color, thickness=2)
        draw_line(image, pose, Affine(0.036, 0.0), Affine(0.026, -0.008), color=arrow_color, thickness=2)
        draw_line(image, pose, Affine(0.036, 0.0), Affine(0.026, 0.008), color=arrow_color, thickness=2)
