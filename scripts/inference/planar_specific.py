import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from actions.action import Action
from orthographical import OrthographicImage
from cfrankr import Affine
from inference.planar import InferencePlanarPose
from picking.image import crop, draw_around_box


class InferencePlanarPoseSpecific(InferencePlanarPose):
    a_space = np.linspace(-1.484, 1.484, 20)  # [rad] # Don't use a=0.0

    def get_images(self, image: OrthographicImage):
        root = tk.Tk()
        image_array = Image.fromarray(cv2.merge(cv2.split(image.mat)))
        image_gtk = ImageTk.PhotoImage(image=image_array)

        load_new_image = False
        ann = [-1000, -1000]

        def mouse_clicked(event):
            x, y = event.x, event.y
            ann[0] = x - self.size_input[0] / 2
            ann[1] = y - self.size_input[1] / 2

            image_show = Image.fromarray(cv2.merge(cv2.split(image.mat)))
            cv2.circle(image_show, (x, y), 2, (255, 255, 255), -1)
            image_array = Image.fromarray(cv2.merge(cv2.split(image_show)))
            image_gtk = ImageTk.PhotoImage(image=image_array)
            panel.configure(image=image_gtk)
            panel.image = image_gtk

        def button_submit():
            root.destroy()

        def new_image():
            load_new_image = True
            root.destroy()

        panel = tk.Label(root, image=image_gtk)
        button = tk.Button(root, text='Grasp Object', command=button_submit)
        button2 = tk.Button(root, text='New Image', command=new_image)

        panel.pack(side='top')
        button.pack(side='bottom')
        button2.pack(side='bottom')

        panel.bind('<Button 1>', mouse_clicked)
        root.mainloop()

        if load_new_image:
            return Action()

        draw_around_box(image, box=self.box)
        background_color = image.value_from_depth(-(image.pose * Affine(0, 0, self.box['size'][2])).z)
        image_resized = cv2.resize(image, self.size_resized)

        images = []
        anns = []
        for a in self.a_space:
            rot_mat = cv2.getRotationMatrix2D(
                (self.size_resized[0] / 2, self.size_resized[1] / 2),
                a * 180.0 / np.pi,
                1.0,
            )
            rot_mat[:, 2] += [
                (self.size_rotated[0] - self.size_resized[0]) / 2,
                (self.size_rotated[1] - self.size_resized[1]) / 2,
            ]
            dst_depth = cv2.warpAffine(image_resized, rot_mat, self.size_rotated, borderValue=background_color)
            images.append(crop(dst_depth, self.size_cropped))

            ann_scaled = (float(ann[0]) / self.scale_factors[0], float(ann[1]) / self.scale_factors[1])
            ann_scaled_rot = self.rotate_vector(ann_scaled, a)
            ann_scaled_rot_pos = (
                int(round(ann_scaled_rot[0] + self.size_cropped[0] / 2)),
                int(round(ann_scaled_rot[1] + self.size_cropped[1] / 2)),
            )

            x_vec = np.zeros(self.size_cropped[0])
            y_vec = np.zeros(self.size_cropped[1])
            if 0 <= ann_scaled_rot_pos[0] < self.size_cropped[0]:
                np.put(x_vec, ann_scaled_rot_pos[0], 1)
            if 0 <= ann_scaled_rot_pos[1] < self.size_cropped[1]:
                np.put(y_vec, ann_scaled_rot_pos[1], 1)

            ann_matrix = 255 * np.outer(y_vec, x_vec)
            anns.append(ann_matrix)
        return images, anns
