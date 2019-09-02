# Make gif: convert -delay 70 -loop 0 did_not_grasp/*.jpg did_not_grasp.gif

from pathlib import Path
import cv2

from config import Config
from data.loader import Loader
from picking.image import draw_around_box, draw_pose


for i, (d, e) in enumerate(Loader.yield_episodes('cylinder-cube-1')):
    action, image = Loader.get_action(d, e['id'], 'ed-v')
    action.pose.b = 0.3

    draw_around_box(image, box=Config.box)
    draw_pose(image, action.pose, convert_to_rgb=True)

    cv2.imshow('test', image.mat)
    cv2.waitKey(1000)

    cv2.imwrite(str(Path.home() / 'Desktop' / f'{e["id"]}.jpg'), image.mat / 255)

    if i >= 0:
        break
