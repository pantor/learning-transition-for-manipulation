from pathlib import Path

import cv2

from cfrankr import Affine
from data.loader import Loader
from picking.image import get_area_of_interest, draw_line


def print_before_after_image(episode_id: str):
    action, before_image, after_image = Loader.get_action('shifting', episode_id, ['ed-v', 'ed-after'])

    area_before_image = get_area_of_interest(before_image, action.pose, size_cropped=(300, 300), size_result=(150, 150))
    area_after_image = get_area_of_interest(after_image, action.pose, size_cropped=(300, 300))

    white = [255*255] * 3
    draw_line(area_before_image, action.pose, Affine(0, 0), Affine(0.036, 0), color=white, thickness=2)
    draw_line(area_before_image, action.pose, Affine(0.036, 0.0), Affine(0.026, -0.008), color=white, thickness=2)
    draw_line(area_before_image, action.pose, Affine(0.036, 0.0), Affine(0.026, 0.008), color=white, thickness=2)

    cv2.imwrite(str(Path.home() / 'Desktop' / f'example-{episode_id}-before.png'), area_before_image.mat)
    cv2.imwrite(str(Path.home() / 'Desktop' / f'example-{episode_id}-after.png'), area_after_image.mat)

    print('---')
    print(episode_id)
    # print('Reward before: ', action.reward)

if __name__ == '__main__':
    episode_ids = [
        '2018-12-05-14-26-17-741',
        '2019-02-20-13-42-41-781',
        '2019-02-20-13-22-41-513',
        '2019-02-15-16-21-41-933',
        '2018-12-07-14-29-55-489',
        '2018-12-07-13-09-54-067',
        '2018-12-07-12-57-30-278',
        '2018-12-07-12-56-17-399',
        '2019-02-20-10-15-53-949'
    ]

    for e in episode_ids:
        print_before_after_image(e)
