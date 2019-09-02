from pathlib import Path

import cv2
import numpy as np

from config import Config
from data.loader import Loader
from learning.utils.layers import one_hot_gen
from picking.image import draw_around_box, get_area_of_interest


class PredictBicycle:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.uncertainty = True
        self.size = (64, 64)

    def predict(self, area, reward, action_type, sampling=False, number=1):
        area_input = np.expand_dims(area.mat, axis=2).astype(np.float32) / np.iinfo(np.uint16).max * 2 - 1
        reward = np.expand_dims(np.expand_dims(np.expand_dims(reward, axis=1), axis=1), axis=1).astype(np.float32)
        action_type = np.expand_dims(np.expand_dims(action_type, axis=1), axis=1)
        latent_size = (number, 1, 1, 8)

        latent = np.random.normal(scale=0.5, size=latent_size) if sampling else np.zeros(latent_size)

        if number > 3:
            latent[0, :, :, :] = 0.0

        result = self.model.predict([
            [area_input for _ in range(number)],
            [reward for _ in range(number)],
            [action_type for _ in range(number)],
            latent
        ])
        result = (result + 1) / 2
        return result


class PredictPix2Pix:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.uncertainty = False
        self.size = (128, 128)

    def predict(self, area, reward, action_type, sampling=False, number=1):
        area_input = np.expand_dims(area.mat, axis=2).astype(np.float32) / np.iinfo(np.uint16).max * 2 - 1

        if not sampling:
            result = self.model.predict([[area_input], [reward], [action_type]])[0]
            result = (result + 1) / 2
            return result

        predictions = []
        for _ in range(number):
            result = self.model.predict([[area_input], [reward], [action_type]])[0]
            result = (result + 1) / 2
            predictions.append(result)

        return np.array(predictions, dtype=np.float32)

        #predictions = np.array(predictions, dtype=np.float32)
        #result = np.mean(predictions, axis=0)

        #predictions[predictions < 0.1] = np.nan
        #uncertainty = np.nanvar(predictions, axis=0)
        #uncertainty /= np.nanmax(uncertainty) * 0.5
        #return result, uncertainty


class PredictVAE:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.uncertainty = True
        self.size = (64, 64)

    def predict(self, area, reward, action_type, sampling=False, number=10):
        area_input = np.expand_dims(area.mat, axis=2).astype(np.float32) / np.iinfo(np.uint16).max * 2 - 1
        result = self.model.predict([[area_input], [reward], [action_type]])[0]

        predictions = []
        for _ in range(number):
            result = self.model.predict([[area_input], [reward], [action_type]])[0][0]
            result = (result + 1) / 2
            predictions.append(result)

        return np.array(predictions, dtype=np.float32)


def save_episode(predictor, database, episode_id, reward=1, action_type=1):
    action, image, image_after = Loader.get_action(database, episode_id, ['ed-v', 'ed-after'])

    draw_around_box(image, box=Config.box)
    draw_around_box(image_after, box=Config.box)

    # background_color = image.value_from_depth(get_distance_to_box(image, Config.box))

    area = get_area_of_interest(image, action.pose, size_cropped=(256, 256), size_result=predictor.size)
    area_after = get_area_of_interest(image_after, action.pose, size_cropped=(256, 256), size_result=predictor.size)

    result = predictor.predict(area, reward=reward, action_type=action_type, sampling=True, number=20)

    save_dir = Path.home() / 'Desktop' / 'predict-examples' / episode_id
    save_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(save_dir / f'{predictor.name}_s_bef.png'), area.mat)
    cv2.imwrite(str(save_dir / f'{predictor.name}_s_aft.png'), area_after.mat)
    cv2.imwrite(str(save_dir / f'{predictor.name}_result.png'), result[0] * 255)

    if predictor.uncertainty:
        result[result < 0.1] = np.nan
        uncertainty = np.nanvar(result, axis=0)
        uncertainty /= np.nanmax(uncertainty) * 0.25

        uncertainty = np.clip(uncertainty * 255, 0, 255).astype(np.uint8)
        uncertainty = cv2.applyColorMap(uncertainty, cv2.COLORMAP_JET)

        cv2.imwrite(str(save_dir / f'{predictor.name}_unc.png'), uncertainty)

    # for i in range(3):
    #     cv2.imwrite(str(save_dir / f'{predictor.name}_result_{i}.png'), result[i] * 255)


    # cv2.waitKey(1000)


if __name__ == '__main__':
    predict_bicycle = PredictBicycle(Loader.get_model('cube-1', 'predict-bi-gen-5'), name='bi')
    predict_pix2pix = PredictPix2Pix(Loader.get_model('cube-1', 'predict-generator-4'), name='pix2pix')
    predict_vae = PredictVAE(Loader.get_model('cube-1', 'predict-vae-2'), name='vae')

    for p in [
        predict_bicycle,
        predict_pix2pix,
        predict_vae
    ]:
        save_episode(p, 'cylinder-cube-1', '2019-03-11-14-56-07-284')
        save_episode(p, 'cylinder-cube-1', '2019-07-01-14-05-53-150')
        save_episode(p, 'cylinder-cube-1', '2019-07-01-13-50-06-016')
        save_episode(p, 'cylinder-cube-1', '2019-07-01-11-10-31-450', reward=0)
        save_episode(p, 'shifting', '2018-12-11-09-54-26-687', reward=0.8, action_type=3)
        save_episode(p, 'small-cubes-2', '2019-01-11-14-24-57-186', action_type=0)
        save_episode(p, 'small-cubes-2', '2019-01-11-14-52-32-222', action_type=2)
        save_episode(p, 'cylinder-cube-1', '2019-06-29-01-08-57-366')
        save_episode(p, 'all-1', '2018-11-13-12-40-21-633')
        save_episode(p, 'all-1', '2018-11-13-11-27-17-195', action_type=2)
        save_episode(p, 'all-1', '2018-11-13-12-42-04-331')

