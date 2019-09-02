import hashlib
from multiprocessing import Pool
import time
from typing import Any, List, Union, Tuple
from pathlib import Path

import imageio
import numpy as np
from pymongo import MongoClient

from actions.action import Action
from actions.indexer import GraspIndexer
from config import Config
from data.loader import Loader
from picking.image import draw_around_box, get_area_of_interest, get_distance_to_box


class Dataset:
    def __init__(self, databases: Union[str, List[str]], validation_databases: Union[str, List[str]] = None, indexer=None):
        validation_databases = validation_databases or []

        self.databases = [databases] if isinstance(databases, str) else databases
        self.validation_databases = [validation_databases] if isinstance(validation_databases, str) else validation_databases

        self.output_path = Loader.get_database_path(self.databases[0])
        self.image_output_path = self.output_path / 'input'
        self.model_path = self.output_path / 'models'
        self.result_path = self.output_path / 'results'

        self.indexer = indexer if indexer else GraspIndexer(gripper_classes=Config.gripper_classes)
        self.box = Config.box

        self.percent_validation_set = 0.2

    def load_data(self, max_number=False, **params) -> Tuple[Any, Any]:
        params.setdefault('scale_around_zero', False)
        params.setdefault('size_input', (752, 480))
        params.setdefault('size_cropped', (200, 200))
        params.setdefault('size_output', (32, 32))

        start = time.time()

        self.image_output_path = self.output_path / f"input-{params['size_output'][0]}"
        self.image_output_path.mkdir(exist_ok=True, parents=True)
        self.model_path.mkdir(exist_ok=True, parents=True)

        mean_reward = 0.0

        episodes = []

        i = 0
        for d, e in Loader.yield_episodes(
                self.databases + self.validation_databases,
                projection={'_id': 0, 'id': 1, 'actions': {'$slice': -1}},
            ):
            episodes.append({
                'database': d,
                'episode': e,
                **params,
            })
            mean_reward += e['actions'][0]['reward']

            i += 1
            if max_number and i >= max_number:
                break

        episodes = list(map(self.assign_set, episodes))

        if not episodes:
            raise Exception('No episodes could be loaded.')
        print(f'Loading {len(episodes)} episodes.')

        train_episodes = filter(lambda x: not x['is_validation'], episodes)
        validation_episodes = filter(lambda x: x['is_validation'], episodes)

        def set_loader():
            Loader.client = MongoClient()  # pymongo will output a warning if created after fork thread method

        def load_data(episodes):
            features, labels, infos = [], [], []
            for element in p.imap_unordered(self.load_element, episodes):
                features += element[0]
                labels += element[1]
                infos += element[2]
            data_x = [np.array(t) for t in zip(*features)]
            data_y = [np.array(labels), np.array(infos)]
            return data_x, data_y

        p = Pool(8, initializer=set_loader)

        train_set = load_data(train_episodes)
        validation_set = load_data(validation_episodes)

        p.close()

        print(f'Train set: {len(train_set[0][0])}')
        print(f'Validation set: {len(validation_set[0][0])}')
        print(f'Mean reward: {mean_reward / len(episodes):0.3}')

        end = time.time()
        print(f'Time [s]: {end-start:0.4}')
        return train_set, validation_set

    def load_element(self, params) -> Tuple[List[Any], List[Any]]:
        episode_id = params['episode']['id']
        action = Action(data=params['episode']['actions'][0])

        features, labels, info = [], [], []
        for suffix_list in params['suffixes']:
            suffix_list_taken = [s for s in suffix_list if s in action.images.keys()]
            if len(suffix_list_taken) < len(suffix_list):
                continue

            try:
                index = self.indexer.from_action(action, suffix_list_taken[0])
            except:
                raise Exception(f'Error in indexer at {params["database"]}, {episode_id}.')
            if index is None:
                continue

            for s in suffix_list:
                if not self.check_output_image(episode_id, s) or params['force_images']:
                    image = self.get_image(params['database'], episode_id, s)
                    draw_around_box(image, box=self.box)
                    area_image = self.get_area_image(image, action.pose, params['size_cropped'], params['size_output']).mat
                    self.write_output_image(area_image, episode_id, s)

            features.append([
                self.get_output_image(episode_id, s, scale_around_zero=params['scale_around_zero']) for s in suffix_list
            ])
            labels.append([action.reward, index])

            object_type = [1, 0, 0]
            if 'final_pose' in action.__dict__ and action.reward == 1:
                object_type = [0, 0, 1] if action.final_pose.d > 0.035 else [0, 1, 0]
            info.append(
                object_type
            )

        return features, labels, info

    def get_image(self, database: str, episode_id: str, suffix: str):
        image = Loader.get_image(database, episode_id, suffix)
        if image is None:
            print(f'File not found: {database}, {episode_id}, {suffix}')
        return image

    def get_area_image(self, image, pose, size_cropped, size_output, border_color=None):
        box_distance = get_distance_to_box(image, self.box)
        border_color = border_color if border_color else image.value_from_depth(box_distance)
        return get_area_of_interest(
            image,
            pose,
            size_cropped=size_cropped,
            size_result=size_output,
            border_color=border_color
        )

    def get_output_image(self, episode_id: str, suffix: str, scale_around_zero=False):
        mat_image = imageio.imread(str(self.get_output_image_path(episode_id, suffix)), 'PNG-PIL')
        if len(mat_image.shape) < 3:
            mat_image = np.expand_dims(mat_image, axis=-1)

        image = mat_image.astype(np.float32) / np.iinfo(np.uint16).max
        if scale_around_zero:
            return 2 * image - 1.0
        return image

    def get_output_image_path(self, episode_id: str, suffix: str) -> Path:
        return self.image_output_path / f'image-{episode_id}-{suffix}.png'

    def check_output_image(self, episode_id: str, suffix: str) -> bool:
        return self.get_output_image_path(episode_id, suffix).is_file()

    def write_output_image(self, image, episode_id: str, suffix: str) -> None:
        imageio.imwrite(str(self.get_output_image_path(episode_id, suffix)), image)

    @staticmethod
    def binary_decision(string: str, p: float) -> bool:
        return float(int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % 2**16) / 2**16 < p

    def assign_set(self, data):
        random_assign = Dataset.binary_decision(data['episode']['id'], self.percent_validation_set)
        data['is_validation'] = random_assign or (data['database'] in self.validation_databases)
        return data

    @staticmethod
    def get_index_count(data_set):
        return np.bincount(np.array(data_set[1])[0][:, 1].astype(np.int32))
