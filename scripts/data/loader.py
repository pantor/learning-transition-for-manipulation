from pathlib import Path
from typing import Any, List, Union, Tuple, Optional

import imageio
from pymongo import MongoClient
from tensorflow.keras import Model, models  # pylint: disable=E0401

from actions.action import Action
from orthographical import OrthographicImage


class Loader:
    client = MongoClient()

    @classmethod
    def get_databases(cls):
        return [d for d in cls.client.list_database_names() if d not in ('admin', 'config', 'local')]

    @classmethod
    def get_episode(cls, database: str, episode_id: str):
        return cls.client[database].episodes.find_one({'id': episode_id})

    @classmethod
    def get_episode_count(cls, databases: Union[str, List[str]], query=None, suffixes=None):
        query = query or {}

        if isinstance(databases, str):
            databases = [databases]

        if suffixes:
            for s in suffixes:
                query[f'actions.0.images.{s}'] = {'$exists': True}

        return sum(cls.client[d].episodes.count(query) for d in databases)

    @classmethod
    def yield_episodes(cls, databases: Union[str, List[str]], query=None, suffixes=None, projection=None):
        query = query or {}

        if isinstance(databases, str):
            databases = [databases]

        if suffixes:
            for s in suffixes:
                query[f'actions.0.images.{s}'] = {'$exists': True}

        for database in databases:
            if 'episodes' not in cls.client[database].list_collection_names():
                print(f'Database {database} has no episode collection!')
                continue

            for episode in cls.client[database].episodes.find(query, projection):
                yield database, episode

    @classmethod
    def get_database_path(cls, database: str) -> Path:
        return Path.home() / 'Documents' / 'data' / database

    @classmethod
    def get_image_path(cls, database: str, episode_id: str, suffix: str) -> Path:
        return Path.home() / 'Documents' / 'data' / database / 'measurement' / f'image-{episode_id}-{suffix}.png'

    @classmethod
    def get_image(cls, database: str, episode_id: str, suffix: str, images=None) -> Optional[OrthographicImage]:
        try:
            image = imageio.imread(str(cls.get_image_path(database, episode_id, suffix)), 'PNG-PIL')
        except FileNotFoundError:
            return None
        if image.dtype == 'uint8':
            image = image.astype('uint16') * 255
        if len(image.shape) == 3:
            image = image[:, :, ::-1]

        if not images:
            episode = cls.get_episode(database, episode_id)
            if not episode:
                return None

            images = Action(data=episode['actions'][0]).images

        return OrthographicImage(
            image,
            images[suffix]['info']['pixel_size'],
            images[suffix]['info']['min_depth'],
            images[suffix]['info']['max_depth'],
            '',
            images[suffix]['pose'],
        )

    @classmethod
    def get_action(cls, database: str, episode_id: str, suffix: Union[str, List[str]] = None) -> Any:
        episode = Loader.get_episode(database, episode_id)
        if not episode:
            return None

        action = Action(data=episode['actions'][0])
        if not suffix:
            return action

        if isinstance(suffix, str):
            suffix = [suffix]

        return (action, *[cls.get_image(database, episode_id, s, images=action.images) for s in suffix])

    @classmethod
    def get_model_path(cls, database: Union[str, Tuple[str, str]], name: str = None) -> Path:
        if isinstance(database, tuple):  # To load (database, model) as tuple
            database, name = database

        return Path.home() / 'Documents' / 'data' / database / 'models' / f'{name}.h5'

    @classmethod
    def get_model(
            cls,
            database: Union[str, Tuple[str, str]],
            name: str = None,
            output_layer: Union[str, List[str]] = None,
            custom_objects=None
        ):
        model_path = cls.get_model_path(database, name)
        model = models.load_model(str(model_path), compile=False, custom_objects=custom_objects)
        if output_layer and isinstance(output_layer, list):
            return Model(inputs=model.input, outputs=[model.get_layer(l).output for l in output_layer])
        elif output_layer:
            return Model(inputs=model.input, outputs=model.get_layer(output_layer).output)
        return model
