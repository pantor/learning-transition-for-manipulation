import io
import json

import cv2
import flask
import flask_socketio
import numpy as np
import pymongo as pm
from werkzeug.datastructures import ImmutableOrderedMultiDict

from actions.action import Action
from config import Config
from data.loader import Loader
from picking.image import draw_pose, draw_around_box


class MyRequest(flask.Request):
    parameter_storage_class = ImmutableOrderedMultiDict


class MyFlask(flask.Flask):
    request_class = MyRequest


app = MyFlask(__name__)
socketio = flask_socketio.SocketIO(app)


client = pm.MongoClient()


@app.route('/api/database-list')
def api_database_list():
    database_list = Loader.get_databases()
    return flask.jsonify(database_list)


@app.route('/api/episodes')
def api_episodes():
    database = client[flask.request.values.get('database')]

    if 'episodes' not in database.collection_names():
        return flask.abort(404)

    filter_dict = {}
    values = flask.request.values

    if values.get('id'):
        filter_dict['id'] = {'$regex': values.get('id')}

    if values.get('reward') and int(values.get('reward')) > -1:
        filter_dict['actions.0.reward'] = int(values.get('reward'))

    if values.get('final_d_lower') and (float(values.get('final_d_lower')) > 0.0 or float(values.get('final_d_upper')) < 0.1):
        filter_dict['actions.0.final_pose.d'] = {'$gt': float(values.get('final_d_lower')), '$lt': float(values.get('final_d_upper'))}

    data = list(database.episodes.find(
        filter_dict,
        {'_id': 0, 'id': 1, 'actions.reward': 1, 'actions.type': 1, 'actions': {'$slice': -1}}
    ))
    return flask.jsonify(data)


@app.route('/api/stats')
def api_stats():
    database = client[flask.request.values.get('database')]
    suffix = flask.request.values.get('suffix')

    if 'episodes' not in database.collection_names():
        return flask.abort(404)

    query = {}
    if suffix:
        query[f'actions.0.images.{suffix}'] = {'$exists': True}

    total = database.episodes.count(query)
    sum_reward = 0
    for episode in database.episodes.find(query):
        sum_reward += episode['actions'][-1]['reward']

    return flask.jsonify({
        'total': total,
        'reward': sum_reward / total,
    })


@app.route('/api/episode/<episode_id>')
def api_episode(episode_id):
    database = client[flask.request.values.get('database')]
    episode = database.episodes.find_one({'id': episode_id}, {'_id': 0})
    if not episode:
        return flask.abort(404)

    return flask.jsonify(episode)


@app.route('/api/image/<episode_id>')
def api_image(episode_id):
    def send_image(image):
        _, image_encoded = cv2.imencode('.jpg', image)
        return flask.send_file(io.BytesIO(image_encoded), mimetype='image/jpeg')

    def send_empty_image():
        empty = np.zeros((480, 752, 1))
        cv2.putText(empty, '?', (310, 300), cv2.FONT_HERSHEY_SIMPLEX, 6, 100, thickness=6)
        return send_image(empty)

    database_name = flask.request.values.get('database')
    suffix = flask.request.values.get('suffix')

    if flask.request.values.get('pose'):
        action = Action(data=json.loads(flask.request.values.get('pose')))
        image = Loader.get_image(database_name, episode_id, suffix, images=action.images)
    else:
        action, image = Loader.get_action(database_name, episode_id, suffix)

    if not action or suffix not in action.images.keys() or not image:
        return send_empty_image()

    draw_pose(image, action.pose, convert_to_rgb=True, reference_pose=action.images['ed-v']['pose'])

    if int(flask.request.values.get('box', default=0)):
        draw_around_box(image, box=Config.box, draw_lines=True)

    return send_image(image.mat / 255)


@app.route('/api/upload-image', methods=['POST'])
def api_upload_image():
    database = flask.request.values.get('database')
    episode_id = flask.request.values.get('id')
    suffix = flask.request.values.get('suffix', default='v')
    filepath = Loader.get_image_path(database, episode_id, suffix)
    filepath.parent.mkdir(exist_ok=True, parents=True)

    image_data = flask.request.data
    if flask.request.files:
        image_data = flask.request.files['file'].read()

    image_buffer = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(filepath), image)
    return flask.jsonify(success=True)


@app.route('/api/new-episode', methods=['POST'])
def api_new_episode():
    database_name = flask.request.values.get('database')
    database = client[database_name]
    data = json.loads(flask.request.values.get('json'))['episode']
    data['database'] = database_name
    socketio.emit('new-episode', data)
    database.episodes.insert_one(data)
    return flask.jsonify(success=True)


@app.route('/api/new-attempt', methods=['POST'])
def api_new_attempt():
    data = json.loads(flask.request.values.get('json'))
    socketio.emit('new-attempt', data)
    return flask.jsonify(success=True)


@app.route('/api/delete/<episode_id>', methods=['POST'])
def api_delete(episode_id):
    database = client[flask.request.values.get('database')]
    database.episodes.delete_one({'id': episode_id})
    return flask.jsonify(success=True)


@app.route('/')
def index():
    return flask.render_template('overview.html')


@app.route('/live')
def live():
    return flask.render_template('live.html')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080)
