import os

from agents.agent import Agent
from data.loader import Loader


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

agent = Agent()

data = []

for i, (d, e) in enumerate(Loader.yield_episodes('cylinder-cube-mc-1')):
    action, image = Loader.get_action(d, e['id'], 'ed-v')

    if not hasattr(action, 'estimated_reward'):
        continue

    data.append({
        'id': e['id'],
        # 'old': action.estimated_reward,
        'new': agent.reward_for_action([image], action),
        'reward': action.reward
    })

sorted_data = sorted(data, key=lambda k: -abs(k['reward'] - k['new']))

for i, e in enumerate(sorted_data[:20]):
    print(i, e)
