from data.loader import Loader

data = []

for i, (d, e) in enumerate(Loader.yield_episodes('cube-1')):
    action = Loader.get_action(d, e['id'])

    if action.reward == 1 and action.final_pose and action.final_pose.d < 0.007:
        print(d, e['id'])
