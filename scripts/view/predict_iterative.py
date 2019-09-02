from agents.agent_predict import Agent
from data.loader import Loader
from learning.utils.layers import one_hot_gen
from picking.param import SelectionMethod

# few objects (7-8): 2019-03-11-14-56-07-284
# many objects: 2019-07-01-14-03-11-150
# three cubes in a row: 2019-08-23-10-52-33-384

_, image = Loader.get_action('cylinder-cube-1', '2019-08-23-10-52-33-384', 'ed-v')

pred_model = Loader.get_model('cube-1', 'predict-bi-gen-5')

agent = Agent(pred_model)

# agent.plan_actions([image], SelectionMethod.Top5, depth=7, leaves=2, verbose=False)
# agent.plan_actions([image], SelectionMethod.Max, depth=7, leaves=1, verbose=False)
agent.plan_actions([image], SelectionMethod.Top5, depth=5, leaves=3, verbose=True)