from pathlib import Path
import unittest

import imageio

import picking.path_fix_ros  # pylint: disable=W0611

from data.loader import Loader
from inference.planar import InferencePlanarPose
from picking.param import SelectionMethod
from view.heatmap import Heatmap


TEST_WITH_GPU = False

if TEST_WITH_GPU:
    from agents.agent import Agent
    from agents.agent_predict import Agent as PredictAgent
    from learning.utils.layers import one_hot_gen


class FunctionTestHelper(unittest.TestCase):
    file_path = Path(__file__).parent / 'documents'

    bin_data = {'center': [-0.001, -0.0065, 0.372], 'size': [0.172, 0.281, 0.068]}

    def test_agent(self):
        _, image = Loader.get_action('cylinder-cube-1', '2019-03-26-09-08-16-480', 'ed-v')

        if TEST_WITH_GPU:
            agent = Agent()
            result = agent.infer([image], SelectionMethod.Max)

            self.assertEqual(result.safe, True)
            self.assertEqual(result.method, SelectionMethod.Max)

    def test_agent_predict(self):
        # 2019-03-11-14-56-07-284, 2019-03-14-11-26-17-352, 2019-03-12-16-14-54-658
        _, image = Loader.get_action('cylinder-cube-1', '2019-03-11-14-56-07-284', 'ed-v')

        if TEST_WITH_GPU:
            prediction_model = Loader.get_model('cylinder-cube-1', 'predict-generator-3', custom_objects={'_one_hot': one_hot_gen(4)})
            grasp_model = Loader.get_model('cylinder-cube-1', 'model-6-arch-more-layer', output_layer='prob')
            shift_model = Loader.get_model('shifting', 'model-1', output_layer='prob')

            agent = PredictAgent(prediction_model, grasp_model, shift_model)
            agent.predict_actions([image], SelectionMethod.Top5, N=5, verbose=True)

    def test_heatmap(self):
        _, image = Loader.get_action('cylinder-cube-1', '2019-03-26-09-51-08-827', 'ed-v')

        if TEST_WITH_GPU:
            model = Loader.get_model('cylinder-cube-1', 'model-6-arch-more-layer', output_layer='prob')

            heatmapper = Heatmap(InferencePlanarPose, model, box=self.box)
            heatmap = heatmapper.render(image)
            imageio.imwrite(str(self.file_path / f'gen-heatmap.png'), heatmap)


if __name__ == '__main__':
    unittest.main(verbosity=3)
