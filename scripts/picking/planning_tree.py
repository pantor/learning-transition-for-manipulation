import numpy as np


class PlanningTree:
    class Node:
        def __init__(self, action, images, uncertainty_images):
            self.action = action

            self.images = images
            self.uncertainty_images = uncertainty_images

            self.actions = []

        def add_action(self, action, images, uncertainty_images):
            self.actions.append(
                PlanningTree.Node(action, images, uncertainty_images)
            )

    def __init__(self, images, uncertainty_images):
        self.start = PlanningTree.Node(None, images, uncertainty_images)

    def get_node(self, max_leaves, max_depth):
        def check_nodes_recursively(node, depth):
            if depth >= max_depth:
                return None, depth

            if len(node.actions) < max_leaves:
                return node, depth

            for n in node.actions:
                r, d = check_nodes_recursively(n, depth + 1)
                if r is not None:
                    return r, d

            return None, depth

        return check_nodes_recursively(self.start, 0)

    def fill_nodes(self, leaves, depth):
        next_node, next_node_depth = self.get_node(max_leaves=leaves, max_depth=depth)
        while next_node is not None:
            yield next_node, next_node_depth
            next_node, next_node_depth = self.get_node(max_leaves=leaves, max_depth=depth)

    def get_actions_maximize_reward(self, max_depth):
        results = {}  # sum of rewards: action_list

        def check_nodes_recursively(node, depth, action_list):
            previous_sum = sum(map(lambda a: a.estimated_reward, action_list), 0)
            if depth >= max_depth:
                results[previous_sum] = action_list

            if not node.actions:
                results[previous_sum] = action_list

            for n in node.actions:
                check_nodes_recursively(n, depth + 1, action_list + [n.action])

        check_nodes_recursively(self.start, 0, [])

        max_reward_sum = max(results.keys())
        mean_reward_sum = np.mean(list(results.keys()))
        max_action_list = results[max_reward_sum]
        return max_action_list, max_reward_sum, mean_reward_sum

    def get_actions_minimize_steps(self):
        results = {}  # steps: actions_list

        def check_nodes_recursively(node, depth, action_list):
            steps = len(action_list) - 1

            if node.action.type == 'bin_empty':
                results[steps] = action_list

            for n in node.actions:
                check_nodes_recursively(n, depth + 1, action_list + [n.action])

        check_nodes_recursively(self.start, 0, [])

        max_steps = max(results.keys())
        mean_steps = np.mean(list(results.keys()))
        max_action_list = results[max_steps]
        return max_action_list, max_steps, mean_steps


if __name__ == '__main__':
    tree = PlanningTree('i', 'u')

    for node, depth in tree.fill_nodes(leaves=3, depth=2):
        node.add_action('a', 'i1', 'u1')
        print(depth, len(node.actions))

    # actions = tree.get_actions_maximize_reward(max_depth=5)
    # print(actions)