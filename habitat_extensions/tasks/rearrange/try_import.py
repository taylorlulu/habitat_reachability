import numpy as np

from habitat_extensions.utils.load_model import CriticNetwork, ActorNetwork, get_q_values

if __name__ == '__main__':
    states = np.random([3, 31])
    actions = np.random([5, 31])
    q = get_q_values(states, actions)
    print(q)
