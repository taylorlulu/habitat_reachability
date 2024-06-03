import numpy as np
# from habitat_extensions.utils import load_model
from habitat_extensions.utils.net_utils import CriticNetwork, ActorNetwork, initial_agent, get_q_values

if __name__ == '__main__':
    initial_agent()
    states = np.random.rand(3, 31)
    actions = np.random.rand(3, 5)
    q = get_q_values(states, actions)
    print("q=", q)


