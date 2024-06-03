import numpy as np
# Use Mushroom RL library
import torch
import torch.nn as nn
import torch.nn.functional as F
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import *


# 定义了Critic网络
class CriticNetwork(nn.Module):
    # 继承自nn.Module的Python类
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        """

        Args:
            input_shape:表示输入的形状
            output_shape:表示输出的形状
            n_features:表示中间层的特征数
            **kwargs:
        """
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        # input_shape 表示输入的形状，output_shape 表示输出的形状，n_features 表示中间层的特征数
        # 定义了从n_input到n_features再到n_output的三个全连接层
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        # 用于对权重进行 Xavier 初始化，使用 ReLU 激活函数时常用这种初始化方法
        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        """
        定义了模型的前向传播过程，即给定输入计算输出
        Args:
            state:状态
            action:动作

        Returns:

        """
        # 状态和动作在第一个维度上拼接起来
        state_action = torch.cat((state.float(), action.float()), dim=1)
        # 通过两个隐藏层进行传播
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        # 不使用激活函数，直接输出q
        q = self._h3(features2)
        # 用于压缩去除维度
        return torch.squeeze(q)


# 定义了Actor网络
class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        """

        Args:
            input_shape:表示输入的形状
            output_shape:表示输出的形状
            n_features:表示中间层的特征数
            **kwargs:
        """
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a

OBJ_BBOXES_NUM = 5

# 定义了Critic网络
class CriticNetworkGraph(nn.Module):
    # 继承自nn.Module的Python类
    def __init__(self, input_shape, output_shape, n_features, graph_model_critic, **kwargs):
        """

        Args:
            input_shape:表示输入的形状
            output_shape:表示输出的形状
            n_features:表示中间层的特征数
            **kwargs:
        """
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.graph_model_critic = graph_model_critic

        # input_shape 表示输入的形状，output_shape 表示输出的形状，n_features 表示中间层的特征数
        # 定义了从n_input到n_features再到n_output的三个全连接层
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        # 用于对权重进行 Xavier 初始化，使用 ReLU 激活函数时常用这种初始化方法
        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state_goals, state_graphs, action):
        """
        定义了模型的前向传播过程，即给定输入计算输出
        Args:
            state:状态
            action:动作

        Returns:

        """
        total_len = action.shape[0]

        global OBJ_BBOXES_NUM

        # Todo:这部分注意如果修改了节点数目一定要相应的记得修改
        robot_ids = torch.arange(0, total_len * OBJ_BBOXES_NUM, OBJ_BBOXES_NUM, device='cuda:0')

        state_embedding = self.graph_model_critic(state_graphs)
        batch_state_embedding = torch.index_select(state_embedding, 0, robot_ids)

        if batch_state_embedding.shape[0] != action.shape[0]:
            print("state_goals.shape[0]=%d != action.shape[0]=%d" % (state_goals.shape[0], action.shape[0]))
            print("state_graphs=")
            print(state_graphs)
            print("state_embedding.shape=")
            print(state_embedding.shape)
            print("batch_state_embedding.shape=")
            print(batch_state_embedding.shape)
            print("robot_ids=")
            print(robot_ids)
            print("len(robot_ids)=")
            print(len(robot_ids))
            print("OBJ_BBOXES_NUM=")
            print(OBJ_BBOXES_NUM)
            assert 1==2, '出现维度错误'

        #     device = batch_state_embedding.device  # 获取 batch_state_embedding 的设备信息
        #     action = torch.zeros(batch_state_embedding.shape[0], 5, device=device)  # 创建与 batch_state_embedding 相同设备的张量
        #     action[:, 3] = 1
        #     print("========进入了这部分==============")

        # 状态和动作在第一个维度上拼接起来
        state_action = torch.cat((batch_state_embedding.float(), action.float()), dim=1)

        # print("state_action=")
        # print(state_action)

        # 通过两个隐藏层进行传播
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        # 不使用激活函数，直接输出q
        q = self._h3(features2)

        # 用于压缩去除维度
        return torch.squeeze(q)


# 定义了Actor网络
class ActorNetworkGraph(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, graph_model_actor, **kwargs):
        """

        Args:
            input_shape:表示输入的形状
            output_shape:表示输出的形状
            n_features:表示中间层的特征数
            **kwargs:
        """
        super(ActorNetworkGraph, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.robot_ids = []

        self.graph_model_actor = graph_model_actor

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))


    def forward(self, state_goals, state_graphs, a_cont=None):
        # total_len = state_graphs.num_nodes()

        # if a_cont is not None:
        #     print("state_goals=")
        #     print(state_goals.shape)
        #
        #     print("state_graphs=")
        #     print(state_graphs)
        #
        #     print("a_cont=")
        #     print(a_cont.shape)
        #     input("actor_network_graph测试输入")
        global OBJ_BBOXES_NUM

        total_len = state_goals.shape[0]
        # Todo:这部分注意如果修改了节点数目一定要相应的记得修改
        # 否则会报错：CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
        # For debugging consider passing CUDA_LAUNCH_BLOCKING=1
        if len(self.robot_ids) != total_len:
            self.robot_ids = torch.arange(0, total_len * OBJ_BBOXES_NUM, OBJ_BBOXES_NUM, device='cuda:0')

        state_embedding = self.graph_model_actor(state_graphs)
        batch_state_embedding = torch.index_select(state_embedding, 0, self.robot_ids)

        if a_cont is None:
            features1 = F.relu(self._h1(torch.squeeze(batch_state_embedding, 1).float()))
        else:
            batch_concat = torch.cat((torch.squeeze(batch_state_embedding, 1).float(), a_cont), dim=1)
            features1 = F.relu(self._h1(batch_concat.float()))
        features2 = F.relu(self._h2(features1))

        a = self._h3(features2)
        return a

AGENT = None
# def initial_agent(model_path="reachability_model/4_obstacles_ori/agent-1000.msh"):
#     global AGENT
#     AGENT = BHyRL.load(model_path)

# def initial_agent(model_path="reachability_model/4_obstacles_ori/agent-1000.msh"):
#     global AGENT
#     AGENT = BHyRL.load(model_path)

def initial_agent(model_path="reachability_model/4_obstacles_gat/agent-94.msh"):
    global AGENT
    AGENT = BHyRL.load(model_path)


"""获取q函数的部分，在这里对求得的q值结果进行了归一化"""
def get_q_values(states, actions):
    # 在这部分已经进行了归一化
    len_ = states.shape[0]
    # actions = np.zeros((len_, 5))
    actions[:, :3] = np.random.rand(len_, 3) * 0.01
    # actions[:, :3] = 0

    # a, log_prob_next = agent.policy.compute_action_and_log_prob(state)

    print("states.shape=")
    print(states.shape)
    print("actions.shape=")
    print(actions.shape)

    q = AGENT._target_critic_approximator.predict(
        states, actions, prediction='min')

    # 首先进行归一化
    reshaped_q_values = q.reshape((-1, 7))

    # 沿着第二个轴计算平均值
    # max_values = np.max(reshaped_q_values, axis=1)
    average_values = np.mean(reshaped_q_values, axis=1)

    print("debug2:average_values.shape=")
    print(average_values.shape)

    # 将q值转为颜色
    min_val = np.min(average_values)
    max_val = np.max(average_values)
    normalized_q = (average_values - min_val) / (max_val - min_val)

    return normalized_q
