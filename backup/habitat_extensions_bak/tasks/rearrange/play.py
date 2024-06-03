import argparse

from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_extensions.tasks.rearrange.env import RearrangeRLEnv
from habitat_extensions.utils.viewer import OpenCVViewer
from mobile_manipulation.config import Config, get_config, load_config
from mobile_manipulation.utils.common import (
    extract_scalars_from_info,
    get_flat_space_names,
)
from mobile_manipulation.utils.wrappers import HabitatActionWrapperV1
from habitat_extensions.utils.net_utils import CriticNetwork, ActorNetwork, get_agent

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/home/lu/Desktop/embodied_ai/hab-mobile-manipulation')

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

def get_action_from_key(key, action_name):
    if "BaseArmGripperAction" in action_name:
        if key == "w":  # forward
            base_action = [1, 0]
        elif key == "s":  # backward
            base_action = [-1, 0]
        elif key == "a":  # turn left
            base_action = [0, 1]
        elif key == "d":  # turn right
            base_action = [0, -1]
        else:
            base_action = [0, 0]

        # End-effector is controlled
        if key == "i":
            arm_action = [1.0, 0.0, 0.0]
        elif key == "k":
            arm_action = [-1.0, 0.0, 0.0]
        elif key == "j":
            arm_action = [0.0, 1.0, 0.0]
        elif key == "l":
            arm_action = [0.0, -1.0, 0.0]
        elif key == "u":
            arm_action = [0.0, 0.0, 1.0]
        elif key == "o":
            arm_action = [0.0, 0.0, -1.0]
        else:
            arm_action = [0.0, 0.0, 0.0]

        if key == "f":  # grasp
            gripper_action = 1.0
        elif key == "g":  # release
            gripper_action = -1.0
        else:
            gripper_action = 0.0

        return {
            "action": "BaseArmGripperAction",
            "action_args": {
                "base_action": base_action,
                "arm_action": arm_action,
                "gripper_action": gripper_action,
            },
        }
    elif "ArmGripperAction" in action_name:
        if key == "i":
            arm_action = [1.0, 0.0, 0.0]
        elif key == "k":
            arm_action = [-1.0, 0.0, 0.0]
        elif key == "j":
            arm_action = [0.0, 1.0, 0.0]
        elif key == "l":
            arm_action = [0.0, -1.0, 0.0]
        elif key == "u":
            arm_action = [0.0, 0.0, 1.0]
        elif key == "o":
            arm_action = [0.0, 0.0, -1.0]
        else:
            arm_action = [0.0, 0.0, 0.0]

        if key == "f":
            gripper_action = 1.0
        elif key == "g":
            gripper_action = -1.0
        else:
            gripper_action = 0.0

        return {
            "action": "ArmGripperAction",
            "action_args": {
                "arm_action": arm_action,
                "gripper_action": gripper_action,
            },
        }
    elif action_name == "BaseVelAction":
        if key == "w":
            base_action = [1, 0]
        elif key == "s":
            base_action = [-1, 0]
        elif key == "a":
            base_action = [0, 1]
        elif key == "d":
            base_action = [0, -1]
        else:
            base_action = [0, 0]
        return {
            "action": "BaseVelAction",
            "action_args": {
                "velocity": base_action,
            },
        }
    elif action_name == "BaseVelAction2":
        if key == "w":
            base_action = [1, 0]
        elif key == "s":
            base_action = [-1, 0]
        elif key == "a":
            base_action = [0, 1]
        elif key == "d":
            base_action = [0, -1]
        else:
            base_action = [0, 0]
        if key == "z":
            stop = 1
        else:
            stop = 0
        return {
            "action": "BaseVelAction2",
            "action_args": {
                "velocity": base_action,
                "stop": stop,
            },
        }
    elif action_name == "BaseDiscVelAction":
        if key == "w":
            base_action = 17
        elif key == "s":
            base_action = 2
        elif key == "a":
            base_action = 9
        elif key == "d":
            base_action = 5
        elif key == "z":
            base_action = 7
        else:
            base_action = 17
        return {
            "action": "BaseDiscVelAction",
            "action_args": {
                "action": base_action,
            },
        }
    elif action_name == "EmptyAction":
        return {"action": "EmptyAction"}
    else:
        raise NotImplementedError(action_name)


def get_env_config_from_task_config(task_config: Config):
    config = Config()
    config.ENV_NAME = "RearrangeRLEnv-v0"
    config.RL = Config()
    config.RL.ACTION_NAME = task_config.TASK.POSSIBLE_ACTIONS[0]
    config.RL.REWARD_MEASURES = []
    config.RL.SUCCESS_MEASURE = ""
    config.RL.SUCCESS_REWARD = 0.0
    config.RL.SLACK_REWARD = 0.0
    config.TASK_CONFIG = task_config
    config.freeze()
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        dest="config_path",
        type=str,
        default="configs/rearrange/tasks/play.yaml",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument("--action", type=str)
    parser.add_argument(
        "--random-action",
        action="store_true",
        help="whether to sample an action from the action space",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="whether to print verbose information",
    )
    parser.add_argument("--debug-obs", action="store_true")
    args = parser.parse_args()

    # -------------------------------------------------------------------------- #
    # Load config
    # -------------------------------------------------------------------------- #
    config = load_config(args.config_path)
    if "TASK_CONFIG" in config:
        # Reload as RLEnv
        config = get_config(args.config_path, opts=args.opts)
    else:
        config = get_env_config_from_task_config(config)
        if args.opts:
            config.defrost()
            config.merge_from_list(args.opts)
            config.freeze()

    # Override RL.ACTION_NAME
    if args.action:
        config.defrost()
        config.RL.ACTION_NAME = args.action
        config.freeze()

    # -------------------------------------------------------------------------- #
    # Env
    # -------------------------------------------------------------------------- #
    env_cls = baseline_registry.get_env(config.ENV_NAME)
    env: RearrangeRLEnv = env_cls(config)
    env = HabitatActionWrapperV1(env)  # 创建环境
    print(config)
    print("obs_space", env.observation_space)
    print("action_space", env.action_space)
    state_keys = get_flat_space_names(env.observation_space)

    def reset():
        obs = env.reset()
        info = {}
        print("episode_id", env.habitat_env.current_episode.episode_id)
        print("scene_id", env.habitat_env.current_episode.scene_id)
        return obs, info

    env.seed(0)
    obs, info = reset()  # 重置环境
    for k, v in obs.items():
        print(k, v.shape)
    viewer = OpenCVViewer(config.ENV_NAME)

    while True:
        metrics = extract_scalars_from_info(info)
        rendered_frame = env.render(info=metrics, overlay_info=False)
        key = viewer.imshow(rendered_frame)

        if key == "r":  # Press r to reset env
            obs, info = reset()
            continue

        if args.random_action:
            action = env.action_space.sample()
        else:
            # Please refer to this function for keyboard-action mapping
            action = get_action_from_key(key, config.RL.ACTION_NAME)

        # 完成动作
        obs, reward, done, info = env.step(action)
        if args.verbose:
            print("step", env.habitat_env._elapsed_steps)
            print("action", action)
            print("reward", reward)
            print("info", info)
        if args.debug_obs:
            print("obs", {k: v for k, v in obs.items() if k in state_keys})

        if done:
            print("Done")
            obs, info = reset()


if __name__ == "__main__":
    get_agent()
    main()
