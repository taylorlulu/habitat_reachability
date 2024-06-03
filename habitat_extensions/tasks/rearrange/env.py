import copy
from typing import List, Optional

import magnum as mn
import numpy as np
from habitat import Config, Dataset, RLEnv
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_extensions.utils.visualizations.utils import (
    draw_border,
    observations_to_image,
    put_info_on_image,
)

# isort: off
from .sim import RearrangeSim
from .task import RearrangeTask
from . import actions, sensors
from . import sub_tasks, composite_tasks, composite_sensors
from .sensors import GripperStatus
import matplotlib.pyplot as plt

@baseline_registry.register_env(name="RearrangeRLEnv-v0")
class RearrangeRLEnv(RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        """
        这一步很重要的作用就是对环境进行初始化，即初始化config参数
        Args:
            config:
            dataset:
        """
        self._rl_config = config.RL  # 获取RL的配置参数
        self._core_env_config = config.TASK_CONFIG  # 获取TASK_CONFIG部分
        self._prev_env_obs = None
        # 调用habitat-lab/habitat/core/env.py
        super().__init__(self._core_env_config, dataset=dataset)

    def reset(self):
        """
        对环境进行重置
        Returns:

        """
        observations = super().reset()
        self._prev_env_obs = observations
        # self._prev_env_obs = copy.deepcopy(observations)
        return observations

    def step(self, *args, **kwargs):
        """
        相当于在gym中运行一步
        Args:
            *args:
            **kwargs:

        Returns:

        """
        observations, reward, done, info = super().step(*args, **kwargs)
        self._prev_env_obs = observations
        return observations, reward, done, info

    def get_success(self):
        """
        返回是否成功
        Returns:

        """
        measures = self._env.task.measurements.measures
        # RL成功的指标
        success_measure = self._rl_config.SUCCESS_MEASURE
        if success_measure in measures:
            success = measures[success_measure].get_metric()
        else:
            success = False
        if self._rl_config.get("SUCCESS_ON_STOP", False):
            success = success and self._env.task.should_terminate
        return success

    # 获取奖励部分
    def get_reward(self, observations: Observations):
        """
        返回奖励
        Args:
            observations:

        Returns:

        """
        metrics = self._env.get_metrics()
        # SLACK惩罚
        reward = self._rl_config.SLACK_REWARD

        # 返回奖励
        for reward_measure in self._rl_config.REWARD_MEASURES:
            # print(reward_measure, metrics[reward_measure])
            reward += metrics[reward_measure]

        # 如果成功返回成功奖励
        if self.get_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def get_done(self, observations: Observations):
        """
        返回任务是否完成
        Args:
            observations:

        Returns:

        """
        # NOTE(jigu): episode is over when task.is_episode_active is False,
        # or time limit is passed.
        # 结束当前episode，task.is_episode_active==False or 到达时间限制
        done = self._env.episode_over

        success = self.get_success()
        end_on_success = self._rl_config.get("END_ON_SUCCESS", True)
        if success and end_on_success:
            done = True

        return done

    def get_info(self, observations: Observations):
        """
        返回信息
        Args:
            observations: 观测中的信息

        Returns:

        """
        info = self._env.get_metrics()
        info["is_episode_active"] = self._env.task.is_episode_active
        if self._env.task.is_episode_active:
            # The episode can only be truncated if not active
            assert (
                not self._env.task.is_episode_truncated
            ), self._env._elapsed_steps
            info["is_episode_truncated"] = self._env._past_limit()
        else:
            info["is_episode_truncated"] = self._env.task.is_episode_truncated
        info["elapsed_steps"] = self._env._elapsed_steps
        return info

    def get_reward_range(self):
        """
        返回奖励的范围
        Returns:

        """
        # Have not found its usage, but required to be implemented.
        return (np.finfo(np.float32).min, np.finfo(np.float32).max)

    # 定义了一个参数 mode，并使用类型注解指定其类型为字符串 (str) = "human" 表示如果调用者没有提供 mode 参数的值，将默认使用 "human"
    # 定义了**kwargs：表示接受任意数量的关键字参数
    # 定义了 -> np.ndarray：使用箭头 -> 表示该函数返回的类型是 NumPy 数组
    def render(self, mode: str = "human", **kwargs) -> np.ndarray:
        """
        返回渲染的范围
        Args:
            mode:
            **kwargs:

        Returns:

        """
        if mode == "human":
            obs = self._prev_env_obs.copy()
            info = kwargs.get("info", {})  # 包含关于环境的附加信息
            show_info = kwargs.get("show_info", True)  # 用于控制是否在渲染图像中显示附加信息
            overlay_info = kwargs.get("overlay_info", True)  # 用于控制是否在渲染图像上叠加附加信息
            render_uuid = kwargs.get("render_uuid", "robot_third_rgb")  # 用于确定渲染的特定部分或视角

            # rendered_frame = self._env.sim.render(render_uuid)
            # 在这一部分对环境进行渲染，返回对环境的渲染是在render部分
            rendered_frame = self._env.task.render(render_uuid)

            # rendered_frame = obs[render_uuid]

            # gripper status，获取当前的measurements
            measures = self._env.task.measurements.measures
            gripper_status = measures.get("gripper_status", None)
            if gripper_status is None:
                gripper_status = measures.get("gripper_status_v1", None)
            if gripper_status is not None:
                print("gripper_status="+str(gripper_status))
                if gripper_status.status == GripperStatus.PICK_CORRECT:
                    rendered_frame = draw_border(
                        rendered_frame, (0, 255, 0), alpha=0.5
                    )
                elif gripper_status.status == GripperStatus.PICK_WRONG:
                    rendered_frame = draw_border(
                        rendered_frame, (255, 0, 0), alpha=0.5
                    )
                elif gripper_status.status == GripperStatus.DROP:
                    rendered_frame = draw_border(
                        rendered_frame, (255, 255, 0), alpha=0.5
                    )

            if show_info:
                # 在这部分加入了指标
                rendered_frame = put_info_on_image(
                    rendered_frame, info, overlay=overlay_info
                )
            obs[render_uuid] = rendered_frame

            # 在这部分加入了观测数据
            return observations_to_image(obs)
        else:
            return super().render(mode=mode)
