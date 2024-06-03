import numpy as np
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry

from ..sensors import (
    GripperStatus,
    GripperStatusMeasure,
    GripperToObjectDistance,
    GripperToRestingDistance,
    MyMeasure,
)
from ..task import RearrangeTask


# -------------------------------------------------------------------------- #
# Measure
# -------------------------------------------------------------------------- #
@registry.register_measure
class ReachObjectSuccess(MyMeasure):
    cls_uuid = "reach_obj_success"

    def reset_metric(self, *args, task: EmbodiedTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [GripperToObjectDistance.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: EmbodiedTask, **kwargs):
        measures = task.measurements.measures
        gripper_to_obj_dist = measures[
            GripperToObjectDistance.cls_uuid
        ].get_metric()
        self._metric = gripper_to_obj_dist <= self._config.THRESHOLD


@registry.register_measure
class RearrangePickSuccess(MyMeasure):
    """
    pick中对应的奖励函数
    包括正确抓握以及小于阈值
    """
    cls_uuid = "rearrange_pick_success"

    def reset_metric(self, *args, task: EmbodiedTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [GripperToRestingDistance.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        # 获取当前末端执行器相对于某个固定位置的位置
        # 改位置指的是机器人在不执行特定位置时保持的一种休息或默认状态
        dist = measures[GripperToRestingDistance.cls_uuid].get_metric()
        # 正确抓握且抓握正确的物品
        correct_grasp = (
            self._sim.gripper.grasped_obj_id == task.tgt_obj.object_id
        )
        self._metric = correct_grasp and dist <= self._config.THRESHOLD


# -------------------------------------------------------------------------- #
# Reward
# -------------------------------------------------------------------------- #
@registry.register_measure
class RearrangePickReward(MyMeasure):
    prev_dist_to_goal: float  # 用于存储上一步到达目标的距离
    cls_uuid = "rearrange_pick_reward"

    def reset_metric(self, *args, task: RearrangeTask, **kwargs):
        if not kwargs.get("no_dep", False):
            # 首先检查依赖关系
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    GripperToObjectDistance.cls_uuid,
                    GripperToRestingDistance.cls_uuid,
                    GripperStatusMeasure.cls_uuid,
                ],
            )

        self.prev_dist_to_goal = None
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeTask, **kwargs):
        measures = task.measurements.measures
        # 获取夹爪到达目标位置的距离
        gripper_to_obj_dist = measures[
            GripperToObjectDistance.cls_uuid
        ].get_metric()
        # 获取夹爪到达放置位置的距离
        gripper_to_resting_dist = measures[
            GripperToRestingDistance.cls_uuid
        ].get_metric()
        # 获取当前的状态
        gripper_status = measures[GripperStatusMeasure.cls_uuid].status
        # print("gripper_status", gripper_status)

        reward = 0.0

        # 根据当前的状态判断
        if gripper_status == GripperStatus.PICK_CORRECT:
            reward += self._config.PICK_REWARD  # 如果成功抓取设置抓取奖励
            self.prev_dist_to_goal = gripper_to_resting_dist
        elif gripper_status == GripperStatus.PICK_WRONG:
            reward -= self._config.PICK_PENALTY  # 如果抓取了错误的物品
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_PICK_WRONG", False
            )
        elif gripper_status == GripperStatus.NOT_HOLDING:  # 如果当前没有抓握物品
            if self._config.USE_DIFF:  # 运用差异
                if self.prev_dist_to_goal is None:
                    diff_dist = 0.0
                else:
                    diff_dist = self.prev_dist_to_goal - gripper_to_obj_dist  # 当前位置和目标位置之间的距离，即夹爪靠近了多少
                    diff_dist = round(diff_dist, 3)  # 取三位的四舍五入

                    # Avoid knocking the object away
                    diff_thresh = self._config.get("DIFF_THRESH", -1.0)  # diff的阈值
                    # 此时已经发生碰撞本轮episode结束
                    if diff_thresh > 0 and np.abs(diff_dist) > diff_thresh:
                        diff_dist = 0.0
                        reward -= self._config.DIFF_PENALTY
                        task._is_episode_active = False
                        task._is_episode_truncated = False
                # 距离奖励本轮相较于上一轮的靠近成都
                dist_reward = diff_dist * self._config.DIST_REWARD
            else:
                # 减去与目标位置的差异
                dist_reward = -gripper_to_obj_dist * self._config.DIST_REWARD
            reward += dist_reward
            # 更新距离参数
            self.prev_dist_to_goal = gripper_to_obj_dist
        elif gripper_status == GripperStatus.HOLDING_CORRECT:
            # 抓握正确的物品
            if self._config.USE_DIFF:
                if self.prev_dist_to_goal is None:
                    diff_dist = 0.0
                else:
                    diff_dist = (
                        self.prev_dist_to_goal - gripper_to_resting_dist
                    )
                    diff_dist = round(diff_dist, 3)
                dist_reward = diff_dist * self._config.DIST_REWARD
            else:
                dist_reward = (
                    -gripper_to_resting_dist * self._config.DIST_REWARD
                )
            reward += dist_reward
            self.prev_dist_to_goal = gripper_to_resting_dist
        elif gripper_status == GripperStatus.HOLDING_WRONG:
            # 拿了错误的物品
            raise RuntimeError()
            # pass
        elif gripper_status == GripperStatus.DROP:
            # 物体掉落，提前终止
            reward -= self._config.DROP_PENALTY
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_DROP", False
            )
        # 设置奖励值
        self._metric = reward
