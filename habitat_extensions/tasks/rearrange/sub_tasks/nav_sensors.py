import magnum as mn
import numpy as np
from gym import spaces
from habitat import logger
from habitat.core.registry import registry

from habitat_extensions.tasks.rearrange.sim import RearrangeSim
from habitat_extensions.utils.geo_utils import wrap_angle
from habitat_extensions.utils.states_utils import *

from ..sensors import MyMeasure, MySensor, PositionSensor
from ..task import RearrangeEpisode, RearrangeTask
from ..task_utils import compute_region_goals_v1
from .nav_task import RearrangeNavTask, RearrangeNavTaskV1


# -------------------------------------------------------------------------- #
# Sensor
# -------------------------------------------------------------------------- #
@registry.register_sensor
class BasePositionSensor(PositionSensor):
    cls_uuid = "base_pos"

    def _get_world_position(self, *args, **kwargs):
        return self._sim.robot.base_pos


@registry.register_sensor
class BaseHeadingSensor(MySensor):
    cls_uuid = "base_heading"

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float32)

    def get_observation(self, *args, task: RearrangeTask, **kwargs):
        base_T = self._sim.robot.base_T
        if self.config.get("EPISODIC", True):
            base_T = task.start_base_T.inverted() @ base_T
        heading = base_T.transform_vector(mn.Vector3(1, 0, 0))
        return np.array([heading[0], heading[2]], dtype=np.float32)


@registry.register_sensor
class NavGoalSensor(PositionSensor):
    """Dynamic navigation goal according to whether the object is grasped."""

    cls_uuid = "nav_goal"

    # 获取世界坐标系下的位置
    def _get_world_position(self, *args, task: RearrangeNavTask, **kwargs):
        # if self._sim.gripper.grasped_obj == task.tgt_obj:
        if self._sim.gripper.is_grasped:
            return task.place_goal
        else:
            return task.pick_goal


# -------------------------------------------------------------------------- #
# Measure
# -------------------------------------------------------------------------- #
@registry.register_measure
class GeoDistanceToNavGoal(MyMeasure):
    cls_uuid = "geo_dist_to_nav_goal"

    def reset_metric(self, *args, episode: RearrangeEpisode, **kwargs):
        assert episode._shortest_path_cache is None, episode.episode_id
        return super().reset_metric(*args, episode=episode, **kwargs)

    def update_metric(
        self,
        *args,
        task: RearrangeNavTask,
        episode: RearrangeEpisode,
        **kwargs
    ):
        # 获取机器人当前位置
        start = self._sim.robot.base_pos
        # 获取目标物体当前位置
        goal_pos = task.nav_goal[0]
        # NOTE(jigu): a shortest path cache will be used if episode is passed.
        self._metric = self._sim.geodesic_distance(
            start, goal_pos, episode=episode
        )


@registry.register_measure
class AngDistanceToNavGoal(MyMeasure):
    cls_uuid = "ang_dist_to_nav_goal"

    def update_metric(self, *args, task: RearrangeNavTask, **kwargs):
        start = self._sim.robot.base_ori
        goal_ori = task.nav_goal[1]
        self._metric = np.abs(wrap_angle(goal_ori - start))


@registry.register_measure
class RearrangeNavSuccess(MyMeasure):
    cls_uuid = "rearrange_nav_success"

    def reset_metric(self, *args, task: RearrangeNavTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [GeoDistanceToNavGoal.cls_uuid, AngDistanceToNavGoal.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeNavTask, **kwargs):
        measures = task.measurements.measures
        geo_dist = measures[GeoDistanceToNavGoal.cls_uuid].get_metric()
        ang_dist = measures[AngDistanceToNavGoal.cls_uuid].get_metric()

        # 用于判断导航任务是否完成
        self._metric = (
            geo_dist <= self._config.GEO_THRESHOLD
            and ang_dist <= self._config.ANG_THRESHOLD
        )

        # Deprecation: use "SUCCESS_ON_STOP" in RLEnv
        # 如果设置了 ON_STOP 为 True，则将任务的终止状态也考虑在内，判断导航任务是否成功完成
        if self._config.get("ON_STOP", False):
            self._metric = self._metric and task._should_terminate


@registry.register_measure
class GeoDistanceToNavGoalsV1(MyMeasure):
    # 重置度量值
    def reset_metric(self, *args, episode: RearrangeEpisode, **kwargs):
        assert episode._shortest_path_cache is None, episode.episode_id
        return super().reset_metric(*args, episode=episode, **kwargs)

    # 更新度量值
    def update_metric(
        self,
        *args,
        task: RearrangeNavTaskV1,  # 表示当前的任务
        episode: RearrangeEpisode,  # 表示当前的任务情景
        **kwargs
    ):
        start = self._sim.robot.base_pos  # 获取机器人的基座位置作为起始点
        # NOTE(jigu): a shortest path cache will be used if episode is passed.
        # 计算机器人当前位置到任务中导航目标的地理距离
        self._metric = self._sim.geodesic_distance(
            start, task.nav_goals, episode=episode
        )
        # 计算得到的地理距离小于等于配置中的最小距离，表示已经到达
        if self._metric <= self._config.get("MIN_DIST", -1.0):
            self._metric = 0.0


@registry.register_measure
class AngDistanceToGoal(MyMeasure):
    def update_metric(self, *args, task: RearrangeNavTask, **kwargs):
        # 如果子任务是place，则目标位置为place_gaol
        if task.sub_task_type == "place":
            goal_pos = task.place_goal
        else:  # 否则为pick_goal
            goal_pos = task.pick_goal
        # 即计算二者之差的连线
        offset = goal_pos - np.array(self._sim.robot.base_pos)
        # 计算机器人当前位置与目标位置的偏移量，和计算expected_theta的方式差不多，但为什么是-offset[2]
        goal_ori = np.arctan2(-offset[2], offset[0])
        # 当前朝向和机器人目前朝向之间的差值
        self._metric = np.abs(wrap_angle(goal_ori - self._sim.robot.base_ori))


@registry.register_measure
class AngDistanceToGoalV1(MyMeasure):
    def update_metric(self, *args, task: RearrangeNavTaskV1, **kwargs):
        # 这里的look_at_pos和AngDistanceToGoal类中的goal_pos其实是一样的，只是在RearrangeNavTaskV1中进行了定义
        offset = task.look_at_pos - np.array(self._sim.robot.base_pos)
        goal_ori = np.arctan2(-offset[2], offset[0])
        self._metric = np.abs(wrap_angle(goal_ori - self._sim.robot.base_ori))


# -------------------------------------------------------------------------- #
# Reward
# -------------------------------------------------------------------------- #
@registry.register_measure
class RearrangeNavReward(MyMeasure):
    cls_uuid = "rearrange_nav_reward"

    # 初始化用于计算奖励的变量 prev_geo_dist 和 prev_ang_dist，然后调用 update_metric 方法来更新度量的值
    def reset_metric(self, *args, task: RearrangeNavTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                GeoDistanceToNavGoal.cls_uuid,
                AngDistanceToNavGoal.cls_uuid,
            ],
        )
        self.prev_geo_dist = 0.0
        self.prev_ang_dist = 0.0
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task: RearrangeNavTask, **kwargs):
        measures = task.measurements.measures
        # 获取地理距离和角度距离的值，然后根据配置计算导航任务的奖励
        geo_dist = measures[GeoDistanceToNavGoal.cls_uuid].get_metric()
        ang_dist = measures[AngDistanceToNavGoal.cls_uuid].get_metric()

        reward = 0.0

        if not np.isfinite(geo_dist):
            logger.warning("The geodesic distance is not finite!")
            geo_dist = self.prev_geo_dist

        # 如果地理距离小于或等于角度奖励的阈值，则计算角度奖励，并将其加到总奖励中
        if geo_dist <= self._config.ANG_REWARD_THRESH:
            diff_ang_dist = self.prev_ang_dist - ang_dist
            diff_ang_dist = round(diff_ang_dist, 3)
            ang_dist_reward = diff_ang_dist * self._config.ANG_DIST_REWARD
            reward += ang_dist_reward

        # 更新上一个时间步的角度距离
        self.prev_ang_dist = ang_dist

        # 计算地理奖励，并将其加到总奖励中，然后更新上一个时间步的地理距离
        diff_geo_dist = self.prev_geo_dist - geo_dist
        diff_geo_dist = round(diff_geo_dist, 3)
        geo_dist_reward = diff_geo_dist * self._config.GEO_DIST_REWARD
        reward += geo_dist_reward
        self.prev_geo_dist = geo_dist

        # 将总奖励存储在 _metric 属性中
        self._metric = reward


# ---------------------------------------------------------------------------- #
# For composite rewards
# ---------------------------------------------------------------------------- #
@registry.register_measure
class RearrangeNavRewardV1(MyMeasure):
    cls_uuid = "rearrange_nav_reward"

    def reset_metric(
        self, *args, task: RearrangeTask, episode: RearrangeEpisode, **kwargs
    ):
        # assert episode._shortest_path_cache is None, episode.episode_id
        episode._shortest_path_cache = None

        if self._sim.gripper.is_grasped:
            T = mn.Matrix4.translation(task.place_goal)
        else:
            T = mn.Matrix4.translation(task.pick_goal)

        # 计算较好的区域
        self.nav_goals = compute_region_goals_v1(
            self._sim,
            T,
            region=None,
            radius=0.8,
            height=self._sim.robot.base_pos[1],
        )

        self.prev_is_grasped = self._sim.gripper.is_grasped
        self.prev_geo_dist = None
        self.update_metric(*args, task=task, episode=episode, **kwargs)

    def update_metric(
        self,
        *args,
        task: RearrangeNavTask,
        episode: RearrangeEpisode,
        **kwargs
    ):
        reward = 0.0

        # 计算当前机器人到达区域的距离
        geo_dist = self._sim.geodesic_distance(
            self._sim.robot.base_pos, self.nav_goals, episode=episode
        )

        if geo_dist is None:
            exit(-1)

        if not np.isfinite(geo_dist):
            logger.warning("The geodesic distance is not finite!")
            geo_dist = self.prev_geo_dist

        if self.prev_geo_dist is None:
            diff_geo_dist = 0.0
        else:
            diff_geo_dist = self.prev_geo_dist - geo_dist
            diff_geo_dist = round(diff_geo_dist, 3)

        geo_dist_reward = diff_geo_dist * self._config.GEO_DIST_REWARD

        print("v1:geo_dist_reward=", geo_dist_reward)
        print("v1:geo_dist=", geo_dist)
        print("v1:self.prev_geo_dist=", self.prev_geo_dist)

        reward += geo_dist_reward
        self.prev_geo_dist = geo_dist

        if self._sim.gripper.is_grasped != self.prev_is_grasped:
            reward -= self._config.GRASP_PENALTY
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_GRASP", False
            )

        self._metric = reward


"""仅考虑位置信息不考虑角度信息"""
@registry.register_measure
class RearrangeNavRewardV2(MyMeasure):
    cls_uuid = "rearrange_nav_rewardv2"

    def reset_metric(
        self, *args, task: RearrangeTask, episode: RearrangeEpisode, **kwargs
    ):
        # assert episode._shortest_path_cache is None, episode.episode_id
        episode._shortest_path_cache = None

        if self._sim.gripper.is_grasped:
            T = mn.Matrix4.translation(task.place_goal)
        else:
            T = mn.Matrix4.translation(task.pick_goal)

        # 计算较好的区域
        self.nav_goals = compute_region_goals_v1(
            self._sim,
            T,
            region=None,
            radius=0.8,
            height=self._sim.robot.base_pos[1],
        )

        self.prev_is_grasped = self._sim.gripper.is_grasped
        self.prev_geo_dist = None
        self.update_metric(*args, task=task, episode=episode, **kwargs)

    def update_metric(
        self,
        *args,
        task: RearrangeNavTask,
        episode: RearrangeEpisode,
        **kwargs
    ):
        reward = 0.0

        print("task.nav_goals.shape=%s, task.q=%s"%(str(task.nav_goals.shape), str(task.q.shape)))


        geo_dist = shortest_reachability_area(self._sim.robot.base_pos, task.nav_goals, task.q)

        # 避免geo_dist为None的情况出现
        if not np.isfinite(geo_dist):
            geo_dist = self._sim.geodesic_distance(
                self._sim.robot.base_pos, self.nav_goals, episode=episode
            )

        if not np.isfinite(geo_dist):
            logger.warning("The geodesic distance is not finite!")
            geo_dist = self.prev_geo_dist

        if self.prev_geo_dist is None:
            diff_geo_dist = 0.0
        else:
            diff_geo_dist = self.prev_geo_dist - geo_dist
            diff_geo_dist = round(diff_geo_dist, 3)

        geo_dist_reward = diff_geo_dist * self._config.GEO_DIST_REWARD

        # print("v2:self.prev_geo_dist=%s, geo_dist=%s, geo_dist_reward=%s"%(self.prev_geo_dist, geo_dist, geo_dist_reward))

        reward += geo_dist_reward
        self.prev_geo_dist = geo_dist

        # print("v2:geo_dist_reward=%.2f, geo_dist=%.2f" % (geo_dist_reward, geo_dist))

        if self._sim.gripper.is_grasped != self.prev_is_grasped:
            reward -= self._config.GRASP_PENALTY
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_GRASP", False
            )

        self._metric = reward


"""同时考虑位置信息和角度信息"""
@registry.register_measure
class RearrangeNavRewardV3(MyMeasure):
    cls_uuid = "rearrange_nav_rewardv3"

    def reset_metric(
        self, *args, task: RearrangeTask, episode: RearrangeEpisode, **kwargs
    ):
        # assert episode._shortest_path_cache is None, episode.episode_id
        episode._shortest_path_cache = None

        if self._sim.gripper.is_grasped:
            T = mn.Matrix4.translation(task.place_goal)
        else:
            T = mn.Matrix4.translation(task.pick_goal)

        # 计算较好的区域
        self.nav_goals = compute_region_goals_v1(
            self._sim,
            T,
            region=None,
            radius=0.8,
            height=self._sim.robot.base_pos[1],
        )

        self.prev_is_grasped = self._sim.gripper.is_grasped
        self.prev_geo_dist = None
        self.update_metric(*args, task=task, episode=episode, **kwargs)

    def update_metric(
        self,
        *args,
        task: RearrangeNavTask,
        episode: RearrangeEpisode,
        **kwargs
    ):
        reward = 0.0

        geo_dist = shortest_reachability_area_with_theta(self._sim.robot.base_pos, self._sim.robot.base_ori,
                                                          task.nav_goals, task.cur_tgt_T, task.q.copy())

        # print("task.nav_goals.shape=%s"%(str(task.nav_goals.shape)))

        # 避免geo_dist为None的情况出现
        if not np.isfinite(geo_dist):
            geo_dist = self._sim.geodesic_distance(
                self._sim.robot.base_pos, self.nav_goals, episode=episode
            )

        if not np.isfinite(geo_dist):
            logger.warning("The geodesic distance is not finite!")
            geo_dist = self.prev_geo_dist

        if self.prev_geo_dist is None:
            diff_geo_dist = 0.0
        else:
            diff_geo_dist = self.prev_geo_dist - geo_dist
            diff_geo_dist = round(diff_geo_dist, 3)

        geo_dist_reward = diff_geo_dist * self._config.GEO_DIST_REWARD

        print("v3:geo_dist_reward=", geo_dist_reward)
        print("v3:geo_dist=", geo_dist)
        print("v3:self.prev_geo_dist=", self.prev_geo_dist)

        reward += geo_dist_reward
        self.prev_geo_dist = geo_dist

        """这里可以考虑技能交接"""
        if self._sim.gripper.is_grasped != self.prev_is_grasped:
            reward -= self._config.GRASP_PENALTY
            task._is_episode_active = False
            task._is_episode_truncated = self._config.get(
                "TRUNCATE_GRASP", False
            )

        self._metric = reward