import copy
import gzip
import json
import os
from typing import Dict, List, Optional, Tuple

import attr
import habitat_sim
import magnum as mn
import numpy as np
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat_extensions.utils import art_utils, coll_utils, mn_utils, obj_utils

from .sim import RearrangeSim
import matplotlib.pyplot as plt

@attr.s(auto_attribs=True, kw_only=True)
class RearrangeEpisode(Episode):
    r"""Specifies additional objects, targets, markers, and ArticulatedObject states for a particular instance of an object rearrangement task.

    :property ao_states: Lists modified ArticulatedObject states for the scene: {instance_handle -> {link, state}}
    :property rigid_objs: A list of objects to add to the scene, each with: (handle, transform)
    :property targets: Maps an object instance to a new target location for placement in the task. {instance_name -> target_transform}
    :property markers: Indicate points of interest in the scene such as grasp points like handles. {marker name -> (type, (params))}
    """
    ao_states: Dict[str, Dict[int, float]]
    rigid_objs: List[Tuple[str, np.ndarray]]
    targets: Dict[str, np.ndarray]
    markers: List[Dict]
    target_receptacles: List[Tuple[str, int]]
    goal_receptacles: List[Tuple[str, int]]

    # path to the SceneDataset config file
    scene_dataset_config: str = attr.ib(
        default="default", validator=not_none_validator
    )
    # list of paths to search for object config files in addition to the SceneDataset
    additional_obj_config_paths: List[str] = attr.ib(
        default=[], validator=not_none_validator
    )


@registry.register_dataset(name="RearrangeDataset-v0")
class RearrangeDataset(Dataset):
    episodes: List[RearrangeEpisode]

    def __init__(self, config: Optional[Config] = None):
        self.episodes = []
        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        # Keep provided episodes only
        episode_ids = config.get("EPISODE_IDS", [])
        if len(episode_ids) > 0:
            episode_ids = [str(x) for x in episode_ids]
            filter_fn = lambda x: x.episode_id in episode_ids
            self.episodes = list(filter(filter_fn, self.episodes))
            assert len(episode_ids) == len(self.episodes)

        num_episodes = config.get("NUM_EPISODES", -1)
        start = config.get("EPISODE_START", 0)
        end = None if num_episodes < 0 else (start + num_episodes)
        self.episodes = self.episodes[start:end]

    def from_json(self, json_str: str, scenes_dir: Optional[str]) -> None:
        deserialized = json.loads(json_str)
        for episode in deserialized["episodes"]:
            episode = RearrangeEpisode(**episode)
            self.episodes.append(episode)

    @property
    def episode_ids(self):
        return [x.episode_id for x in self.episodes]


"""这个函数很重要在这个函数中设置目标"""
@registry.register_task(name="RearrangeTask-v0")
class RearrangeTask(EmbodiedTask):
    _sim: RearrangeSim
    _is_episode_truncated: bool
    # should be called for force termination only
    _should_terminate: bool

    def overwrite_sim_config(self, sim_config, episode: RearrangeEpisode):
        sim_config.defrost()
        sim_config.SCENE = episode.scene_id
        # sim_config.SCENE_DATASET = episode.scene_dataset_config

        # To use baked lighting
        if self._config.get("USE_BAKED_SCENES", False):
            sim_config.SCENE = episode.scene_id.replace(
                "replica_cad", "replica_cad_baked_lighting"
            ).replace("v3", "Baked")
            sim_config.SCENE_DATASET = "data/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json"

        # Make a copy to avoid in-place modification
        sim_config["EPISODE"] = copy.deepcopy(episode.__dict__)
        # Initialize out of the room, so that it will not affect others
        sim_config["EPISODE"]["start_position"] = [50.0, 0, 50.0]
        sim_config.freeze()
        return sim_config

    def _check_episode_is_active(self, *args, **kwargs) -> bool:
        # NOTE(jigu): Be careful when you use this function to terminate.
        # It is called in task.step() after observations are updated.
        # task.step() is called in env.step() before measurements are updated.
        # return True
        return not self._should_terminate

    @property
    def is_episode_truncated(self):
        return self._is_episode_truncated

    @property
    def should_terminate(self):
        return self._should_terminate

    def seed(self, seed: int) -> None:
        # NOTE(jigu): Env will set the seed for random and np.random
        # when initializing episode iterator.
        self.np_random = np.random.RandomState(seed)

    def reset(self, episode: RearrangeEpisode):
        self._sim.reset()

        # Clear and cache
        self.tgt_idx = None
        self.tgt_obj, self.tgt_T = None, None
        self.tgt_receptacle_info = None
        self.start_ee_pos = None  # for ee-space controller
        # [['frl_apartment_table_02_:0000', None], ['frl_apartment_sofa_:0000', None], ['kitchen_counter_:0000', 0], ['kitchen_counter_:0000', 0], ['frl_apartment_chair_01_:0001', None]]
        self._target_receptacles = episode.target_receptacles
        # [['kitchen_counter_:0000', 0], ['frl_apartment_tvstand_:0000', None], ['frl_apartment_tvstand_:0000', None], ['frl_apartment_sofa_:0000', None], ['kitchen_counter_:0000', 0]]
        self._goal_receptacles = episode.goal_receptacles

        # print("========Debug:self._target_receptacles=%s============"%(self._target_receptacles))
        # print("========Debug:self._goal_receptacles=%s============" % (self._goal_receptacles))

        self.initialize(episode)
        self._reset_stats()

        for action_instance in self.actions.values():
            action_instance.reset(episode=episode, task=self)
        return self._get_observations(episode)

    def initialize(self, episode: RearrangeEpisode):
        # 初始化末端执行器的位置
        self._initialize_ee_pos()
        # 设置机器人的初始位置
        start_pos = self._sim.pathfinder.get_random_navigable_point()
        self._sim.robot.base_pos = start_pos
        # 设置机器人的初始朝向
        self._sim.robot.base_ori = self.np_random.uniform(0, 2 * np.pi)
        self._sim.internal_step_by_time(0.1)

    """获取末端执行器的初始位置"""
    def _get_start_ee_pos(self):
        # NOTE(jigu): defined in pybullet link frame
        # 获取模锻执行器的位置，其三维的位置
        start_ee_pos = np.array(
            self._config.get("START_EE_POS", [0.5, 0.0, 1.0]),
            dtype=np.float32,
        )

        # The noise can not be too large (e.g. 0.05)
        # 设置末端执行器的噪声
        ee_noise = self._config.get("EE_NOISE", 0.025)
        if ee_noise > 0:
            noise = self.np_random.normal(0, ee_noise, [3])
            noise = np.clip(noise, -ee_noise * 2, ee_noise * 2)
            start_ee_pos = start_ee_pos + noise

        return start_ee_pos

    """初始化末端执行器"""
    def _initialize_ee_pos(self, start_ee_pos=None):
        """Initialize end-effector position.初始化夹爪End Effector的位置"""
        # 这是一个三维坐标的位置
        if start_ee_pos is None:
            start_ee_pos = self._get_start_ee_pos()

        # print("start_ee_pos", start_ee_pos)
        self.start_ee_pos = start_ee_pos
        # 重置机械臂
        self._sim.robot.reset_arm()
        self._sim.sync_pyb_robot()
        # 逆向运动学根据机械臂末端的三维位置获取当前关节角度
        arm_tgt_qpos = self._sim.pyb_robot.IK(self.start_ee_pos, max_iters=100)
        # err = self._sim.pyb_robot.compute_IK_error(start_ee_pos, arm_tgt_qpos)
        # 这两个arm_joint_pos和arm_motor_pos之间存在什么区别
        self._sim.robot.arm_joint_pos = arm_tgt_qpos
        self._sim.robot.arm_motor_pos = arm_tgt_qpos

    def _reset_stats(self):
        # NOTE(jigu): _is_episode_active is on-the-fly set in super().step()
        self._is_episode_active = True  # 当前轮被激活
        self._is_episode_truncated = False
        self._should_terminate = False

        # Record the initial robot pose for episodic sensors
        self.start_base_T = self._sim.robot.base_T  # 获取当前机器人的robot
        # habitat frame，resting_position是用来干什么的，可能是机械臂的某个停留的位置
        self.resting_position = np.array(
            self._config.get("RESTING_POSITION", [0.5, 1.0, 0.0]),
            dtype=np.float32,
        )

    def _get_observations(self, episode):
        observations = self._sim.get_observations()
        # 更新当前机器人的观测
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations, episode=episode, task=self
            )
        )
        return observations

    def transformation2robot_frame(self, bbox, obj_tf, robot_tf):
        bbox_ = np.zeros([2, 3])
        bbox_[0, :] = np.array(bbox.min)
        bbox_[1, :] = np.array(bbox.max)
        obj_tf = np.array(obj_tf)
        robot_tf = np.array(robot_tf)

        # 将bbox都转换为点
        points = np.ones([8, 3])
        points[0, :] = np.array([bbox_[0, 0], bbox_[0, 1], bbox_[0, 2]])
        points[1, :] = np.array([bbox_[0, 0], bbox_[1, 1], bbox_[0, 2]])
        points[2, :] = np.array([bbox_[1, 0], bbox_[0, 1], bbox_[0, 2]])
        points[3, :] = np.array([bbox_[1, 0], bbox_[1, 1], bbox_[0, 2]])
        points[4, :] = np.array([bbox_[0, 0], bbox_[0, 1], bbox_[1, 2]])
        points[5, :] = np.array([bbox_[0, 0], bbox_[1, 1], bbox_[1, 2]])
        points[6, :] = np.array([bbox_[1, 0], bbox_[0, 1], bbox_[1, 2]])
        points[7, :] = np.array([bbox_[1, 0], bbox_[1, 1], bbox_[1, 2]])

        # 第一步：将bbox的坐标扩展为齐次坐标
        bbox_homogeneous = np.hstack((bbox_[0], 1)), np.hstack((bbox_[1], 1))

        points_global = np.ones([8, 3])
        # 第二步：将 bbox 的坐标从物品自身坐标系转换到全局坐标系，这一步验证过是正确的
        for i in range(points.shape[0]):
            points_global[i, :] = np.dot(obj_tf, np.hstack((points[i, :], 1))).flatten()[:3]

        # 第三步：计算机器人坐标系到全局坐标系的逆变换矩阵
        inv_base_tf = np.linalg.inv(robot_tf)

        # 第四步：将 bbox 的坐标从全局坐标系转换到机器人的局部坐标系
        points_local = np.ones([8, 3])
        points_global2 = np.ones([8, 3])
        for i in range(points.shape[0]):
            points_local[i, :] = np.dot(inv_base_tf, np.hstack((points_global[i, :], 1))).flatten()[:3]

        points_global2 = np.ones([8, 3])
        for i in range(points.shape[0]):
            points_global2[i, :] = np.dot(robot_tf, np.hstack((points_local[i, :], 1))).flatten()[:3]

        # 在局部坐标系下
        min_coords_fridge = np.min(points_local, axis=0)
        max_coords_fridge = np.max(points_local, axis=0)

        print("局部坐标系下:bbox_local_min=%s, bbox_local_max=%s"%(min_coords_fridge, max_coords_fridge))
        theta = np.arctan2(max_coords_fridge[2] - min_coords_fridge[2], max_coords_fridge[0] - min_coords_fridge[0])
        print("在全局坐标系下的坐标为points=%s， \n 转换后的坐标为%s, \n 在局部坐标系下为%s \n 角度为%s"%(points_global, points_global2, points_local, np.rad2deg(theta)))

        return mn.Range3D(mn.Vector3(min_coords_fridge[0], min_coords_fridge[1], min_coords_fridge[2]),
                                mn.Vector3(max_coords_fridge[0], max_coords_fridge[1], max_coords_fridge[2])), points_local

    def render(self, mode):
        n_targets = len(self._sim.targets)

        debug_line_render = self._sim.get_debug_line_render()

        # 绘制bounding box的位置
        for i in range(n_targets):
            tgt_obj, _ = self._sim.get_target(i)
            self._sim.set_object_bb_draw(True, tgt_obj.object_id)
            self._sim.visualize_target(i)

            # 绘制方块，如果采用push_transform则是在transformation的坐标系下进行绘制
            debug_line_render.push_transform(tgt_obj.root_scene_node.transformation)
            obj_bb = obj_utils.get_aabb(tgt_obj)
            debug_line_render.draw_box(
                obj_bb.min, obj_bb.max, mn.Color4(0.0, 1.0, 0.0, 1.0)
            )
            debug_line_render.pop_transform()


        # 绘制桌面把手的位置
        self._sim.visualize_markers()

        """这是一部分调试是否能够绘制圆圈的代码，注意这个代码必须在每一帧更新而且必须是作为第一位的传感器在SENSORS中只会显示在第一个"""
        agent_translation = mn.Vector3(self._sim.robot.base_T.translation)

        # 绘制圆圈
        debug_line_render.set_line_width(2.0)
        agent_translation[1] = 0.1
        debug_line_render.draw_circle(agent_translation, 0.5, mn.Color4(1.0, 0.0, 0.0, 1.0))

        # 冰箱和抽屉在Replica中不提供bbox，可能是因为是.glb格式的文件，所以只能采用试的方法得到结果
        fridge_viz_box = mn.Range3D(mn.Vector3(-0.38, -0.95, -0.35), mn.Vector3(0.38, 0.95, 0.35))
        drawer_viz_box = mn.Range3D(mn.Vector3(-0.48, 0.0, -1.5), mn.Vector3(0.15, 0.82, 1.5))

        # 首先获取对应物品的绝对坐标值
        art_obj_mgr = self._sim.get_articulated_object_manager()
        fridge_handle = "fridge_:0000"
        fridge_obj = art_obj_mgr.get_object_by_handle(fridge_handle)
        drawer_handle = "kitchen_counter_:0000"
        drawer_obj = art_obj_mgr.get_object_by_handle(drawer_handle)

        robot_tf = mn.Matrix4(self._sim.robot.base_T)
        debug_line_render.push_transform(robot_tf)
        # fridge_viz_box = self.transformation2robot_frame(fridge_viz_box, fridge_obj.root_scene_node.transformation, robot_tf)
        drawer_viz_box, points = self.transformation2robot_frame(drawer_viz_box, drawer_obj.root_scene_node.transformation, robot_tf)

        # 对于机器人而言坐标同样是x, y, z，其中x是对于机器人前后，z是机器人上下，y是机器人左右
        drawer_viz_box3 = mn.Range3D(mn.Vector3(1.0, 0, -1.0), mn.Vector3(2.0, 3.0, 1.5))


        # print("drawer_bb=%s \n drawer_bb2=%s \n 当前朝向为base_ori=%s"%(drawer_bb, drawer_bb2, np.rad2deg(self._sim.robot.base_ori)))

        # debug_line_render.draw_box(
        #     fridge_bb.min, fridge_bb.max, mn.Color4(0.0, 0.0, 1.0, 1.0)
        # )
        # debug_line_render.draw_box(
        #     drawer_viz_box.min, drawer_viz_box.max, mn.Color4(0.0, 0.0, 1.0, 1.0)
        # )

        # for i in range(7):
            # debug_line_render.draw_transformed_line(
            #         mn.Vector3.zero_init(),
            #         mn.Vector3(points[i, 0], points[i, 1], points[i, 2]),
            #         mn.Color4(1.0, 0.0, 0.0, 1.0),
            #         mn.Color4(50.0, 1.0, 10.0, 1.0),
            #     )

            # debug_line_render.draw_transformed_line(
            #     mn.Vector3(points[i, 0], points[i, 1], points[i, 2]),
            #     mn.Vector3(points[i+1, 0], points[i+1, 1], points[i+1, 2]),
            #     mn.Color4(1.0, 0.0, 0.0, 1.0),
            #     mn.Color4(50.0, 1.0, 10.0, 1.0),
            # )

        # debug_line_render.draw_transformed_line(
        #     mn.Vector3(points[0, 0], points[0, 1], points[0, 2]),
        #     mn.Vector3(points[7, 0], points[7, 1], points[7, 2]),
        #     mn.Color4(1.0, 0.0, 0.0, 1.0),
        #     mn.Color4(50.0, 1.0, 10.0, 1.0),
        # )

        # debug_line_render.draw_transformed_line(
        #     drawer_viz_box.min,
        #     drawer_viz_box.max,
        #     mn.Color4(1.0, 0.0, 0.0, 1.0),
        #     mn.Color4(50.0, 1.0, 10.0, 1.0),
        # )

        debug_line_render.draw_box(
            mn.Vector3(points[0, 0], points[0, 1], points[0, 2]), mn.Vector3(points[7, 0], points[7, 1], points[7, 2]), mn.Color4(0.0, 0.0, 1.0, 1.0)
        )

        # debug_line_render.draw_transformed_line(
        #     mn.Vector3.zero_init(),
        #     mn.Vector3(1.0, 5.0, -0.5),
        #     mn.Color4(1.0, 0.0, 0.0, 1.0),
        #     mn.Color4(1.0, 1.0, 1.0, 1.0),
        # )

        # debug_line_render.draw_transformed_line(
        #     mn.Vector3.zero_init(),
        #     drawer_viz_box.min,
        #     mn.Color4(1.0, 0.0, 0.0, 1.0),
        #     mn.Color4(1.0, 1.0, 1.0, 1.0),
        # )
        #
        # debug_line_render.draw_transformed_line(
        #     mn.Vector3.zero_init(),
        #     drawer_viz_box.max,
        #     mn.Color4(1.0, 0.0, 0.0, 1.0),
        #     mn.Color4(1.0, 1.0, 1.0, 1.0),
        # )
        debug_line_render.pop_transform()

        """
        # 进行坐标的转换
        inv_base_tf = self._sim.robot.base_T.inverted()

        fridge_in_robot_frame = inv_base_tf.__matmul__(fridge_obj.root_scene_node.transformation)
        drawer_in_robot_frame = inv_base_tf.__matmul__(drawer_obj.root_scene_node.transformation)
        # 转换为齐次坐标
        fridge_T = mn.Matrix2x4(mn.Vector4(fridge_viz_box.min, 1.0), mn.Vector4(fridge_viz_box.max, 1.0))
        drawer_T = mn.Matrix2x4(mn.Vector4(drawer_viz_box.min, 1.0), mn.Vector4(drawer_viz_box.max, 1.0))

        bbox_in_fridge_frame = np.dot(np.array(fridge_in_robot_frame), np.array(fridge_T)).T
        bbox_in_drawer_frame = np.dot(np.array(drawer_in_robot_frame), np.array(drawer_T)).T

        min_coords_fridge = np.min(bbox_in_fridge_frame, axis=0)
        max_coords_fridge = np.max(bbox_in_fridge_frame, axis=0)

        min_coords_drawer = np.min(bbox_in_drawer_frame, axis=0)
        max_coords_drawer = np.max(bbox_in_drawer_frame, axis=0)

        fridge_viz_box = mn.Range3D(mn.Vector3(min_coords_fridge[0], min_coords_fridge[1], min_coords_fridge[2]),
                                    mn.Vector3(max_coords_fridge[0], max_coords_fridge[1], max_coords_fridge[2]))
        drawer_viz_box = mn.Range3D(mn.Vector3(min_coords_drawer[0], min_coords_drawer[1], min_coords_drawer[2]),
                                    mn.Vector3(max_coords_drawer[0], max_coords_drawer[1], max_coords_drawer[2]))

        debug_line_render.draw_box(
            fridge_viz_box.min, fridge_viz_box.max, mn.Color4(0.0, 0.0, 1.0, 1.0)
        )
        debug_line_render.draw_box(
            drawer_viz_box.min, drawer_viz_box.max, mn.Color4(0.0, 0.0, 1.0, 1.0)
        )

        debug_line_render.pop_transform()
        """

        """尝试直接画圆是可以实现的
        # 在冰箱之间画框线
        art_obj_mgr = self._sim.get_articulated_object_manager()
        fridge_handle = "fridge_:0000"
        fridge_obj = art_obj_mgr.get_object_by_handle(fridge_handle)
        debug_line_render.push_transform(fridge_obj.root_scene_node.transformation)
        debug_line_render.draw_box(
            fridge_viz_box.min, fridge_viz_box.max, mn.Color4(0.0, 0.0, 1.0, 1.0)
        )
        debug_line_render.pop_transform()

        # 在厨房柜台之间画框线
        drawer_handle = "kitchen_counter_:0000"
        drawer_obj = art_obj_mgr.get_object_by_handle(drawer_handle)
        debug_line_render.push_transform(drawer_obj.root_scene_node.transformation)
        debug_line_render.draw_box(
            drawer_viz_box.min, drawer_viz_box.max, mn.Color4(0.0, 0.0, 1.0, 1.0)
        )
        debug_line_render.pop_transform()
        """
        """这是一部分调试是否能够绘制圆圈的代码"""

        ret = self._sim.render(mode)

        # for i in range(n_targets):
        #     tgt_obj, _ = self._sim.get_target(i)
        #     self._sim.set_object_bb_draw(False, tgt_obj.object_id)

        return ret

    # -------------------------------------------------------------------------- #
    # Targets
    # -------------------------------------------------------------------------- #
    tgt_idx: int
    tgt_obj: habitat_sim.physics.ManagedBulletRigidObject
    tgt_T: mn.Matrix4
    tgt_receptacle_info: Tuple[str, int]

    def _set_target(self, index):
        self.tgt_idx = index
        self.tgt_obj, self.tgt_T = self._sim.get_target(self.tgt_idx)
        self.tgt_receptacle_info = self._target_receptacles[self.tgt_idx]

    def _has_target_in_drawer(self):
        receptacle_handle, receptacle_link_id = self.tgt_receptacle_info
        if receptacle_handle is None:  # for baked scenes
            return False
        if "kitchen_counter" in receptacle_handle and receptacle_link_id != 0:
            return True
        else:
            return False

    def _has_target_in_fridge(self):
        receptacle_handle, _ = self.tgt_receptacle_info
        if receptacle_handle is None:  # for baked scenes
            return False
        if "frige" in receptacle_handle or "fridge" in receptacle_handle:
            return True
        else:
            return False

    def _has_target_in_container(self):
        return self._has_target_in_drawer() or self._has_target_in_fridge()
