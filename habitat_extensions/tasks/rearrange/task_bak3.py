import copy
import gzip
import json
import os
import time
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
from habitat_extensions.utils import art_utils, coll_utils, mn_utils, obj_utils, states_utils
from habitat_extensions.utils.net_utils import initial_agent, get_q_values
from .sim import RearrangeSim
from habitat_sim.utils import viz_utils as vut

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
    cnt = 0

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
        # 这里初始化robot的模型
        initial_agent()

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


    def render(self, mode):
        n_targets = len(self._sim.targets)

        debug_line_render = self._sim.get_debug_line_render()
        rigid_obj_mgr = self._sim.get_rigid_object_manager()

        # 绘制桌面把手的位置
        self._sim.visualize_markers()

        """第一步：绘制当前待抓取物品'024_bowl_:0000'的bbox"""
        tgt_obj, _ = self._sim.get_target(2)
        self._sim.set_object_bb_draw(True, tgt_obj.object_id)
        debug_line_render.push_transform(tgt_obj.root_scene_node.transformation)
        obj_bb = obj_utils.get_aabb(tgt_obj)
        debug_line_render.draw_box(
            obj_bb.min, obj_bb.max, mn.Color4(0.0, 1.0, 0.0, 1.0)
        )
        debug_line_render.pop_transform()

        # 测试expected_theta是否正确
        # expected_theta = np.arctan2(tgt_obj.root_scene_node.translation[2] - self._sim.robot.base_pos[2],
        #                             tgt_obj.root_scene_node.translation[0] - self._sim.robot.base_pos[0])
        # print("当前机器人当前位置为%s, 朝向为%.2f，expected_theta=%.2f"%(self._sim.robot.base_pos, np.rad2deg(self._sim.robot.base_ori),
        #                                                                 np.rad2deg(states_utils.limit_angle(2*np.pi-expected_theta))))

        """第二步：获取物品Object并绘制"""
        obj_list = []
        test_handles = ['024_bowl_:0002', '004_sugar_box_:0003', '009_gelatin_box_:0001']
        for obj_handle in test_handles:
            obj_dict = rigid_obj_mgr.get_objects_by_handle_substring(obj_handle)
            if len(obj_dict) != 0:
                for obj in obj_dict.values():
                    obj_list.append(obj)
                    # 绘制方块，如果采用push_transform则是在transformation的坐标系下进行绘制
                    debug_line_render.push_transform(obj.root_scene_node.transformation)
                    obj_bb = obj_utils.get_aabb(obj)
                    debug_line_render.draw_box(
                        obj_bb.min, obj_bb.max, mn.Color4(0.0, 0.0, 1.0, 1.0)
                    )
                    debug_line_render.pop_transform()

        art_obj_mgr = self._sim.get_articulated_object_manager()
        drawer_handle = "kitchen_counter_:0000"
        drawer_obj = art_obj_mgr.get_object_by_handle(drawer_handle)
        obj_list.append(drawer_obj)

        debug_line_render.push_transform(drawer_obj.root_scene_node.transformation)
        debug_line_render.draw_box(
            states_utils.DRAWER_VIZ_BOX.min, states_utils.DRAWER_VIZ_BOX.max, mn.Color4(0.0, 0.0, 1.0, 1.0)
        )
        debug_line_render.pop_transform()

        """第三步：获取当前对应的states"""
        # state, debug_points = states_utils.get_normal_state(self._sim.robot, tgt_obj, obj_list)

        # self._sim.robot.set_robot_pos(orientation=np.deg2rad(10 * self.cnt))
        # self.cnt += 1

        # 调试部分：测试在机器人坐标系下的点是否正确
        if self.cnt == 0:
            self._sim.robot.set_robot_pos(position=np.array([0.5, 0, 1.8]))

            (states, self.curr_round_loc, round_actions, self.curr_expected_theta,
             self.debug_robot_locations, self.debug_points, self.debug_robot_transformed,
             self.debug_cur_theta, self.debug_rads) = states_utils.get_round_states(self._sim.robot, tgt_obj, obj_list, debug_line_render)

            self.q = get_q_values(states, round_actions)

        """第四步：绘制可达性区域"""
        self._sim.show_reachability(self.q, self.curr_round_loc)

        """调试：当前机器人的位置是否正确"""
        len_ = self.debug_robot_locations.shape[0]

        if self.cnt < len_:
            self._sim.robot.set_robot_pos(position=self.debug_robot_locations[self.cnt],
                                          orientation=self.debug_cur_theta[self.cnt])

            debug_line_render.push_transform(self.debug_robot_transformed[self.cnt])
            debug_line_render.draw_transformed_line(
                mn.Vector3.zero_init(),
                mn.Vector3(self.debug_points[self.cnt, 0], self.debug_points[self.cnt, 1],
                           self.debug_points[self.cnt, 2]),
                mn.Color4(1.0, 0.0, 0.0, 1.0),
                mn.Color4(50.0, 1.0, 10.0, 1.0),
            )
            debug_line_render.pop_transform()
            expected_theta = states_utils.limit_angle(2*np.pi - np.arctan2(tgt_obj.root_scene_node.translation[2] - self._sim.robot.base_pos[2],
                                        tgt_obj.root_scene_node.translation[0] - self._sim.robot.base_pos[0]))
            print("当前索引为%d，目标物体朝向的角度为%s，robot当前真实朝向为%s，计算出的角度是%s，当前目标物体在机器人视角下的角度为%s，目标物体的朝向为%s"%(self.cnt, np.rad2deg(states_utils.limit_angle(self.debug_cur_theta[self.cnt])),
                                                                                        np.rad2deg(states_utils.limit_angle(self._sim.robot.base_ori)),
                                                                                        np.rad2deg(expected_theta),
                                                                                        np.rad2deg(states_utils.limit_angle(self.debug_rads[self.cnt])),
                                                                                        np.rad2deg(states_utils.limit_angle(float(tgt_obj.root_scene_node.rotation.angle())))))
            self.cnt += 1
        else:
            self.cnt = 0
        """调试：当前机器人的位置是否正确"""


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
