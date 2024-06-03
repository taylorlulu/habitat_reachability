#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import magnum as mn
import numpy as np
from gym import spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import (
    HabitatSim,
    HabitatSimSemanticSensor,
)
from habitat_sim.physics import (
    JointMotorSettings,
    ManagedBulletArticulatedObject,
    ManagedBulletRigidObject,
    MotionType,
)

from habitat_extensions.robots.fetch_robot import FetchRobot
from habitat_extensions.robots.grippers import MagicGripper
from habitat_extensions.robots.marker import Marker
from habitat_extensions.robots.pybullet_utils import PybulletRobot
from habitat_extensions.utils import art_utils, coll_utils, mn_utils, obj_utils
from habitat_extensions.utils.sim_utils import (
    get_navmesh_settings,
    get_object_handle_by_id,
)


@registry.register_sensor(name="HabitatSimSemanticSensor")
class MyHabitatSimSemanticSensor(HabitatSimSemanticSensor):
    """Overwrite the original one to use uint8 instead of uint32."""

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            low=np.iinfo(np.uint8).min,
            high=np.iinfo(np.uint8).max,
            shape=(self.config.HEIGHT, self.config.WIDTH),
            dtype=np.uint8,
        )

    def get_observation(self, sim_obs):
        obs = super().get_observation(sim_obs)
        return obs.astype(np.uint8, copy=False)


"""
该类继承自HabitatSim并在一定程度上进行了封装
"""
@registry.register_simulator(name="RearrangeSim-v0")
class RearrangeSim(HabitatSim):
    robot: FetchRobot

    RIGID_OBJECT_DIR = "data/objects/ycb"
    PRIMITIVE_DIR = "habitat_extensions/assets/objects/primitives"

    def __init__(self, config: Config):
        super().__init__(config)

        # NOTE(jigu): The first episode is used to initialized the simulator
        # When `habitat.Env` is initialized.
        # NOTE(jigu): DO NOT set `_current_scene` to None.
        self._prev_scene_id = None
        self._prev_scene_dataset = config.SCENE_DATASET
        self._initial_state = None

        self._initialize_templates()
        self.navmesh_settings = get_navmesh_settings(self._get_agent_config())

        # objects
        self.rigid_objs: Dict[str, ManagedBulletRigidObject] = OrderedDict()
        self.art_objs: Dict[
            str, ManagedBulletArticulatedObject
        ] = OrderedDict()
        self.viz_objs: Dict[str, ManagedBulletRigidObject] = OrderedDict()
        self.markers: Dict[str, Marker] = OrderedDict()

        # robot
        if config.FETCH_ROBOT.TYPE == "hab_fetch":
            self.robot = FetchRobot(self)
        else:
            raise NotImplementedError(config.FETCH_ROBOT.TYPE)

        self.robot.update_params(config.FETCH_ROBOT.PARAMS)

        # NOTE(jigu): hardcode (arm-only) pyrobot
        ARM_URDF = "habitat_extensions/assets/robots/hab_fetch/robots/hab_fetch_arm.urdf"
        self.pyb_robot = PybulletRobot(
            ARM_URDF, joint_indices=[0, 1, 2, 3, 4, 5, 6], ee_link_idx=7
        )

        # gripper
        self.gripper = MagicGripper(self, self.robot)

    def _initialize_templates(self):
        obj_attr_mgr = self.get_object_template_manager()
        obj_attr_mgr.load_configs(self.RIGID_OBJECT_DIR)
        # primitives for visualization
        obj_attr_mgr.load_configs(self.PRIMITIVE_DIR)
        # print(obj_attr_mgr.get_template_handles())

    @property
    def timestep(self):
        return self.habitat_config.CONTROL_FREQ / self.habitat_config.SIM_FREQ

    @property
    def verbose(self):
        return self.habitat_config.get("VERBOSE", False)

    def reconfigure(self, habitat_config: Config):
        """Called before sim.reset() in `habitat.Env`."""
        # NOTE(jigu): release before super().reconfigure()
        # otherwise, there might be memory leak for constraint.
        # This extra release might also change results, but the reason is unknown.
        self.gripper.desnap(True)

        # NOTE(jigu): DO NOT use self._current_scene to judge
        is_same_scene = habitat_config.SCENE == self._prev_scene_id
        if self.verbose:
            print("is_same_scene", is_same_scene)

        is_same_scene_dataset = (
            habitat_config.SCENE_DATASET == self._prev_scene_dataset
        )

        # The simulator backend will be reconfigured.
        # Assets are invalid after a new scene is configured.
        # Note that ReplicaCAD articulated objects are managed by the backend.
        super().reconfigure(habitat_config)
        self._prev_scene_id = habitat_config.SCENE
        self._prev_scene_dataset = habitat_config.SCENE_DATASET

        if not is_same_scene:
            self.art_objs = OrderedDict()
            self.rigid_objs = OrderedDict()
            self.robot.sim_obj = None
            self._initial_state = None

        if not is_same_scene_dataset:
            self._initialize_templates()

        # Called before new assets are added
        self.gripper.reconfigure()
        if not is_same_scene:
            self.robot.reconfigure()
            self.robot.set_semantic_ids(100)
        elif self._initial_state is not None:
            self.robot.set_state(self._initial_state["robot_state"])

        if not is_same_scene:
            self._add_articulated_objects()
            self._initialize_articulated_objects()
        elif self._initial_state is not None:
            art_objs_qpos = self._initial_state["art_objs_qpos"]
            for handle, qpos in art_objs_qpos.items():
                art_obj = self.art_objs[handle]
                art_obj.clear_joint_states()  # joint positions are also zeroed.
                art_obj.joint_positions = qpos

        self._remove_rigid_objects()
        self._add_rigid_objects()
        self._add_markers()
        self._add_targets()

        assert len(self.viz_objs) == 0, self.viz_objs
        self.viz_objs = OrderedDict()

        if self.habitat_config.get("AUTO_SLEEP", False):
            self.sleep_all_objects()

        if not is_same_scene:
            self._recompute_navmesh()

        # Cache initial state
        self._initial_state = self.get_state()

    def _add_rigid_objects(self):
        rigid_obj_mgr = self.get_rigid_object_manager()
        episode = self.habitat_config.EPISODE

        for obj_info in episode["rigid_objs"]:
            template_handle = osp.join(self.RIGID_OBJECT_DIR, obj_info[0])
            obj = rigid_obj_mgr.add_object_by_template_handle(template_handle)
            T = mn.Matrix4(np.array(obj_info[1]))
            # obj.transformation = T
            obj.transformation = mn_utils.orthogonalize(T)
            obj.motion_type = MotionType.DYNAMIC
            self.rigid_objs[obj.handle] = obj
            if self.verbose:
                print("Add a rigid body", obj.handle, obj.object_id)

    def _remove_rigid_objects(self):
        rigid_obj_mgr = self.get_rigid_object_manager()
        for handle, obj in self.rigid_objs.items():
            assert obj.is_alive, handle
            if self.verbose:
                print(
                    "Remove a rigid object",
                    obj.handle,
                    obj.object_id,
                    obj.is_alive,
                )
            rigid_obj_mgr.remove_object_by_id(obj.object_id)
        self.rigid_objs = OrderedDict()

    def _add_articulated_objects(self):
        art_obj_mgr = self.get_articulated_object_manager()
        for handle in art_obj_mgr.get_object_handles():
            if handle == self.robot.sim_obj.handle:  # ignore robot
                continue
            self.art_objs[handle] = art_obj_mgr.get_object_by_handle(handle)

    def _remove_articulated_objects(self):
        art_obj_mgr = self.get_articulated_object_manager()
        for art_obj in self.art_objs.values():
            assert art_obj.is_alive
            if self.verbose:
                print(
                    "Remove an articulated object",
                    art_obj.handle,
                    art_obj.object_id,
                    art_obj.is_alive,
                )
            art_obj_mgr.remove_object_by_id(art_obj.object_id)
        self.art_objs = OrderedDict()

    def _initialize_articulated_objects(self):
        # NOTE(jigu): params from p-viz-plan/orp/sim.py
        for handle in self.art_objs:
            art_obj = self.art_objs[handle]
            for motor_id, link_id in art_obj.existing_joint_motor_ids.items():
                art_utils.update_motor(
                    art_obj, motor_id, velocity_gain=0.3, max_impulse=1.0
                )

    def _set_articulated_objects_from_episode(self):
        episode = self.habitat_config.EPISODE
        art_obj_mgr = self.get_articulated_object_manager()

        for handle, joint_states in episode["ao_states"].items():
            # print(handle, joint_states)
            art_obj = art_obj_mgr.get_object_by_handle(handle)
            qpos = art_obj.joint_positions
            for link_id, joint_state in joint_states.items():
                pos_offset = art_obj.get_link_joint_pos_offset(int(link_id))
                qpos[pos_offset] = joint_state
            art_obj.joint_positions = qpos

    def print_articulated_objects(self):
        art_obj_mgr = self.get_articulated_object_manager()
        for handle in art_obj_mgr.get_object_handles():
            art_obj = art_obj_mgr.get_object_by_handle(handle)
            print(handle, art_obj, art_obj.object_id)

    # 增加
    def _add_markers(self):
        self.markers = OrderedDict()
        art_obj_mgr = self.get_articulated_object_manager()

        # NOTE(jigu): The official one does not include all markers
        # episode = self.habitat_config.EPISODE
        # for marker_info in episode["markers"]:
        #     name = marker_info["name"]
        #     params = marker_info["params"]
        #     art_obj = art_obj_mgr.get_object_by_handle(params["object"])
        #     link_id = art_utils.get_link_id_by_name(art_obj, params["link"])
        #     marker = Marker(name, art_obj, link_id, params["offset"])
        #     self.markers[name] = marker

        # 这个是冰箱
        fridge_handle = "fridge_:0000"
        art_obj = art_obj_mgr.get_object_by_handle(fridge_handle)
        link_id = art_utils.get_link_id_by_name(art_obj, "top_door")
        marker = Marker(
            "fridge_push_point", art_obj, link_id, offset=[0.10, -0.62, 0.2]
        )
        self.markers[marker.uuid] = marker

        # 这个是厨房的柜台
        drawer_handle = "kitchen_counter_:0000"
        art_obj = art_obj_mgr.get_object_by_handle(drawer_handle)
        drawer_link_names = [
            "drawer1_bottom",
            "drawer1_top",
            "drawer2_bottom",
            "drawer2_middle",
            "drawer2_top",
            "drawer4",
            "drawer3",
        ]
        for idx, link_name in enumerate(drawer_link_names):
            link_id = art_utils.get_link_id_by_name(art_obj, link_name)
            marker_name = "cab_push_point_{}".format(idx + 1)
            marker = Marker(marker_name, art_obj, link_id, offset=[0.3, 0, 0])
            self.markers[marker.uuid] = marker

    def _add_targets(self):
        """
        增加环境中目标物体的部分
        Returns:

        """
        self.targets = OrderedDict()
        # 在这部分中获取环境信息
        episode = self.habitat_config.EPISODE
        # handles = sorted(episode["targets"].keys())
        # NOTE(jigu): The order of targets is used in `target_receptacles` and `goal_receptacles`

        # target_receptacles需要从goal_receptacles放置到target_receptacles，所以它们的数目是相等的
        handles = list(episode["targets"].keys())
        for handle in handles:
            T = episode["targets"][handle]
            self.targets[handle] = mn_utils.orthogonalize(T)

    def _recompute_navmesh(self):
        # navmesh_path = self._current_scene + ".navmesh"
        # force_recompute = self.habitat_config.get("RECOMPUTE_NAVMESH", False)
        # if osp.exists(navmesh_path) and not force_recompute:
        #     self.pathfinder.load_nav_mesh(navmesh_path)
        #     return

        # Set all articulated objects static
        motion_types = OrderedDict()
        for handle, art_obj in self.art_objs.items():
            motion_types[handle] = art_obj.motion_type
            art_obj.motion_type = MotionType.STATIC

        # Recompute navmesh
        self.recompute_navmesh(
            self.pathfinder,
            self.navmesh_settings,
            include_static_objects=True,
        )

        # Restore motion type
        for handle, motion_type in motion_types.items():
            self.art_objs[handle].motion_type = motion_type

        # self.pathfinder.save_nav_mesh(navmesh_path)

        self._cache_largest_island()

    def _cache_largest_island(self):
        navmesh_vertices = np.stack(
            self.pathfinder.build_navmesh_vertices(), axis=0
        )
        self._largest_island_radius = max(
            [self.pathfinder.island_radius(p) for p in navmesh_vertices]
        )

    def is_at_larget_island(self, position, eps=1e-4):
        assert self.pathfinder.is_navigable(position), position
        island_raidus = self.pathfinder.island_radius(position)
        return np.abs(island_raidus - self._largest_island_radius) <= eps

    def sleep_all_objects(self):
        rigid_obj_mgr = self.get_rigid_object_manager()
        for handle in rigid_obj_mgr.get_object_handles():
            obj = rigid_obj_mgr.get_object_by_handle(handle)
            obj.awake = False

        art_obj_mgr = self.get_articulated_object_manager()
        for handle in art_obj_mgr.get_object_handles():
            art_obj = art_obj_mgr.get_object_by_handle(handle)
            art_obj.awake = False

    def reset(self):
        # The agent and sensors are reset.
        super().reset()

        # Uncomment if the simulator is reset but not reconfigured
        # self.set_state(self._initial_state)

        # # Reset the articulated objects
        # self._set_articulated_objects_from_episode()

        # Reset the robot，重置机器人位置的代码
        self.robot.reset()

        # Place the robot
        # NOTE(jigu): I will set `start_position` out of the room,
        # so that some articulated objects can be initialized in tasks.
        episode = self.habitat_config.EPISODE
        # 机器人当前位置
        self.robot.base_T = mn_utils.to_Matrix4(
            episode["start_position"], episode["start_rotation"]
        )

        # Reset the gripper
        self.gripper.reset()

        # Sync before getting observations
        # self.sync_agent()
        self.sync_pyb_robot()

        return self.get_observations()

    def get_observations(self):
        self.robot.update_cameras()
        self._prev_sim_obs = self.get_sensor_observations()
        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations

    def internal_step(self, dt=None):
        """Internal simulation step."""
        if dt is None:
            dt = 1.0 / self.habitat_config.SIM_FREQ
        self.step_world(dt)
        self.robot.step()

    def internal_step_by_time(self, seconds):
        steps = int(seconds * self.habitat_config.SIM_FREQ)
        for _ in range(steps):
            self.internal_step()

    def get_state(self, include_robot=True):
        """Get the (kinematic) state of the simulation."""
        state = {
            "rigid_objs_T": {
                handle: obj.transformation
                for handle, obj in self.rigid_objs.items()
            },
            "art_objs_T": {
                handle: obj.transformation
                for handle, obj in self.art_objs.items()
            },
            "art_objs_qpos": {
                handle: obj.joint_positions
                for handle, obj in self.art_objs.items()
            },
        }
        if include_robot:
            state.update(
                {
                    "robot_state": self.robot.get_state(),
                    "grasped_obj": self.gripper.grasped_obj,
                    "grasped_marker": self.gripper.grasped_marker,
                }
            )
        return state

    def set_state(self, state: dict, include_robot=True):
        """Set the kinematic state of the simulation.

        Notes:
            The velocities and forces are set to 0.
            Be careful when using this function.
        """
        for handle, T in state["rigid_objs_T"].items():
            obj = self.rigid_objs[handle]
            obj.transformation = mn_utils.orthogonalize(T)
            obj.linear_velocity = mn.Vector3.zero_init()
            obj.angular_velocity = mn.Vector3.zero_init()

        for handle, T in state["art_objs_T"].items():
            art_obj = self.art_objs[handle]
            art_obj.transformation = mn_utils.orthogonalize(T)

        for handle, qpos in state["art_objs_qpos"].items():
            art_obj = self.art_objs[handle]
            art_obj.clear_joint_states()
            art_obj.joint_positions = qpos
            # art_obj.joint_velocities = np.zeros_like(art_obj.joint_velocities)
            # art_obj.joint_forces = np.zeros_like(art_obj.joint_forces)

        if include_robot:
            self.robot.set_state(state["robot_state"])

            self.gripper.desnap(True)  # desnap anyway
            if state["grasped_obj"] is not None:
                self.gripper.snap_to_obj(state["grasped_obj"])
            elif state["grasped_marker"] is not None:
                self.gripper.snap_to_marker(state["grasped_marker"])

    def sync_agent(self):
        """Synchronize the virtual agent with the robot.
        Thus, we can reuse habitat-baselines utilities for map.

        Notes:
            `habitat_sim.AgentState` uses np.quaternion (w, x, y, z) for rotation;
            however, it accepts a list of (x, y, z, w) as rvalue.
        """
        agent_state = self._default_agent.get_state()
        # agent_state.position = np.array(self.robot.sim_obj.translation)
        agent_state.position = self.robot.base_pos
        # align robot x-axis with agent z-axis
        agent_state.rotation = mn_utils.to_list(
            self.robot.sim_obj.rotation
            * mn.Quaternion.rotation(mn.Rad(-1.57), mn.Vector3(0, 1, 0))
        )
        self._default_agent.set_state(agent_state)

    def sync_pyb_robot(self):
        self.pyb_robot.set_joint_states(self.robot.arm_joint_pos)

    def step(self, action: Optional[int] = None):
        # virtual agent's action, only for compatibility.
        if action is not None:
            self._default_agent.act(action)

        # step physics
        for _ in range(self.habitat_config.CONTROL_FREQ):
            self.internal_step()

        # sync virtual agent
        # self.sync_agent()
        self.sync_pyb_robot()

        return self.get_observations()

    # -------------------------------------------------------------------------- #
    # Utilities
    # -------------------------------------------------------------------------- #
    def get_rigid_obj(self, index: int):
        handle = list(self.rigid_objs.keys())[index]
        return self.rigid_objs[handle]

    def get_rigid_objs_pos(self):
        """Get the positions of all rigid objects."""
        return np.stack(
            [
                np.array(obj.translation, dtype=np.float32)
                for obj in self.rigid_objs.values()
            ],
            axis=0,
        )

    def get_rigid_objs_pos_dict(self):
        return {
            k: np.array(obj.translation, dtype=np.float32)
            for k, obj in self.rigid_objs.items()
        }

    def get_target(self, index):
        handle = list(self.targets.keys())[index]
        tgt_obj = self.rigid_objs[handle]
        tgt_T = self.targets[handle]
        return tgt_obj, tgt_T

    def get_marker(self, index):
        return list(self.markers.values())[index]

    def get_robot_collision(
        self, include_grasped_obj=True, link_ids=None, verbose=False
    ):
        robot_id = self.robot.object_id
        grasped_obj_id = self.gripper.grasped_obj_id
        contact_points = self.get_physics_contact_points()

        contact_infos = coll_utils.get_contact_infos(
            contact_points, robot_id, link_ids=link_ids
        )
        if include_grasped_obj and grasped_obj_id is not None:
            contact_infos.extend(
                coll_utils.get_contact_infos(contact_points, grasped_obj_id)
            )

        if len(contact_infos) > 0:
            max_force = max(x["normal_force"] for x in contact_infos)

            # -------------------------------------------------------------------------- #
            # DEBUG(jigu): too large force usually means that base has penetrated some obj.
            # -------------------------------------------------------------------------- #
            if verbose and max_force > 1e6:
                print(
                    "DEBUG (collision)",
                    self.habitat_config["EPISODE"]["episode_id"],
                    self.habitat_config["EPISODE"]["scene_id"],
                )
                for info in contact_infos:
                    # if info["normal_force"] < 1e3:
                    #     continue
                    print(
                        "collide with",
                        get_object_handle_by_id(self, info["object_id"]),
                        info,
                    )
            # -------------------------------------------------------------------------- #
        else:
            max_force = 0.0
        return max_force

    def set_joint_pos_by_motor(
        self, art_obj: ManagedBulletArticulatedObject, link_id, pos, dt
    ):
        art_obj.awake = True
        motor_id = art_utils.get_motor_id_by_link_id(art_obj, link_id)
        jms = JointMotorSettings(pos, 0.3, 0, 0.3, 0.5)
        if motor_id is not None:
            ori_jms = art_obj.get_joint_motor_settings(motor_id)
            art_obj.update_joint_motor(motor_id, jms)
            self.internal_step_by_time(dt)
            art_obj.update_joint_motor(motor_id, ori_jms)
        else:
            motor_id = art_obj.create_joint_motor(link_id, jms)
            self.internal_step_by_time(dt)
            art_obj.remove_joint_motor(motor_id)

        # NOTE(jigu): Simulate one step after motor gain changes.
        self.internal_step()

    def set_fridge_state_by_motor(self, angle, dt=0.6):
        art_obj_mgr = self.get_articulated_object_manager()
        art_obj = art_obj_mgr.get_object_by_handle("fridge_:0000")
        self.set_joint_pos_by_motor(art_obj, 2, angle, dt=dt)

    def set_fridge_state(self, angle):
        art_obj_mgr = self.get_articulated_object_manager()
        art_obj = art_obj_mgr.get_object_by_handle("fridge_:0000")
        art_utils.set_joint_pos(art_obj, [1], [angle])

    def get_fridge_state(self):
        art_obj_mgr = self.get_articulated_object_manager()
        art_obj = art_obj_mgr.get_object_by_handle("fridge_:0000")
        return art_obj.joint_positions[1]

    def update_camera(self, sensor_name, cam2world: mn.Matrix4):
        agent_inv_T = self._default_agent.scene_node.transformation.inverted()
        sensor = self._sensors[sensor_name]._sensor_object
        sensor.node.transformation = mn_utils.orthogonalize(
            agent_inv_T @ cam2world
        )

    # -------------------------------------------------------------------------- #
    # Visualization
    # -------------------------------------------------------------------------- #
    def _remove_viz_objs(self):
        rigid_obj_mgr = self.get_rigid_object_manager()
        for name, obj in self.viz_objs.items():
            assert obj.is_alive, name
            if self.verbose:
                print(
                    "Remove a vis object",
                    name,
                    obj.handle,
                    obj.object_id,
                    obj.is_alive,
                )
            rigid_obj_mgr.remove_object_by_id(obj.object_id)
        # 相当于是清除了所有的可见物品
        self.viz_objs = OrderedDict()

    # -------------------------------------------------------------------------- #
    # 可视化部分，将物体加入环境
    # -------------------------------------------------------------------------- #
    def add_viz_obj( # 增加可视化的物体
        self,
        position: mn.Vector3,
        scale=mn.Vector3(1, 1, 1),
        rotation: Optional[mn.Quaternion] = None,
        template_name="coord_frame",
    ):
        # 获取物理对象管理器
        obj_attr_mgr = self.get_object_template_manager()
        # 获取刚体对象管理器
        rigid_obj_mgr = self.get_rigid_object_manager()

        # register a new template for visualization
        # 第一步，先要获取template
        # 在这里obj_attr_mgr.get_template_handles(template_name)[0]其实对应的就是'habitat_extensions/assets/objects/primitives/transform_box.object_config.json'
        template = obj_attr_mgr.get_template_by_handle(
            obj_attr_mgr.get_template_handles(template_name)[0]
        )
        # 第二步，修改scale，即在Isaac-sim中也尝试过的进行缩放
        template.scale = scale
        # 第三步，进行注册register_template
        template_id = obj_attr_mgr.register_template(
            template, f"viz_{template_name}"
        )

        # 第四步，通过id添加物品
        viz_obj = rigid_obj_mgr.add_object_by_template_id(template_id)
        obj_utils.make_render_only(viz_obj)
        viz_obj.translation = position
        if rotation is not None:
            viz_obj.rotation = rotation
        return viz_obj

    def visualize_frame(self, name, T: mn.Matrix4, scale=1.0):
        assert name not in self.viz_objs, name
        self.viz_objs[name] = self.add_viz_obj(
            position=T.translation,
            scale=mn.Vector3(scale),
            rotation=mn_utils.mat3_to_quat(T.rotation()),
            template_name="coord_frame",
        )

    def visualize_arrow(self, name, position, orientation, scale=1.0):
        assert name not in self.viz_objs, name
        rotation = mn.Quaternion.rotation(
            mn.Rad(orientation), mn.Vector3(0, 1, 0)
        )
        self.viz_objs[name] = self.add_viz_obj(
            position=position,
            scale=mn.Vector3(scale),
            rotation=rotation,
            template_name="arrow",
        )

    def visualize_markers(self):
        for name, marker in self.markers.items():
            self.visualize_frame(name, marker.transformation, scale=0.15)

    def visualize_target(self, index):
        # TODO:在此函数中可以获得物品的bbox
        tgt_obj, tgt_T = self.get_target(index)
        obj_bb = obj_utils.get_aabb(tgt_obj)
        viz_obj = self.add_viz_obj(
            position=tgt_T.translation,
            scale=obj_bb.size() * 0.5,
            rotation=mn_utils.mat3_to_quat(tgt_T.rotation()),
            template_name="transform_box",
        )
        self.viz_objs[f"target.{index}"] = viz_obj
        return viz_obj

    def visualize_region(
        self,
        name,
        region: mn.Range2D,
        T: mn.Matrix4,
        height=None,
        template="region_green",
    ):
        center = mn.Vector3(region.center_x(), region.center_y(), 0.0)
        center = T.transform_point(center)
        if height is not None:
            center.y = height
        scale = mn.Vector3(region.size_x(), region.size_y(), 1.0)
        viz_obj = self.add_viz_obj(
            position=center,
            scale=scale,
            rotation=mn_utils.mat3_to_quat(T.rotation()),
            template_name=template,
        )
        self.viz_objs[name] = viz_obj

    def render(self, mode: str):
        """Render with additional debug info.
        Users can add more visualization to viz_objs before calling sim.render().
        """
        # self.visualize_frame("ee_frame", self.robot.gripper_T, scale=0.15)
        rendered_frame = super().render(mode=mode)
        # print(rendered_frame)
        # Remove visualization in case polluate observations
        self._remove_viz_objs()
        return rendered_frame

    def get_color(self, idx):
        # 可达性：蓝色>绿色>黄色>橘色>红色
        dark_blue = mn.Color4(0, 1 / 255, 249 / 255, 1)
        green = mn.Color4(2 / 255, 247 / 255, 2 / 255, 1)
        yellow = mn.Color4(248 / 255, 250 / 255, 1 / 255, 1)
        orange = mn.Color4(255 /255, 153 / 255, 51 / 255, 1)
        ori = mn.Color4(247 / 255, 204 / 255, 239 / 255, 1)
        red = mn.Color4(247 / 255, 0, 0, 1)
        if idx == 0:
            return dark_blue
        elif idx == 1:
            return green
        elif idx == 2:
            return yellow
        elif idx == 3:
            return orange
        elif idx == 4:
            return red

    """JL修改，用于显示机器人周围的可达性区域"""
    def show_reachability(self, q_values, round_loc, resolution_dis=0.2, resolution_ang=10):
        debug_line_render = self.get_debug_line_render()
        debug_line_render.set_line_width(1)

        reshaped_q_values = q_values.reshape((-1, 18))
        # 沿着第二个轴计算平均值
        average_values = np.mean(reshaped_q_values, axis=1)

        # average_values = q_values.copy()

        _len = len(average_values)

        # print("_len=%d"%(_len))

        # 将q值转为颜色
        min_val = np.min(average_values)
        max_val = np.max(average_values)
        normalized_q = (average_values - min_val) / (max_val - min_val)

        """这是一段调试q值大小的代码"""
        # 靠近的过程中正的值也大负的值也大
        avg_val = np.average(average_values)
        print("q值中：min_val=%s, max_val=%s, avg_val=%s" % (min_val, max_val, avg_val))
        """这是一段调试q值大小的代码"""

        colors = np.zeros([_len])
        for i in range(_len):
            if normalized_q[i] > 0.85:
                colors[i] = 0
            elif normalized_q[i] > 0.8:
                colors[i] = 1
            elif normalized_q[i] > 0.7:
                colors[i] = 2
            elif normalized_q[i] > 0.6:
                colors[i] = 3
            else:
                colors[i] = 4

        # TODO：索引测试正确，见索引值横向连接测试通过.png和索引值涟漪状连接测试通过.png
        # 采用画线的方式
        # 画线包括两部分：从圆心向外画线，在Jupyter文件中从测试过索引值的正确性
        start = 0
        line_len = int((1.3 - 0.3) / resolution_dis)
        # self.console_logger.info("line_len=%d"%(line_len))
        for phi_cnt in range(0, 360, resolution_ang):
            for j in range(start, start+line_len-1, 1):
                cur_line = round_loc[j, :]
                cur_color = self.get_color(colors[j])
                next_line = round_loc[j+1, :]
                line_list = []
                line_list.append(mn.Vector3(cur_line[0], cur_line[1], cur_line[2]))
                line_list.append(mn.Vector3(next_line[0], next_line[1], next_line[2]))
                debug_line_render.draw_path_with_endpoint_circles(
                    line_list,
                    0.02,
                    cur_color,
                )
                # debug_line_render.draw_transformed_line(
                #     mn.Vector3(cur_line[0], cur_line[1], cur_line[2]),
                #     mn.Vector3(next_line[0], next_line[1], next_line[2]),
                #     cur_color,
                #     cur_color,
                # )

            start += line_len

        # # 一圈一圈画线如同涟漪一般
        start = 0
        round_len = int((360 - 0) / resolution_ang)
        for r in np.arange(0.3, 1.3, resolution_dis):
            for j in range(start, _len - line_len, line_len):
                cur_line = round_loc[j, :]
                cur_color = self.get_color(colors[j])
                next_line = round_loc[j+line_len, :]
                line_list = []
                line_list.append(mn.Vector3(cur_line[0], cur_line[1], cur_line[2]))
                line_list.append(mn.Vector3(next_line[0], next_line[1], next_line[2]))
                debug_line_render.draw_path_with_endpoint_circles(
                    line_list,
                    0.02,
                    cur_color,
                )

            cur_line = round_loc[j+line_len, :]
            cur_color = self.get_color(colors[j+line_len])
            next_line = round_loc[start, :]
            line_list = []
            line_list.append(mn.Vector3(cur_line[0], cur_line[1], cur_line[2]))
            line_list.append(mn.Vector3(next_line[0], next_line[1], next_line[2]))
            debug_line_render.draw_path_with_endpoint_circles(
                line_list,
                0.02,
                cur_color,
            )

            # print("横向cur_color[%d]=%s" % (j+line_len, cur_color))
                # debug_line_render.draw_transformed_line(
                #     mn.Vector3(cur_line[0], cur_line[1], cur_line[2]),
                #     mn.Vector3(next_line[0], next_line[1], next_line[2]),
                #     cur_color,
                #     cur_color,
                # )
            start += 1


    """JL修改：用于显示桌子周围的可达性"""
    def show_surrounding_reachablity(self, q_val, surround_loc, size=8):
        debug_line_render = self.get_debug_line_render()
        debug_line_render.push_transform(self.robot.base_T)
        debug_line_render.set_line_width(1)

        _len = len(q_val)

        # # # 将q值转为颜色
        # min_val = np.min(q_val)
        # max_val = np.max(q_val)
        # normalized_q = (q_val - min_val) / (max_val - min_val)
        #
        # """这是一段调试q值大小的代码"""
        # # 靠近的过程中正的值也大负的值也大
        # avg_val = np.average(q_val)
        # print("q值中：min_val=%s, max_val=%s, avg_val=%s" % (min_val, max_val, avg_val))
        # """这是一段调试q值大小的代码"""

        normalized_q = q_val

        colors = np.zeros([_len])
        for i in range(_len):
            if normalized_q[i] > 0.8:
                colors[i] = 0
            elif normalized_q[i] > 0.7:
                colors[i] = 1
            elif normalized_q[i] > 0.5:
                colors[i] = 2
            elif normalized_q[i] > 0.3:
                colors[i] = 3
            else:
                colors[i] = 4

        for i in range(_len-1):
            cur_line = surround_loc[i, :]
            cur_color = self.get_color(colors[i])
            next_line = surround_loc[i+1, :]
            line_list = []
            line_list.append(mn.Vector3(cur_line[0], cur_line[1], cur_line[2]))
            line_list.append(mn.Vector3(next_line[0], next_line[1], next_line[2]))
            debug_line_render.draw_path_with_endpoint_circles(
                line_list,
                0.02,
                cur_color,
            )

        debug_line_render.pop_transform()