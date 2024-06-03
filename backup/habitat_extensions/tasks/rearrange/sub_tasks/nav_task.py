import magnum as mn
import numpy as np
from habitat import logger
from habitat.core.registry import registry

from habitat_extensions.utils import art_utils

from ..task import RearrangeEpisode, RearrangeTask
from ..task_utils import (
    check_collision_free,
    check_start_state,
    compute_region_goals_v1,
    compute_start_positions_from_map_v1,
    compute_start_state,
    filter_positions,
    sample_navigable_point_within_region,
    sample_random_start_state_v1,
)


@registry.register_task(name="RearrangeNavTask-v0")
class RearrangeNavTask(RearrangeTask):
    sub_task: str

    # 初始化方法，接受RearrangeEpisode对象作为参数，用于初始化
    def initialize(self, episode: RearrangeEpisode):
        # 获取当前的状态
        sim_state = self._sim.get_state()  # snapshot
        self.sub_task = None
        is_initialized = False  # whether nav_goals is set

        # 获取环境中目标物体的数量
        n_targets = len(self._sim.targets)
        if "TARGET_INDEX" in self._config:
            tgt_indices = [self._config.TARGET_INDEX]
        else:
            tgt_indices = self.np_random.permutation(n_targets)

        # 遍历目标物体索引
        for tgt_idx in tgt_indices:
            # 设置目标物体
            self._set_target(tgt_idx)
            # 获取支持的子任务类型
            supported_tasks = self._get_supported_tasks()
            supported_tasks = self.np_random.permutation(
                supported_tasks
            ).tolist()

            # 设置拾取目标和放置目标的位置
            # Decide pick goal before initializing subtask and receptacle
            self.pick_goal = np.array(
                self.tgt_obj.translation, dtype=np.float32
            )
            self.place_goal = np.array(
                self.tgt_T.translation, dtype=np.float32
            )

            # 遍历支持的子任务类型
            for sub_task in supported_tasks:
                # 设置子任务类型、初始化目标容器，然后初始化目标
                self._set_sub_task(sub_task)
                self._initialize_target_receptacle()
                is_initialized = self._initialize_goals(episode)
                if is_initialized:
                    break
                else:
                    logger.warning(
                        "Episode {}({}): can not initialize goals for {}({})".format(
                            episode.episode_id,
                            episode.scene_id,
                            self.sub_task,
                            self.tgt_idx,
                        )
                    )
                    # 如果初始化失败，恢复仿真器状态
                    self._sim.set_state(sim_state)  # recover from snapshot

            if is_initialized:
                break

        # 如果无法找到任何目标，抛出运行时错误
        if not is_initialized:
            raise RuntimeError(
                "Episode {}: fail to find any goal".format(episode.episode_id)
            )

        self._initialize_ee_pos()
        start_state = self.sample_start_state()
        if start_state is None:
            raise RuntimeError(
                "Episode {}: fail to find a valid start state".format(
                    episode.episode_id
                )
            )

        # 当前初始化的位置
        self._sim.robot.base_pos = start_state[0]
        # 当前初始的朝向
        self._sim.robot.base_ori = start_state[1]
        if self.sub_task == "place":
            self._sim.robot.open_gripper()
            self._sim.gripper.snap_to_obj(self.tgt_obj)
        self._sim.internal_step_by_time(0.1)

    def _get_supported_tasks(self):
        supported_tasks = ["pick", "place"]
        n_targets = len(self._sim.targets)
        # NOTE(jigu): hardcode, using n_targets to distinguish SetTable
        if self._has_target_in_fridge() and n_targets == 2:
            supported_tasks.extend(["open_fridge", "close_fridge"])
        elif self._has_target_in_drawer() and n_targets == 2:
            supported_tasks.extend(["open_drawer", "close_drawer"])
        supported_tasks = [
            x for x in supported_tasks if x in self._config.SUB_TASKS
        ]
        # print(supported_tasks)
        return supported_tasks

    def _set_sub_task(self, sub_task):
        self.sub_task = sub_task
        if sub_task == "place":
            self.tgt_receptacle_info = self._goal_receptacles[self.tgt_idx]
        else:
            self.tgt_receptacle_info = self._target_receptacles[self.tgt_idx]

    def _initialize_target_receptacle(self):
        self.tgt_receptacle = None
        self.tgt_receptacle_link = None
        self.init_start_pos = None  # for Pick/Place

        receptacle_handle, receptacle_link_id = self.tgt_receptacle_info
        art_obj_mgr = self._sim.get_articulated_object_manager()

        if self._has_target_in_fridge():
            self.tgt_receptacle = art_obj_mgr.get_object_by_handle(
                receptacle_handle
            )
            self.tgt_receptacle_link = self.tgt_receptacle.get_link_scene_node(
                receptacle_link_id
            )

            # Open the fridge
            if self.sub_task in ["pick", "place", "close_fridge"]:
                init_range = self._config.get(
                    "FRIDGE_INIT_RANGE", [2.356, 2.356]
                )
                init_qpos = self.np_random.uniform(*init_range)

                # Kinematic alternative to set link states
                # art_utils.set_joint_pos(self.tgt_receptacle, [1], [init_qpos])

                # Dynamic way to set link
                self._sim.set_joint_pos_by_motor(
                    self.tgt_receptacle, 2, init_qpos, dt=0.6
                )
                # print(init_qpos, self.tgt_receptacle.joint_positions)

            T = self.tgt_receptacle.transformation
            offset = mn.Vector3(1.0, 0, 0)
            self.init_start_pos = np.array(T.transform_point(offset))

        elif self._has_target_in_drawer():
            self.tgt_receptacle = art_obj_mgr.get_object_by_handle(
                receptacle_handle
            )
            self.tgt_receptacle_link = self.tgt_receptacle.get_link_scene_node(
                receptacle_link_id
            )

            # Open the drawer
            if self.sub_task in ["pick", "place", "close_drawer"]:
                init_range = self._config.get("DRAWER_INIT_RANGE", [0.5, 0.5])
                init_qpos = self.np_random.uniform(*init_range)

                # Kinematic alternative to set link states
                pos_offset = self.tgt_receptacle.get_link_joint_pos_offset(
                    receptacle_link_id
                )
                T1 = self.tgt_receptacle_link.transformation
                art_utils.set_joint_pos(
                    self.tgt_receptacle, [pos_offset], [init_qpos]
                )
                T2 = self.tgt_receptacle_link.transformation
                t = T2.translation - T1.translation

                if self.sub_task == "close_drawer":
                    self.tgt_obj.transformation = self.tgt_T
                else:
                    self.tgt_obj.translation = self.tgt_obj.translation + t

            T = self.tgt_receptacle_link.transformation
            offset = mn.Vector3(0.8, 0, 0)
            self.init_start_pos = np.array(T.transform_point(offset))

        # PrepareGroceries
        elif (
            self._config.get("FRIDGE_INIT", False)
            and len(self._sim.targets) == 3
        ):
            init_range = self._config.get("FRIDGE_INIT_RANGE", [2.356, 2.356])
            init_qpos = self.np_random.uniform(*init_range)
            self._sim.set_fridge_state_by_motor(init_qpos)

    def _initialize_goals(self, episode: RearrangeEpisode) -> bool:
        self.nav_goal = None

        self.marker = None
        self.spawn_region = None
        self.spawn_T = None

        if self.sub_task == "pick":
            self.nav_goal = compute_start_state(
                self._sim, self.pick_goal, init_start_pos=self.init_start_pos
            )
        elif self.sub_task == "place":
            self.nav_goal = compute_start_state(
                self._sim, self.place_goal, init_start_pos=self.init_start_pos
            )

        receptacle_link_id = self.tgt_receptacle_info[1]
        if self.sub_task == "open_drawer":
            marker_name = "cab_push_point_{}".format(receptacle_link_id)
            self.marker = self._sim.markers[marker_name]
            self.spawn_region = mn.Range2D([0.80, -0.35], [0.95, 0.35])
            self.spawn_T = self.marker.transformation
        elif self.sub_task == "close_drawer":
            marker_name = "cab_push_point_{}".format(receptacle_link_id)
            self.marker = self._sim.markers[marker_name]
            self.spawn_region = mn.Range2D([0.30, -0.35], [0.45, 0.35])
            self.spawn_T = self.marker.transformation
        elif self.sub_task == "open_fridge":
            marker_name = "fridge_push_point"
            self.marker = self._sim.markers[marker_name]
            self.spawn_region = mn.Range2D([0.933, -0.6], [1.833, 0.6])
            self.spawn_T = self.marker.art_obj.transformation
        elif self.sub_task == "close_fridge":
            marker_name = "fridge_push_point"
            self.marker = self._sim.markers[marker_name]
            self.spawn_region = mn.Range2D([0.933, -0.6], [1.833, 0.6])
            self.spawn_T = self.marker.art_obj.transformation

        if self.sub_task in [
            "open_drawer",
            "close_drawer",
            "open_fridge",
            "close_fridge",
        ]:
            self.nav_goal = self.sample_nav_goal_within_region(
                self.spawn_region, self.spawn_T
            )
            if self.nav_goal is None:
                return False

        if not self._sim.is_at_larget_island(self.nav_goal[0]):
            logger.warning(
                "Episode {}({}): nav_goal is not at the largest island for {}({})".format(
                    episode.episode_id,
                    episode.scene_id,
                    self.sub_task,
                    self.tgt_idx,
                )
            )
            return False

        return True

    def sample_nav_goal_within_region(
        self,
        spawn_region: mn.Range2D,
        T: mn.Matrix4,
        max_trials=100,
        max_collision_force=0.0,
        verbose=False,
    ):
        state = self._sim.get_state()  # snapshot

        look_at_pos = np.array(T.translation, dtype=np.float32)
        start_pos, _ = compute_start_state(self._sim, look_at_pos)
        height = start_pos[1]

        for _ in range(max_trials):
            start_pos = sample_navigable_point_within_region(
                self._sim,
                region=spawn_region,
                height=height,
                T=T,
                rng=self.np_random,
            )
            if start_pos is None:
                continue

            _, start_ori = compute_start_state(
                self._sim, look_at_pos, init_start_pos=start_pos
            )

            self._sim.robot.base_pos = start_pos
            self._sim.robot.base_ori = start_ori

            if max_collision_force is not None:
                is_safe = check_collision_free(self._sim, max_collision_force)
                self._sim.set_state(state)  # restore snapshot
                if not is_safe:
                    if verbose:
                        print("Not collision-free")
                    continue

            return start_pos, start_ori

    def sample_start_state(self, max_trials=20, verbose=False):
        for i in range(max_trials):
            start_state = sample_random_start_state_v1(
                self._sim, max_trials=20, rng=self.np_random
            )
            if start_state is None:
                if verbose:
                    print("The goal is not navigable")
                continue
            is_valid = check_start_state(
                self._sim,
                self,
                *start_state,
                task_type=self.sub_task,
                max_collision_force=0.0,
                verbose=verbose,
            )
            if is_valid:
                if verbose:
                    print(f"Find a valid start state at {i}-th trial")
                return start_state

    def render(self, mode):
        self._sim.set_object_bb_draw(True, self.tgt_obj.object_id)
        self._sim.visualize_target(self.tgt_idx)
        self._sim.visualize_arrow(
            "nav_goal", self.nav_goal[0], self.nav_goal[1], scale=0.3
        )

        # Show pick goal
        if self.sub_task != "place":
            self._sim.viz_objs["pick_goal"] = self._sim.add_viz_obj(
                self.pick_goal
            )

        if self.spawn_region is not None:
            self._sim.visualize_region(
                "spawn_region",
                self.spawn_region,
                self.spawn_T,
                height=self._sim.robot.base_pos[1],
            )

        ret = self._sim.render(mode)
        self._sim.set_object_bb_draw(False, self.tgt_obj.object_id)
        return ret


@registry.register_task(name="RearrangeNavTask-v1")
class RearrangeNavTaskV1(RearrangeNavTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_nav_goals = dict()

    def _has_cache_nav_goals(self, episode_id):
        if episode_id not in self._cache_nav_goals:
            return False
        key = (self.tgt_idx, self.sub_task)
        if key not in self._cache_nav_goals[episode_id]:
            return False
        return True

    def _get_cache_nav_goals(self, episode_id):
        key = (self.tgt_idx, self.sub_task)
        # print("Cache is used", episode_id, key)
        return self._cache_nav_goals[episode_id][key]

    def _set_cache_nav_goals(self, episode_id):
        if episode_id not in self._cache_nav_goals:
            self._cache_nav_goals[episode_id] = dict()
        key = (self.tgt_idx, self.sub_task)
        self._cache_nav_goals[episode_id][key] = self.nav_goals
        # print("Cache is set", episode_id, key)

    def _initialize_goals(self, episode: RearrangeEpisode) -> bool:
        self.nav_goals = None
        self.look_at_pos = None  # 用于确定机器人视线方向或朝向

        height = self._sim.pathfinder.snap_point(self.pick_goal)[1]
        assert not np.isnan(height), self.pick_goal
        receptacle_link_id = self.tgt_receptacle_info[1]

        self.marker = None
        self.spawn_region = None
        self.spawn_T = None

        if self.sub_task in ["pick", "place"]:
            # 如果是 "place" 子任务，则 self.look_at_pos 的值为 place_goal，表示机器人放置物体的目标位置
            # 如果是其他子任务（如 "pick"），则 self.look_at_pos 的值为 pick_goal，表示机器人拾取物体的目标位置
            self.look_at_pos = (
                self.place_goal if self.sub_task == "place" else self.pick_goal
            )

            if self._has_target_in_fridge():
                self.marker = self._sim.markers["fridge_push_point"]
                # self.spawn_region = mn.Range2D([0.933, -0.6], [1.833, 0.6])
                self.spawn_region = mn.Range2D([1.0, -0.5], [1.8, 0.5])
                self.spawn_T = self.tgt_receptacle.transformation
                # radius = 2.0
                radius = None
            elif self._has_target_in_drawer():  # only for pick
                marker_name = "cab_push_point_{}".format(receptacle_link_id)
                self.marker = self._sim.markers[marker_name]
                self.spawn_region = mn.Range2D([0.30, -0.35], [0.45, 0.35])
                self.spawn_T = self.marker.transformation
                radius = None
            else:
                self.spawn_region = None
                self.spawn_T = mn.Matrix4.translation(self.look_at_pos)
                radius = 0.8
                # radius = 2.0

            if self._has_cache_nav_goals(episode.episode_id):
                self.nav_goals = self._get_cache_nav_goals(episode.episode_id)
            else:
                if self._has_target_in_container():
                    self.nav_goals = compute_start_positions_from_map_v1(
                        self._sim,
                        T=self.spawn_T,
                        region=self.spawn_region,
                        radius=radius,
                        height=height,
                        debug=False,
                    )
                else:
                    self.nav_goals = compute_region_goals_v1(
                        self._sim,
                        T=self.spawn_T,
                        region=self.spawn_region,
                        radius=radius,
                        height=height,
                        debug=False,
                    )

                # The drawer can have different initial states for one episode 
                if not self._has_target_in_drawer():
                    self._set_cache_nav_goals(episode.episode_id)

            # Post-processing for picking or placing in fridge
            if self._has_target_in_fridge():
                self.nav_goals = filter_positions(
                    self.nav_goals,
                    self.marker.transformation,
                    direction=[-1.0, 0.0, 0.0],
                    clearance=0.4,
                )

        if self.sub_task in [
            "open_drawer",
            "close_drawer",
            "open_fridge",
            "close_fridge",
        ]:
            if self.sub_task == "open_drawer":
                marker_name = "cab_push_point_{}".format(receptacle_link_id)
                self.marker = self._sim.markers[marker_name]
                self.spawn_region = mn.Range2D([0.80, -0.35], [0.95, 0.35])
                self.spawn_T = self.marker.transformation
            elif self.sub_task == "close_drawer":
                marker_name = "cab_push_point_{}".format(receptacle_link_id)
                self.marker = self._sim.markers[marker_name]
                self.spawn_region = mn.Range2D([0.30, -0.35], [0.45, 0.35])
                self.spawn_T = self.marker.transformation
            elif self.sub_task == "open_fridge":
                self.marker = self._sim.markers["fridge_push_point"]
                # self.spawn_region = mn.Range2D([0.933, -0.6], [1.833, 0.6])
                self.spawn_region = mn.Range2D([0.9, -0.5], [1.8, 0.5])
                self.spawn_T = self.marker.art_obj.transformation
            elif self.sub_task == "close_fridge":
                self.marker = self._sim.markers["fridge_push_point"]
                # self.spawn_region = mn.Range2D([0.933, -0.6], [1.833, 0.6])
                self.spawn_region = mn.Range2D([0.9, -0.5], [1.8, 0.5])
                self.spawn_T = self.marker.art_obj.transformation

            if self._has_cache_nav_goals(episode.episode_id):
                self.nav_goals = self._get_cache_nav_goals(episode.episode_id)
            else:
                self.nav_goals = compute_start_positions_from_map_v1(
                    self._sim,
                    T=self.spawn_T,
                    region=self.spawn_region,
                    radius=None,
                    height=height,
                    debug=False,
                )
                # NOTE(jigu): We assume that the fridge state is not considered
                if self.sub_task in ["open_fridge", "close_fridge"]:
                    self._set_cache_nav_goals(episode.episode_id)

            self.look_at_pos = np.array(
                self.spawn_T.translation, dtype=np.float32
            )

        if self.nav_goals is None or len(self.nav_goals) == 0:
            return False

        return True

    def render(self, mode):
        self._sim.set_object_bb_draw(True, self.tgt_obj.object_id)
        self._sim.visualize_target(self.tgt_idx)

        # Visualize navigation goals
        for i, nav_goal in enumerate(self.nav_goals[::10]):
            pos, ori = compute_start_state(
                self._sim, self.look_at_pos, init_start_pos=nav_goal
            )
            self._sim.visualize_arrow(f"nav_goal_{i}", pos, ori, scale=0.3)

        # Show pick goal
        if self.sub_task != "place":
            self._sim.viz_objs["pick_goal"] = self._sim.add_viz_obj(
                self.pick_goal
            )

        if self.spawn_region is not None:
            self._sim.visualize_region(
                "spawn_region",
                self.spawn_region,
                self.spawn_T,
                height=self._sim.robot.base_pos[1],
            )

        ret = self._sim.render(mode)
        self._sim.set_object_bb_draw(False, self.tgt_obj.object_id)
        return ret
