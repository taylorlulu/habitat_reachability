import magnum as mn
import numpy as np
from habitat import logger
from habitat.core.registry import registry

from habitat_extensions.utils import art_utils, states_utils, obj_utils
from habitat_extensions.utils.net_utils import get_q_values

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
            # 设置目标物体，这是在仿真环境中设置待抓取的目标物品
            # 在这一步中获得self.tgt_obj, self.tgt_T
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


        """应该将可达性的数据放在初始化部分"""
        # 第一步，获取当前的handle
        rigid_obj_mgr = self._sim.get_rigid_object_manager()
        art_obj_mgr = self._sim.get_articulated_object_manager()

        # 第二步，对环境建模，获取当前待抓取物品附近的所有物品
        drawer_handle = self.tgt_receptacle_info[0]

        # 获取当前的transformation矩阵
        if self.sub_task == 'pick':
            self.cur_tgt_T = self.tgt_obj.root_scene_node.transformation
        else:
            self.cur_tgt_T = self.tgt_T

        # 获取当前容器的obj
        self.container_obj = art_obj_mgr.get_object_by_handle(drawer_handle)
        if self.container_obj is None:
            self.container_obj = rigid_obj_mgr.get_object_by_handle(drawer_handle)

        self.obj_list, self.points_global = states_utils.get_nearby_objects(self.container_obj, self.cur_tgt_T,
                                                                            self._sim.rigid_objs)

        # 获取当前状态的方法
        # (states, self.surround_loc, surrounding_actions, self.surround_theta) = states_utils.get_object_states2(self._sim.robot,
        #                                                                                    self.cur_tgt_T,
        #                                                                                    self.obj_list,
        #                                                                                    self.nav_goals)

        (states, self.surround_loc, surrounding_actions, self.surround_theta) = states_utils.get_object_states_graph(
            self._sim.robot,
            self.cur_tgt_T,
            self.obj_list,
            self.nav_goals)

        # (states, self.surround_loc, surrounding_actions) = states_utils.get_round_states(self._sim.robot,
        #                                                                                    self.cur_tgt_T,
        #                                                                                    self.obj_list)

        # 获取可达性的q值函数数据
        # 为什么在这部分会存在states与surrounding_actions的维度不同的情况而在get_object_states2中不会出现
        self.q = get_q_values(states, surrounding_actions)

        print("self.q.shape=%s"%(str(self.q.shape)))


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
            # 如果是放置操作，则是goal_receptacles
            self.tgt_receptacle_info = self._goal_receptacles[self.tgt_idx]
        else:
            # 这是当前的目标容器，如果不是放置操作则都是target_receptacles
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
        # self._sim.visualize_target(self.tgt_idx)
        # 显示导航的终点位置
        self._sim.visualize_arrow(
            "nav_goal", self.nav_goal[0], self.nav_goal[1], scale=0.3
        )

        # Show pick goal
        # if self.sub_task != "place":
        #     self._sim.viz_objs["pick_goal"] = self._sim.add_viz_obj(
        #         self.pick_goal
        #     )

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

        # 标记方向
        self.marker = None
        # 定义可行的绿色区域
        self.spawn_region = None
        # 就是容器当前的位置
        self.spawn_T = None
        # 每轮的起始重新更新索引
        self.cnt = 0

        if self.sub_task in ["pick", "place"]:
            # 如果是 "place" 子任务，则 self.look_at_pos 的值为 place_goal，表示机器人放置物体的目标位置
            # 如果是其他子任务（如 "pick"），则 self.look_at_pos 的值为 pick_goal，表示机器人拾取物体的目标位置
            self.look_at_pos = (
                self.place_goal if self.sub_task == "place" else self.pick_goal
            )

            if self._has_target_in_fridge():
                # 获取当前marker
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
                # TODO: 这个参数很重要，是设置的机器人抓取的位置范围
                radius = 0.9
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
                    # 针对普通导航区域的获取，TODO：这个参数很重要获取容器附近的可导航信息
                    self.nav_goals = compute_region_goals_v1(
                        self._sim,
                        T=self.spawn_T,
                        region=self.spawn_region,
                        radius=radius,
                        height=height,
                        meters_per_pixel=0.02,  # 0.03, 0.02，在官方给的默认值为0.05
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
                # spawn_region是一个固定区域
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
                # 如果将debug参数设置为False可以显示可到达区域，部分可能到达的区域
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
        viz_obj = self._sim.visualize_target(self.tgt_idx)

        if self.cnt == 0:
            if self.sub_task == 'pick':
                self.cur_tgt_T = self.tgt_obj.root_scene_node.transformation
            else:
                self.cur_tgt_T = self.tgt_T

        # Visualize navigation goals
        # 绘制指向目标的方向箭头
        # for i, nav_goal in enumerate(self.nav_goals[::10]):
        #     pos, ori = compute_start_state(
        #         self._sim, self.look_at_pos, init_start_pos=nav_goal
        #     )
        #     self._sim.visualize_arrow(f"nav_goal_{i}", pos, ori, scale=0.3)

        # Show pick goal
        # if self.sub_task != "place":
        #     self._sim.viz_objs["pick_goal"] = self._sim.add_viz_obj(
        #         self.pick_goal
        #     )

        # 只针对spawn_region存在的情况
        # 绘制spawn_region区域
        if self.spawn_region is not None:
            self._sim.visualize_region(
                "spawn_region",
                self.spawn_region,
                self.spawn_T,
                height=self._sim.robot.base_pos[1],
            )

        # 其实只需要在它的半径范围内的点就好了self.nav_goals范围内
        debug_line_render = self._sim.get_debug_line_render()
        rigid_obj_mgr = self._sim.get_rigid_object_manager()
        art_obj_mgr = self._sim.get_articulated_object_manager()

        """第一步：绘制当前待抓取物品'024_bowl_:0000'的bbox"""
        """用绿色方框绘制当前，目标物体位置"""
        debug_line_render.push_transform(self.cur_tgt_T)
        obj_bb = obj_utils.get_aabb(self.tgt_obj)
        # debug_line_render.draw_box(
        #     obj_bb.min, obj_bb.max, mn.Color4(236/255.0, 116/255.0, 25/255.0, 1.0)
        # )
        debug_line_render.draw_box(
            obj_bb.min, obj_bb.max, mn.Color4(1, 0, 0, 1.0)
        )
        debug_line_render.pop_transform()

        """在机器人坐标系下绘制直线，到达目标物体位置"""
        # debug_line_render.push_transform(self._sim.robot.base_T)
        # cur_bbox = states_utils.transformation2robot_frame(states_utils.get_points(self.tgt_obj.root_scene_node.cumulative_bb),
        #                                                 self.cur_tgt_T,
        #                                                 self._sim.robot.base_T)
        #
        # debug_line_render.draw_transformed_line(
        #     mn.Vector3.zero_init(),
        #     mn.Vector3(cur_bbox[0, 0], cur_bbox[0, 1], cur_bbox[0, 2]),
        #     mn.Color4(1.0, 0.0, 0.0, 1.0),
        #     mn.Color4(50.0, 1.0, 10.0, 1.0),
        # )
        # debug_line_render.pop_transform()

        """用蓝色方框绘制障碍物物品"""
        # for obj in self.obj_list:
        #     # 绘制方块，如果采用push_transform则是在transformation的坐标系下进行绘制
        #     debug_line_render.push_transform(obj.root_scene_node.transformation)
        #     obj_bb = obj_utils.get_aabb(obj)
        #     debug_line_render.draw_box(
        #         obj_bb.min, obj_bb.max, mn.Color4(156/255.0, 192/255.0, 229/255.0, 1.0)
        #     )
        #     debug_line_render.pop_transform()
        #
        # debug_line_render.push_transform(self.container_obj.root_scene_node.transformation)
        # bbox = states_utils.get_initial_box(self.container_obj)
        # debug_line_render.draw_box(
        #     bbox.min, bbox.max, mn.Color4(144/255.0, 207/255.0, 76/255.0, 1.0)
        # )
        # debug_line_render.pop_transform()

        """绘制障碍物部分"""
        # obstacle_name = "frl_apartment_table_01_:0000"
        # obstacle_obj = art_obj_mgr.get_object_by_handle(obstacle_name)
        # if obstacle_obj is None:
        #     obstacle_obj = rigid_obj_mgr.get_object_by_handle(obstacle_name)
        # debug_line_render.push_transform(obstacle_obj.root_scene_node.transformation)
        # bbox = states_utils.get_initial_box(obstacle_obj)
        # debug_line_render.draw_box(
        #     bbox.min, bbox.max, mn.Color4(255 / 255.0, 192 / 255.0, 0 / 255.0, 1.0)
        # )
        # debug_line_render.pop_transform()

        """获取以容器为中心的部分区域，即显示可达性"""
        # local_xzy = states_utils.transformation2robot_frame2(self.surround_loc, self._sim.robot.base_T)
        # self._sim.show_surrounding_reachablity(self.q, local_xzy)

        """获取以容器为中心的部分区域"""
        ret = self._sim.render(mode)
        self._sim.set_object_bb_draw(False, self.tgt_obj.object_id)

        """获取以物体为中心的部分区域"""
        """采用裁剪选择部分区域的方式"""
        # # 裁剪部分区域
        # # processed_points, processed_q = states_utils.process_points(self._sim.robot.base_pos, self.nav_goals, self.q, num_of_points=64)
        # #
        # # # 将nav_goals转到机器人的局部坐标系下进行测试转移得到的坐标是否正确
        # # local_xzy = states_utils.transformation2robot_frame2(processed_points, self._sim.robot.base_T)
        # #
        # # # 调试在局部坐标系下的坐标是否正确，这部分的q值其实不需要再进行归一化了
        # # self._sim.show_surrounding_reachablity(processed_q, local_xzy)
        #
        # # 将nav_goals转到机器人的局部坐标系下进行测试转移得到的坐标是否正确
        # local_xzy = states_utils.transformation2robot_frame2(self.nav_goals, self._sim.robot.base_T)
        #
        # # 调试在局部坐标系下的坐标是否正确
        # self._sim.show_surrounding_reachablity(self.q, local_xzy)
        #
        # ret = self._sim.render(mode)
        # self._sim.set_object_bb_draw(False, self.tgt_obj.object_id)
        """获取以物体为中心的部分区域"""

        return ret
