import argparse
import time

from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_extensions.tasks.rearrange.env import RearrangeRLEnv
from habitat_extensions.utils.viewer import OpenCVViewer
from mobile_manipulation.config import Config, get_config, load_config
from mobile_manipulation.utils.common import (
    extract_scalars_from_info,
    get_flat_space_names,
)
from mobile_manipulation.utils.wrappers import HabitatActionWrapperV1

import os
import magnum as mn

os.chdir('/home/lu/Desktop/embodied_ai/hab-mobile-manipulation')

# 从键盘输入获取动作
def get_action_from_key(key, action_name):
    if "BaseArmGripperAction" in action_name:
        # 底盘动作是二维
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
        # 机械臂动作是三维
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

        # 夹爪动作是实数
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
        if key == "w":  # 前进
            base_action = [1, 0]
        elif key == "s":  # 后退
            base_action = [-1, 0]
        elif key == "a":  # 左转
            base_action = [0, 1]
        elif key == "d":  # 右转
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
        if key == "w":  # 前进
            base_action = [1, 0]
        elif key == "s":  # 后退
            base_action = [-1, 0]
        elif key == "a":  # 左转
            base_action = [0, 1]
        elif key == "d":  # 右转
            base_action = [0, -1]
        else:
            base_action = [0, 0]
        if key == "z":  # 设置停止标志位
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

# 获取环境的配置
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

# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        dest="config_path",
        type=str,
        default="/home/lu/Desktop/embodied_ai/hab-mobile-manipulation/configs/rearrange/tasks/play.yaml",
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
    # print(config)  # 这里的config和文件中读取到的有些不同
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
    # 重新修改配置文件中的config.RL.ACTION_NAME
    if args.action:
        config.defrost()
        config.RL.ACTION_NAME = args.action
        config.freeze()

    print("config=%s" % (config))

    # -------------------------------------------------------------------------- #
    # Env
    # -------------------------------------------------------------------------- #
    # 初始化深度学习环境，作者自定义的baseline_registry环境
    env_cls = baseline_registry.get_env(config.ENV_NAME)
    # 调用habitat_extensions/tasks/rearrange/env.py的部分
    env: RearrangeRLEnv = env_cls(config)
    env = HabitatActionWrapperV1(env)  # 创建一个环境实例 (env)，类型为RearrangeRLEnv
    print("obs_space", env.observation_space)
    print("action_space", env.action_space)
    state_keys = get_flat_space_names(env.observation_space)

    def reset():
        obs = env.reset()  # 重置环境，作者自定义的与环境交互函数
        info = {}
        print("episode_id", env.habitat_env.current_episode.episode_id)
        print("scene_id", env.habitat_env.current_episode.scene_id)
        return obs, info

    env.seed(0)
    # 第一步：重置环境
    obs, info = reset()
    for k, v in obs.items():
        print(k, v.shape)
    viewer = OpenCVViewer(config.ENV_NAME)  # 显示采用的是opencv，其实就是采用opencv进行显示

    while True:  # 作用是逐帧进行显示
        metrics = extract_scalars_from_info(info)
        # 在环境中逐帧进行显示
        rendered_frame = env.render(info=metrics, overlay_info=False)  # 返回当前的画面其实rendered_frame就是一个numpy数组
        # print("metrices=")
        # print(metrics)
        # print("rendered_frame=")
        # print(type(rendered_frame))
        # print("rendered_frame.shape=")
        # print(rendered_frame.shape)

        key = viewer.imshow(rendered_frame)  # 仅仅用于显示渲染画面

        if key == "r":  # Press r to reset env
            obs, info = reset()
            continue

        if args.random_action:
            action = env.action_space.sample()
        else:
            # Please refer to this function for keyboard-action mapping
            action = get_action_from_key(key, config.RL.ACTION_NAME)

        # 完成动作
        """训练场景的数据
        obs = {'arm_joint_pos': array([-0.21390784, -1.1940902 ,  0.411061  ,  0.98038083,  0.08839583, 1.9694287 ,  0.07698423], dtype=float32), 
        'gripper_pos_at_base': array([ 0.52818346,  1.0553488 , -0.03550023], dtype=float32), 'is_grasped': array([0.], dtype=float32), 
        'pick_goal_at_gripper': array([ 0.11108065, -0.77064246,  0.03606964], dtype=float32), 
        'pick_goal_at_base': array([0.62823963, 0.76592827, 0.68126607], dtype=float32)}
        action = {'action': 'BaseArmGripperAction', 
                  'action_args': {'base_action': [0, 1], 
                  'arm_action': [0.0, 0.0, 0.0], 
                  'gripper_action': 0.0}}
        reward = 0.0
        info = {'gripper_to_obj_dist': 0.7794372, 
                'gripper_to_resting_dist': 0.071540624, 
                'gripper_status': {'not_holding': 7, 'pick_correct': 0, 'pick_wrong': 0, 'holding_correct': 0, 'holding_wrong': 0, 'drop': 0}, 
                'robot_force': 0.0, 
                'force_penalty': -0.0, 
                'invalid_grasp_penalty': 0.0, 
                'rearrange_pick_success': False, 
                'rearrange_pick_reward': -0.09600000083446503, 
                'is_episode_active': True, 
                'is_episode_truncated': False, 
                'elapsed_steps': 7}
        """
        obs, reward, done, info = env.step(action)  # 返回奖励等数据

        """在play.py中实现画圆"""
        # for i in range(1000):
        #     # 在当前机器人附近画圆
        #     debug_line_render = env.habitat_env.sim.get_debug_line_render()
        #     debug_line_render.set_line_width(2.0)
        #     agent_translation = mn.Vector3(env.habitat_env.sim.robot.base_T.translation)
        #     agent_translation[1] = -1.5
        #     # print("translation=%s"%(env.habitat_env.sim.robot.base_T.translation))
        #     debug_line_render.draw_circle(agent_translation, 1.0, mn.Color4(1.0, 0.0, 0.0, 1.0))
        #
        #     agent_viz_box = mn.Range3D(mn.Vector3(-0.1, 0.0, -0.1), mn.Vector3(0.1, 0.4, 0.1))
        #     debug_line_render.draw_box(
        #         agent_viz_box.min, agent_viz_box.max, mn.Color4(1.0, 0.0, 0.0, 1.0)
        #     )
        #     debug_line_render.push_transform(env.habitat_env.sim.robot.base_T)
        #
        #     debug_line_render.pop_transform()
        #     time.sleep(0.0001)

        # 打印当前robot的位置
        # print("env.habitat_env.sim.robot.base_T=%s, \n env.habitat_env.sim.robot.base_pos=%s, \n env.habitat_env.sim.robot.base_ori=%s"
        #       %(env.habitat_env.sim.robot.base_T, env.habitat_env.sim.robot.base_pos, env.habitat_env.sim.robot.base_ori))

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
    main()
