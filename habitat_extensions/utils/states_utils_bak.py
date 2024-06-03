import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from habitat_extensions.utils.transformations import *
from itertools import permutations
import magnum as mn

# 冰箱和抽屉在Replica中不提供bbox，可能是因为是.glb格式的文件，所以只能采用试的方法得到结果
FRIDGE_VIZ_BOX = mn.Range3D(mn.Vector3(-0.38, -0.95, -0.35), mn.Vector3(0.38, 0.95, 0.35))
DRAWER_VIZ_BOX = mn.Range3D(mn.Vector3(-0.48, 0.0, -1.5), mn.Vector3(0.15, 0.82, 1.5))

"""JL修改：这是一个简单的根据bbox获得其他点的函数，注意这个函数仅仅适用于bbox在原点且无任何旋转的情况下，否则加入矩形旋转了45度，需要考虑旋转获得当前点"""
def get_points(bbox, flag=2):
    bbox_ = np.zeros([2, 3])
    bbox_[0, :] = np.array(bbox.min)
    bbox_[1, :] = np.array(bbox.max)

    # 将bbox都转换为点
    if flag == 1:
        points = np.ones([8, 3])
        points[0, :] = np.array([bbox_[0, 0], bbox_[0, 1], bbox_[0, 2]])
        points[1, :] = np.array([bbox_[0, 0], bbox_[1, 1], bbox_[0, 2]])
        points[2, :] = np.array([bbox_[1, 0], bbox_[0, 1], bbox_[0, 2]])
        points[3, :] = np.array([bbox_[1, 0], bbox_[1, 1], bbox_[0, 2]])
        points[4, :] = np.array([bbox_[0, 0], bbox_[0, 1], bbox_[1, 2]])
        points[5, :] = np.array([bbox_[0, 0], bbox_[1, 1], bbox_[1, 2]])
        points[6, :] = np.array([bbox_[1, 0], bbox_[0, 1], bbox_[1, 2]])
        points[7, :] = np.array([bbox_[1, 0], bbox_[1, 1], bbox_[1, 2]])
    elif flag == 2:
        points = np.ones([2, 3])
        points[0, :] = np.array([bbox_[0, 0], bbox_[0, 1], bbox_[0, 2]])
        points[1, :] = np.array([bbox_[1, 0], bbox_[1, 1], bbox_[1, 2]])

    return points

"""JL修改：检查转换后的点的坐标和朝向是否正确"""
def test_pos_quat(obj_tf, robot_tf):
    obj_tf = np.array(obj_tf)
    robot_tf = np.array(robot_tf)

    inv_robot_tf = np.linalg.inv(robot_tf)
    curr_goal_tf = np.dot(inv_robot_tf, obj_tf)

    curr_goal_pos = curr_goal_tf[0:3, 3]

    curr_goal_quat = Rotation.from_matrix(curr_goal_tf[:3, :3]).as_quat()[[2, 1, 3, 0]]
    # eular = self.quaternion_to_angle(curr_goal_quat)
    eular = euler_from_matrix(curr_goal_tf, "syzx")

    """调试角度是否正确部分"""
    """
    # 给定的列表
    lst = [0, 1, 2, 3]

    # 获取所有可能的排列
    perms = permutations(lst)

    eular = euler_from_matrix(curr_goal_tf, "syzx")
    quat = quaternion_from_euler(eular[0], eular[1], eular[2], "syzx")
    print("============================================================================")
    # 将排列打印出来，可能的组合有[0, 1, 3, 2],[1, 0, 2, 3],[1, 2, 0, 3], [2, 1, 3, 0]
    for perm in perms:
        test_goal_quat = curr_goal_quat[[perm]]
        eular2 = self.quaternion_to_angle(test_goal_quat)
        if np.abs(eular2 - eular[0]) < 0.1:
            print(perm)
            print("eular=%s, eular2=%s"%(str(eular), str(eular2)))
            print("quat=%s, quat2=%s， curr_goal_quat=%s"%(str(quat), str(test_goal_quat), str(curr_goal_quat)))
    """


    return curr_goal_pos, curr_goal_quat, eular[0]

"""JL修改：将四元数转为角度"""
def quaternion_to_angle(quaternion):
    rotation = Rotation.from_quat(quaternion)
    # yxz可行
    euler_angles = rotation.as_euler('xzy')
    # angle_rad = 2*np.pi - euler_angles[0]
    angle_rad = euler_angles[0]
    angle_rad = -angle_rad + np.pi
    angle_rad = angle_rad % (2 * np.pi)
    if angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    elif angle_rad < -np.pi:
        angle_rad += 2 * np.pi
    return angle_rad

"""JL修改：这是一个将点从物品坐标系转换到机器人的局部坐标系下的函数"""
def transformation2robot_frame(points, obj_tf, robot_tf):
    obj_tf = np.array(obj_tf)
    robot_tf = np.array(robot_tf)

    # 第一步：将bbox的坐标扩展为齐次坐标
    bbox_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # 第二步：将 bbox 的坐标从物品自身坐标系转换到全局坐标系，这一步验证过是正确的
    points_global = np.dot(obj_tf, bbox_homogeneous.T)[:3].T

    # 第三步：计算机器人坐标系到全局坐标系的逆变换矩阵
    inv_base_tf = np.linalg.inv(robot_tf)

    # 第四步：将 bbox 的坐标从全局坐标系转换到机器人的局部坐标系
    bbox_homogeneous2 = np.hstack((points_global, np.ones((points.shape[0], 1))))
    points_local = np.dot(inv_base_tf, bbox_homogeneous2.T)[:3].T

    return points_local

# 进行tf矩阵之间的转换
def transform_tf_matrix(matrix_xzy):
    matrix_xyz = matrix_xzy.copy()
    # 更改角度的坐标
    cos_theta = matrix_xyz[0, 0]
    sin_theta = matrix_xyz[0, 2]
    matrix_xyz[:3, :3] = np.array([[cos_theta, -sin_theta, 0],
                                   [sin_theta, cos_theta, 0],
                                   [0, 0, 1]])
    # 更改x, y, z部分的坐标
    matrix_xyz[1, 3] = matrix_xzy[2, 3]
    matrix_xyz[2, 3] = matrix_xzy[1, 3]
    return matrix_xyz

# 标准化四元数
def uniform_quaternion(quat):
    # JL:修改标准化四元数
    quat /= np.linalg.norm(quat)
    if quat[0] < 0:
        quat *= -1
    return quat

# 将角度限制在 -π 到 π 之间
def limit_angle(angle):
    while angle < -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle

def get_normal_state(robot, target, obj_list, robot_tf=None, theta=None):
    """
    与rlmmbp中进行对应获取当前的状态
    Args:
        robot:机器人的obj
        target:目标物体的obj
        obj_list:当前其他物品的obj

    Returns:
        返回获取的状态结果
    """
    total = len(obj_list)

    if robot_tf is None and theta is None:
        robot_tf = robot.base_T
        theta = robot.base_ori

    debug_points = []

    # 获取当前所有对象的bbox，这些获得的都不是在世界坐标系下的结果
    # 这里需要特别注意的一点是在habitat中是(x, z, y)而在Isaac-sim中是(x, y, z)
    curr_obj_bboxes = np.zeros([total, 6])
    for obj_num in range(total):
        # 需要先转移到世界坐标系下
        initial_bbox = obj_list[obj_num].root_scene_node.cumulative_bb  # 获取当前物体的bbox
        obj_tf = obj_list[obj_num].root_scene_node.transformation  # 获取当前物体的tf矩阵
        obj_rad = obj_list[obj_num].root_scene_node.rotation.angle()  # 获取当前物体的朝向

        if initial_bbox.min.sum() == 0:  # 这种一般是art_obj
            if obj_list[obj_num].handle == "kitchen_counter_:0000":
                initial_bbox = DRAWER_VIZ_BOX
            elif obj_list[obj_num].handle == "fridge_:0000":
                initial_bbox = FRIDGE_VIZ_BOX
        cur_bbox = transformation2robot_frame(get_points(initial_bbox), obj_tf, robot_tf)
        # TODO: 0-4维记录bbox，测试通过4.26，见视频：物品位置调试通过1.mp4
        curr_obj_bboxes[obj_num, 0:4] = np.array([cur_bbox[0, 0], cur_bbox[0, 2], cur_bbox[1, 0], cur_bbox[1, 2]])
        # TODO: 第4维记录高度，测试通过4.26，见视频：物品位置调试通过1.mp4
        curr_obj_bboxes[obj_num, 4] = cur_bbox[1, 1]
        # 第5维记录角度，这里最好做一个限制角度因为对于rl而言角度不同其实观测也是不同的，直接采用绝对坐标系下的值进行相减即可
        curr_obj_bboxes[obj_num, 5] = limit_angle(float(obj_rad) - theta)

    # 获取目标物体的tf和bbox
    inv_base_tf = np.linalg.inv(robot_tf)
    # 获取当前target的tf，在局部坐标系下
    curr_goal_tf_xzy = np.dot(inv_base_tf, target.root_scene_node.transformation)

    # 获取当前可能的状态
    curr_bboxes_flattened = curr_obj_bboxes.flatten()
    # 获取目标物品位置，注意这个是转换后的结果
    curr_goal_tf = transform_tf_matrix(curr_goal_tf_xzy)
    # TODO: 测试通过4.26，见视频：目标物品位置调试通过.mp4
    curr_goal_pos = curr_goal_tf[0:3, 3]
    # 当前的朝向，转到了与rlmmbp相同的坐标系下按理来说角度应该是
    # TODO：测试通过4.27，见视频：cur_quat调试通过.mp4
    curr_goal_quat = Rotation.from_matrix(curr_goal_tf[:3, :3]).as_quat()[[3, 0, 1, 2]]
    # 进行四元数的标准化
    # curr_goal_quat = uniform_quaternion(curr_goal_quat)

    """这是一段调试代码"""
    debug_points.append(curr_goal_pos[[0, 2, 1]])
    # 这个计算的角度仍然有些不正确，变成了机器人-全局目标下物体相反了刚好
    debug_rad = quaternion_to_angle(curr_goal_quat)
    # TODO：直接计算的话是正确的，4.26测试通过，就是用当前全局坐标系下目标物体的朝向-机器人真实的朝向就是局部坐标系下物体的角度
    # debug_rad = np.arctan2(curr_goal_tf[1, 0], curr_goal_tf[0, 0])

    # 获得最终状态
    state = np.hstack([curr_goal_pos, curr_goal_quat, curr_bboxes_flattened])

    return state, debug_points, debug_rad


"""JL修改：角度采用朝向目标物体的角度，用于获取以机器人为中心的状态，局部坐标点的位置以及对应的动作"""
def get_round_states(robot, target, obj_list, resolution_dis=0.2, resolution_ang=10, point_loop=18, _action_xy_radius=1.0, _action_ang_lim=1.0):
    """数据利用部分"""
    base_tf = np.array(robot.base_T)
    base_ori = robot.base_ori
    global_goal_pos = np.array(target.root_scene_node.translation)

    """开始循环"""
    total_len = int(360 / resolution_ang * (1.3 - 0.3) / resolution_dis)
    # total_len = 3600 * 18
    states = np.zeros([total_len * point_loop, 31])  # 当前对应的观测状态
    round_actions = np.zeros([total_len * point_loop, 5])  # 当前对应的动作
    round_actions[:, 3] = 1
    curr_round_loc = np.zeros([total_len, 3])  # 当前实际的位姿
    curr_round_loc[:, 1] = 0.1  # 这里设置的画图部分的高度
    angle_gap = math.pi / 180  # 角度转弧度
    cnt = 0
    cnt_single = 0

    for phi_cnt in range(0, 360, resolution_ang):
        for r in np.arange(0.3, 1.3, resolution_dis):
            phi = phi_cnt * angle_gap

            round_actions[cnt:cnt+point_loop, 0] = r / _action_xy_radius
            round_actions[cnt:cnt+point_loop, 1] = phi / _action_ang_lim

            x_scaled = np.array([r * np.cos(phi)])
            y_scaled = np.array([r * np.sin(phi)])

            # Transform actions to world frame and apply to base
            # 将动作转换为世界坐标系并应用到基座上。根据动作的旋转角度（theta_scaled[0]）设置旋转矩阵的旋转部分，并将动作的位移设置为变换矩阵的最后一列
            action_tf = np.zeros((4, 4))

            action_tf[2, 2] = 1.0  # No rotation here
            action_tf[:, -1] = np.array([x_scaled[0], 0.0, y_scaled[0], 1.0])  # x,z,y,1
            # 计算应用动作后的新基座的变换矩阵。将基座的位移提取出来作为新的基座XY坐标，并计算新的基座角度
            new_base_tf = np.dot(base_tf, action_tf)
            new_base_xy = new_base_tf[0:3, 3]

            curr_round_loc[cnt_single, [0, 2]] = new_base_xy[[0, 2]]

            # 在世界坐标系下目前想要的角度为，在Isaac-sim的坐标系下是正确的，但是转换到habitat中其实是360 - theta
            # 当前的朝向，如何到达期望的角度，这个角度其实应该是相反的，TODO: 4.26调试通过，见视频：角度测试调试通过.mp4
            expected_theta = 2*np.pi - math.atan2(global_goal_pos[2] - new_base_xy[2], global_goal_pos[0] - new_base_xy[0])

            for ang in range(-90, 90, 10):
                # TODO: 测试机器人应该朝向的角度，4.26调试通过，见视频：角度测试调试通过.mp4
                cur_theta = expected_theta + ang * angle_gap

                theta_scaled = np.array([cur_theta - base_ori])
                round_actions[cnt, 2] = theta_scaled / _action_ang_lim

                action_tf[:3, :3] = np.array([[np.cos(theta_scaled[0]), 0, np.sin(theta_scaled[0])],
                                              [0, 1, 0],
                                              [-np.sin(theta_scaled[0]), 0, np.cos(theta_scaled[0])]])
                # TODO: 4.26调试通过，可以实现机器人的tf坐标，机器人的坐标没有问题，见视频：物品位置调试通过1.mp4和目标物品位置调试通过.mp4
                new_base_tf = np.dot(base_tf, action_tf)

                states[cnt, :] = get_normal_state(robot, target, obj_list, new_base_tf, cur_theta)

                """检查机器人在当前位置和朝向下，能否将机器人运动过去"""
                cnt += 1

                # print("cnt=%d"%(cnt), end=',')

            cnt_single += 1

            # print("cnt_single=%d" % (cnt_single))

    np.savetxt("debug_states/%d_states.txt"%(np.rad2deg(limit_angle2(robot.base_ori))), states, fmt='%.2f')
    np.savetxt("debug_states/%d_curr_round_loc.txt"%(np.rad2deg(limit_angle2(robot.base_ori))), curr_round_loc, fmt='%.2f')
    np.savetxt("debug_states/%d_round_actions.txt"%(np.rad2deg(limit_angle2(robot.base_ori))), round_actions, fmt='%.2f')

    return (states, curr_round_loc, round_actions)

    return (states, curr_round_loc, round_actions, curr_expected_theta,
            debug_robot_locations, debug_points, debug_robot_transformed, debug_cur_theta, debug_rads)


