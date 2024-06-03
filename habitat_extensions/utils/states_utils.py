import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from habitat_extensions.utils.transformations import *
from itertools import permutations
import magnum as mn
from shapely.geometry import Point, Polygon
from habitat_extensions.utils.scene_graph import SceneGraph

# 冰箱和抽屉在Replica中不提供bbox，可能是因为是.glb格式的文件，所以只能采用试的方法得到结果
FRIDGE_VIZ_BOX = mn.Range3D(mn.Vector3(-0.38, -0.95, -0.35), mn.Vector3(0.38, 0.95, 0.35))
DRAWER_VIZ_BOX = mn.Range3D(mn.Vector3(-0.48, 0.0, -1.5), mn.Vector3(0.15, 0.82, 1.55))

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
    elif flag == 3:
        points = np.ones([4, 3])
        points[0, :] = np.array([bbox_[0, 0], bbox_[0, 1], bbox_[0, 2]])
        points[1, :] = np.array([bbox_[1, 0], bbox_[0, 1], bbox_[0, 2]])
        points[2, :] = np.array([bbox_[1, 0], bbox_[0, 1], bbox_[1, 2]])
        points[3, :] = np.array([bbox_[0, 0], bbox_[0, 1], bbox_[1, 2]])

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

def transformation2global_frame(points, obj_tf):
    obj_tf = np.array(obj_tf)

    # 第一步：将bbox的坐标扩展为齐次坐标
    bbox_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # 第二步：将 bbox 的坐标从物品自身坐标系转换到全局坐标系，这一步验证过是正确的
    points_global = np.dot(obj_tf, bbox_homogeneous.T)[:3].T

    return points_global

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


def get_nearby_objects(container, tgt_T, rigid_objs):
    tgt_T = np.array(tgt_T)

    # 第一步是必须在容器的范围内
    initial_bbox = get_initial_box(container)

    # 第二步将物体从局部坐标系下转到世界坐标系下
    points = get_points(initial_bbox, flag=3)

    # 将bbox的坐标扩展为齐次坐标
    bbox_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    obj_tf = container.root_scene_node.transformation

    # 将 bbox 的坐标从物品自身坐标系转换到全局坐标系，这一步验证过是正确的
    points_global = np.dot(obj_tf, bbox_homogeneous.T)[:3].T

    rectangle = [(points_global[0, 0], points_global[0, 2]),
                 (points_global[1, 0], points_global[1, 2]),
                 (points_global[2, 0], points_global[2, 2]),
                 (points_global[3, 0], points_global[3, 2])]

    remove_keys = set()
    obj_list = []
    distances = {}
    x0 = tgt_T[0, 3]
    y0 = tgt_T[2, 3]

    poly = Polygon(rectangle)

    # 第二步是筛选三个最近的
    for key, value in rigid_objs.items():
        xzy = value.root_scene_node.translation
        point_to_check = (xzy[0], xzy[2])
        pt = Point(point_to_check)
        if not poly.contains(pt):
            continue
        distance = np.sqrt((xzy[0] - x0) ** 2 +
                           (xzy[2] - y0) ** 2)
        if distance < 0.1:
            continue
        distances[key] = distance

    # 选择最近的三个节点，如果没有三个就重复填充
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])

    # 获取距离最近的三个目标
    cnt = 0
    for i in range(len(sorted_distances)):
        # 可能是物体本身
        if sorted_distances[i][1] < 0.1:
            continue
        # 加入容器
        obj_list.append(rigid_objs[sorted_distances[i][0]])
        cnt += 1
        if cnt == 3:
            break

    # 如果不足的部分用容器本身进行填充
    for i in range(0, 3 - len(obj_list)):
        obj_list.append(container)

    obj_list.append(container)

    return obj_list, points_global


"""JL修改：这是一个将点从世界坐标系转换到机器人的局部坐标系下的函数"""
def transformation2robot_frame2(points_global, robot_tf):
    robot_tf = np.array(robot_tf)

    # 第一步：计算机器人坐标系到全局坐标系的逆变换矩阵
    inv_base_tf = np.linalg.inv(robot_tf)

    # 第二步：将 bbox 的坐标从全局坐标系转换到机器人的局部坐标系
    bbox_homogeneous2 = np.hstack((points_global, np.ones((points_global.shape[0], 1))))
    points_local = np.dot(inv_base_tf, bbox_homogeneous2.T)[:3].T

    return points_local


def min_max_normalize(q):
    # 使用 Min-Max 标准化方法将 q 值缩放到 [0, 1] 区间
    q_min = np.min(q)
    q_max = np.max(q)
    normalized_q = (q - q_min) / (q_max - q_min)
    return normalized_q


def shortest_reachability_area(robot_xyz, surrounding_area, q_val):
    # 标准化 q 值
    # normalized_q = min_max_normalize(q) * 1
    # q_val = 1 - normalized_q

    # 计算机器人当前位置与每个目标点的欧氏距离，其实应该考虑最近距离
    distances = np.linalg.norm(surrounding_area - robot_xyz, axis=1)
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    # print("在v1中最小距离为%.2f"%(min_distance))
    if min_distance > 0.45:
        # print("最小距离%.2f大于0.05"%(min_distance))
        # 设置权重
        w1, w2 = 0.8, 0.2
        # 根据标准化后的 q 值计算加权距离
        weighted_distances_total = w1 * distances + w2 * q_val
        # 取平均值
        weighted_distances = np.mean(weighted_distances_total)
    else:
        # print("最小距离%.2f小于0.05"%(q_val[min_index]))
        weighted_distances = q_val[min_index]

    # print("weighted_distances=%s"%(weighted_distances))

    # 返回加权距离列表或字典
    return weighted_distances

"""有一个问题在robot到达这个位置后不能很快的下降"""
def shortest_reachability_area_with_theta(robot_xyz, robot_ori, surrounding_area, target_T, q_val):
    # 标准化 q 值
    # normalized_q = min_max_normalize(q) * 1
    # # 将q值取反
    # q_val = 1 - normalized_q

    # 计算机器人当前位置与每个目标点的欧氏距离，点的话应该考虑最近的那个点
    distances = np.linalg.norm(surrounding_area - robot_xyz, axis=1)
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    # print("在v3中最小距离为%.2f" % (min_distance))
    # 在大于0.05范围时，其实只需要考虑xyz距离让机器人可以快点到达该范围内
    if min_distance > 0.45:
        # 设置权重
        w1, w2 = 0.8, 0.2
        # 根据标准化后的 q 值计算加权距离
        weighted_distances_total = w1 * distances + w2 * q_val
        # 取平均值
        weighted_distances = np.mean(weighted_distances_total)
    else: # 到达后只需要进行微调当前的角度值，以及到达q值相对更大的地方
        # 设置权重
        w2, w3 = 0.7, 0.3

        # 当前目标位置
        target_T = np.array(target_T)
        global_goal_pos = np.array(target_T[0:3, 3])
        # 获取当前期望的角度
        expected_theta = 2 * np.pi - np.arctan2(global_goal_pos[2] - robot_xyz[2], global_goal_pos[0] - robot_xyz[0])
        angle_diff = np.abs(robot_ori - expected_theta)
        angle_diff = 1 - np.cos(angle_diff)

        # 根据标准化后的 q 值计算加权距离
        # print("w2*%.2f=%.2f"%(q_val[min_index], w2 * q_val[min_index]))
        weighted_distances = w2 * q_val[min_index] + w3 * angle_diff

    # 返回加权距离列表或字典
    return weighted_distances


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
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle

# 将角度限制在 -π 到 π 之间
def limit_angle2(angle):
    while angle < 0:
        angle += 2 * np.pi
    while angle > 2 * np.pi:
        angle -= 2 * np.pi
    return angle

def get_initial_box(object_):
    initial_bbox = object_.root_scene_node.cumulative_bb
    if initial_bbox.min.sum() == 0:  # 这种一般是art_obj
        if object_.handle == "kitchen_counter_:0000":
            initial_bbox = DRAWER_VIZ_BOX
        elif object_.handle == "fridge_:0000":
            initial_bbox = FRIDGE_VIZ_BOX
    return initial_bbox

def get_normal_state(robot, target_T, obj_list, robot_tf=None, theta=None):
    """
    与rlmmbp中进行对应获取当前的状态
    Args:
        robot:机器人的obj
        target:目标物体的obj
        obj_list:当前其他物品的obj
        robot_tf:当前机器人的tf
        theta:当前机器人在世界坐标系下的角度
    Returns:
        返回获取的状态结果
    """
    total = len(obj_list)
    target_T = np.array(target_T)

    if robot_tf is None and theta is None:
        robot_tf = robot.base_T
        theta = robot.base_ori

    # 获取当前所有对象的bbox，这些获得的都不是在世界坐标系下的结果
    # 这里需要特别注意的一点是在habitat中是(x, z, y)而在Isaac-sim中是(x, y, z)
    curr_obj_bboxes = np.zeros([total, 6])
    for obj_num in range(total):
        # 需要先转移到世界坐标系下
        initial_bbox = get_initial_box(obj_list[obj_num])  # 获取当前物体的bbox
        obj_tf = obj_list[obj_num].root_scene_node.transformation  # 获取当前物体的tf矩阵
        obj_rad = obj_list[obj_num].root_scene_node.rotation.angle()  # 获取当前物体的朝向

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
    curr_goal_tf_xzy = np.dot(inv_base_tf, target_T)

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
    curr_goal_quat = uniform_quaternion(curr_goal_quat)

    # 获得最终状态
    state = np.hstack([curr_goal_pos, curr_goal_quat, curr_bboxes_flattened])

    return state

    # return curr_goal_pos, curr_goal_quat, curr_bboxes_flattened

"""JL修改：处理可以改变的区域"""
def process_points(robot_loc, surrounding_points, surrounding_q, num_of_points):
    n = surrounding_points.shape[0]
    if n > num_of_points:
        # 计算每个点到机器人的距离，并排序，机器人只看得到距离自己最近的点
        distances = np.linalg.norm(surrounding_points - robot_loc, axis=1)
        sorted_indices = np.argsort(distances)
        processed_points = surrounding_points[sorted_indices[:num_of_points]]
        processed_q = surrounding_q[sorted_indices[:num_of_points]]
    else:
        cur_n = n
        processed_points = surrounding_points
        processed_q = surrounding_q
        while cur_n < num_of_points:
            cur_num = min(n, num_of_points - cur_n)
            processed_points = np.vstack([processed_points, surrounding_points[:cur_num, :]])
            processed_q = np.hstack([processed_q, surrounding_q[:cur_num]])
            cur_n += cur_num

    return processed_points, processed_q


"""JL修改：角度采用朝向目标物体的角度，用于获取以机器人为中心的状态，局部坐标点的位置以及对应的动作"""
def get_round_states(robot, target_T, obj_list, resolution_dis=0.2, resolution_ang=10, point_loop=18, _action_xy_radius=1.0, _action_ang_lim=1.0):
    """

    Args:
        robot: 获取当前机器人的handle；
        target: 获取当前目标物品的handle；
        obj_list: 获取当前的所有其他物品的handle；
        resolution_dis: 当前的分辨率；
        resolution_ang: 当前的分辨角度；
        point_loop: 同一个角度下的重复此时；
        _action_xy_radius: 对xy范围进行缩放；
        _action_ang_lim: 对角度进行缩放；

    Returns:

    """
    """数据利用部分"""
    base_tf = np.array(robot.base_T)
    target_T = np.array(target_T)

    base_ori = robot.base_ori
    global_goal_pos = np.array(target_T[0:3, 3])

    """开始循环"""
    total_len = int(360 / resolution_ang * (1.3 - 0.3) / resolution_dis)
    # total_len = 3600 * 18
    states = np.zeros([total_len * point_loop, 31])  # 当前对应的观测状态
    round_actions = np.zeros([total_len * point_loop, 5])  # 当前对应的动作
    round_actions[:, 3] = 1
    curr_round_loc = np.zeros([total_len, 3])  # 当前实际的位姿
    curr_round_loc[:, 1] = 0.1  # 这里设置的画图部分的高度
    angle_gap = np.pi / 180  # 角度转弧度
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
            expected_theta = 2*np.pi - np.arctan2(global_goal_pos[2] - new_base_xy[2], global_goal_pos[0] - new_base_xy[0])

            for ang in range(-30, 31, 10):
                # TODO: 测试机器人应该朝向的角度，4.26调试通过，见视频：角度测试调试通过.mp4
                cur_theta = expected_theta + ang * angle_gap

                theta_scaled = np.array([cur_theta - base_ori])
                round_actions[cnt, 2] = theta_scaled / _action_ang_lim

                action_tf[:3, :3] = np.array([[np.cos(theta_scaled[0]), 0, np.sin(theta_scaled[0])],
                                              [0, 1, 0],
                                              [-np.sin(theta_scaled[0]), 0, np.cos(theta_scaled[0])]])
                # TODO: 4.26调试通过，可以实现机器人的tf坐标，机器人的坐标没有问题，见视频：物品位置调试通过1.mp4和目标物品位置调试通过.mp4
                new_base_tf = np.dot(base_tf, action_tf)

                # assert np.abs(limit_angle(np.arctan2(new_base_tf[0, 2], new_base_tf[0, 0])) - limit_angle(cur_theta)) < 0.1, \
                #     "(%s, %s)计算出的角度结果不相等为%s"%(np.arctan2(new_base_tf[0, 2], new_base_tf[0, 0]), cur_theta,
                #                                           np.abs(np.arctan2(new_base_tf[0, 2], new_base_tf[0, 0]) - cur_theta))

                states[cnt, :] = get_normal_state(robot, target_T, obj_list, new_base_tf, cur_theta)

                """检查机器人在当前位置和朝向下，能否将机器人运动过去"""
                cnt += 1

                # print("cnt=%d"%(cnt), end=',')

            cnt_single += 1

            # print("cnt_single=%d" % (cnt_single))

    # np.savetxt("debug_states/%d_states.txt"%(np.rad2deg(limit_angle2(robot.base_ori))), states, fmt='%.2f')
    # np.savetxt("debug_states/%d_curr_round_loc.txt"%(np.rad2deg(limit_angle2(robot.base_ori))), curr_round_loc, fmt='%.2f')
    # np.savetxt("debug_states/%d_round_actions.txt"%(np.rad2deg(limit_angle2(robot.base_ori))), round_actions, fmt='%.2f')

    return (states, curr_round_loc, round_actions)


"""JL修改：获得以物体为中心的可操作区域"""
def get_object_states(robot, target_T, obj_list, r_max=0.8, r_min=0.3, resolution=0.1, point_loop=18):
    """
    获取某个半径范围内的q值
    Args:
        robot: 获取当前机器人的handle；
        target: 获取当前目标物品的handle；
        obj_list: 获取当前容器的列表；
        r_max: 可达性评价的最远范围；
        r_min: 可达性评价的最近范围；
        resolution: 将区域离散化的分辨率；
        point_loop: 同一个角度下旋转的范围；

    Returns:

    """
    container = obj_list[-1]
    target_T = np.array(target_T)

    # 获取当前容器所在的中心点
    center = container.root_scene_node.translation
    center[1] = 0  # 将高度设为0

    # 获取当前容器的bbox
    initial_bbox = get_initial_box(container)
    length = initial_bbox.max[0] - initial_bbox.min[1]
    width = initial_bbox.max[2] - initial_bbox.min[2]
    height = initial_bbox.max[1] - initial_bbox.min[1]

    half_length = length / 2
    half_width = width / 2

    # 获取当前容器的yaw角度
    yaw = float(container.root_scene_node.rotation.angle())

    area_points = list()

    # 排除不需要的区域
    for j in np.arange(-half_length - r_max, half_length + r_max, resolution):
        for i in np.arange(-half_width - r_max, half_width + r_max, resolution):
            if -half_width - r_min < i < half_width + r_min and -half_length - r_min < j < half_length + r_min:
                continue
            area_points.append([j, 0, i])

    # 构建旋转矩阵
    rotation_matrix = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # 应用旋转矩阵
    rotated_corners = np.dot(area_points, rotation_matrix.T)

    # 平移到桌子中心
    surrounding_area = rotated_corners + np.array(center)

    # 获取当前的状态
    len_surrounding_area = len(surrounding_area)

    cnt = 0
    states = np.zeros([len_surrounding_area*point_loop, 31])  # 当前对应的观测状态
    surrounding_actions = np.zeros([len_surrounding_area*point_loop, 5])

    surrounding_actions[:, 3] = 1

    angle_gap = np.pi / 180

    # 获取世界坐标系下的绝对值
    global_goal_pos = np.array(target_T[0:3, 3])

    for i in range(len_surrounding_area):
        # 获取当前物体周围的坐标
        cur_loc = surrounding_area[i, :]

        # 获取当前期望的角度
        expected_theta = 2 * np.pi - np.arctan2(global_goal_pos[2] - cur_loc[2], global_goal_pos[0] - cur_loc[0])

        for ang in range(-27, 27, 3):
            cur_theta = expected_theta + ang * angle_gap

            # 获取当前所在位置的base_tf
            base_tf = np.zeros((4, 4))
            base_tf[:3, :3] = np.array([[np.cos(cur_theta), 0, np.sin(cur_theta)],
                                        [0, 1, 0],
                                        [-np.sin(cur_theta), 0, np.cos(cur_theta)]])
            base_tf[:, -1] = np.array([cur_loc[0], 0.0, cur_loc[2], 1.0])  # x,z,y,1

            states[cnt, :] = get_normal_state(robot, target_T, obj_list, base_tf, cur_theta)

            cnt += 1

    return states, surrounding_area, surrounding_actions


def get_object_states2(robot, target_T, obj_list, surrounding_area, point_loop=7):
    """
    获取某个目标附近的可达性图的Q值
    Args:
        robot: 获取当前机器人的handle；
        target: 获取当前目标物品的handle；
        obj_list: 获取当前容器的列表；
        surrounding_area: 当前需要评估的区域；
        point_loop: 同一个角度下旋转的范围；

    Returns:

    """
    container = obj_list[-1]
    target_T = np.array(target_T)

    # 获取当前的状态
    len_surrounding_area = len(surrounding_area)

    cnt = 0
    states = np.zeros([len_surrounding_area*point_loop, 31])  # 当前对应的观测状态
    surrounding_actions = np.zeros([len_surrounding_area*point_loop, 5])
    surrounding_theta = np.zeros([len_surrounding_area])

    surrounding_actions[:, 3] = 1

    angle_gap = np.pi / 180

    # 获取世界坐标系下的绝对值
    global_goal_pos = np.array(target_T[0:3, 3])

    for i in range(len_surrounding_area):
        # 获取当前物体周围的坐标
        cur_loc = surrounding_area[i, :]

        # 获取当前期望的角度
        expected_theta = 2 * np.pi - np.arctan2(global_goal_pos[2] - cur_loc[2], global_goal_pos[0] - cur_loc[0])

        surrounding_theta[i] = expected_theta

        for ang in range(-30, 31, 10):
            cur_theta = expected_theta + ang * angle_gap

            # 获取当前所在位置的base_tf
            base_tf = np.zeros((4, 4))
            base_tf[:3, :3] = np.array([[np.cos(cur_theta), 0, np.sin(cur_theta)],
                                        [0, 1, 0],
                                        [-np.sin(cur_theta), 0, np.cos(cur_theta)]])
            base_tf[:, -1] = np.array([cur_loc[0], 0.0, cur_loc[2], 1.0])  # x,z,y,1

            states[cnt, :] = get_normal_state(robot, target_T, obj_list, base_tf, cur_theta)

            cnt += 1

    return states, surrounding_area, surrounding_actions, surrounding_theta


def get_object_states_graph(robot, target_T, obj_list, surrounding_area, point_loop=7):
    """
    获取给定的坐标附近可达性图的q值
    Args:
        robot: 获取当前机器人的handle；
        target: 获取当前目标物品的handle；
        obj_list: 获取当前容器的列表；
        surrounding_area: 当前需要评估的区域；
        point_loop: 同一个角度下旋转的范围；

    Returns: graph构建好的图结果

    """
    container = obj_list[-1]
    target_T = np.array(target_T)

    # 获取当前的状态
    len_surrounding_area = len(surrounding_area)

    cnt = 0
    states = np.zeros([len_surrounding_area*point_loop, 8], dtype=object)  # 当前对应的观测状态
    surrounding_actions = np.zeros([len_surrounding_area*point_loop, 5])
    surrounding_theta = np.zeros([len_surrounding_area])

    print("Debug1:len_surrounding_area=%d, point_loop=%d"%(len_surrounding_area, point_loop))

    surrounding_actions[:, 3] = 1

    angle_gap = np.pi / 180

    # 获取世界坐标系下的绝对值
    global_goal_pos = np.array(target_T[0:3, 3])

    for i in range(len_surrounding_area):
        # 获取当前物体周围的坐标
        cur_loc = surrounding_area[i, :]

        # 获取当前期望的角度
        expected_theta = 2 * np.pi - np.arctan2(global_goal_pos[2] - cur_loc[2], global_goal_pos[0] - cur_loc[0])

        surrounding_theta[i] = expected_theta

        # 每个点都有18个角度范围
        for ang in range(-30, 31, 10):
            cur_theta = expected_theta + ang * angle_gap

            # 获取当前所在位置的base_tf
            base_tf = np.zeros((4, 4))
            base_tf[:3, :3] = np.array([[np.cos(cur_theta), 0, np.sin(cur_theta)],
                                        [0, 1, 0],
                                        [-np.sin(cur_theta), 0, np.cos(cur_theta)]])
            base_tf[:, -1] = np.array([cur_loc[0], 0.0, cur_loc[2], 1.0])  # x,z,y,1

            curr_goal_pos, curr_goal_quat, curr_bboxes_flattened = get_normal_state_graph(robot, target_T, obj_list, base_tf, cur_theta)
            obj_bboxes_indexes = [[0, 1, 2], [3], []]

            state_graph = SceneGraph(curr_goal_pos, curr_goal_quat,
                                 curr_bboxes_flattened, obj_bboxes_indexes).graph

            # hstack 并转换为 NumPy 数组
            curr_goal_pos_quat = np.hstack((curr_goal_pos, curr_goal_quat))

            # 组合数据
            arr = [curr_goal_pos_quat, state_graph]

            # 创建包含不同类型对象的NumPy数组
            """
            combined_arr=
            [[-0.23890377581119537 -2.258483409881592 0.7187395691871643
              -0.28392767906188965 -0.647599458694458 -0.28392767906188965
              0.647599458694458
              Graph(num_nodes=5, num_edges=4,
                    ndata_schemes={'h': Scheme(shape=(33,), dtype=torch.float32)}
                    edata_schemes={'rel_type': Scheme(shape=(), dtype=torch.float32), 'norm': Scheme(shape=(1,), dtype=torch.float32)})]]
            """
            combined_arr = np.array([np.hstack(arr)], dtype=object)

            # print("combined_arr=")
            # input(combined_arr)

            states[cnt, :] = combined_arr
            # print(states[cnt])
            # print(state_graph)
            # input("测试图数据为：")

            cnt += 1

    return states, surrounding_area, surrounding_actions, surrounding_theta


def get_normal_state_graph(robot, target_T, obj_list, robot_tf=None, theta=None):
    """
    与rlmmbp中进行对应获取当前的状态
    Args:
        robot:机器人的obj
        target:目标物体的obj
        obj_list:当前其他物品的obj
        robot_tf:当前机器人的tf
        theta:当前机器人在世界坐标系下的角度
    Returns:
        返回获取的状态结果
    """
    total = len(obj_list)
    target_T = np.array(target_T)

    if robot_tf is None and theta is None:
        robot_tf = robot.base_T
        theta = robot.base_ori

    # 获取当前所有对象的bbox，这些获得的都不是在世界坐标系下的结果
    # 这里需要特别注意的一点是在habitat中是(x, z, y)而在Isaac-sim中是(x, y, z)
    curr_obj_bboxes = np.zeros([total, 6])
    for obj_num in range(total):
        # 需要先转移到世界坐标系下
        initial_bbox = get_initial_box(obj_list[obj_num])  # 获取当前物体的bbox
        obj_tf = obj_list[obj_num].root_scene_node.transformation  # 获取当前物体的tf矩阵
        obj_rad = obj_list[obj_num].root_scene_node.rotation.angle()  # 获取当前物体的朝向

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
    curr_goal_tf_xzy = np.dot(inv_base_tf, target_T)

    # 获取当前可能的状态
    # curr_bboxes_flattened = curr_obj_bboxes.flatten()
    # 获取目标物品位置，注意这个是转换后的结果
    curr_goal_tf = transform_tf_matrix(curr_goal_tf_xzy)
    # TODO: 测试通过4.26，见视频：目标物品位置调试通过.mp4
    curr_goal_pos = curr_goal_tf[0:3, 3]
    # 当前的朝向，转到了与rlmmbp相同的坐标系下按理来说角度应该是
    # TODO：测试通过4.27，见视频：cur_quat调试通过.mp4
    curr_goal_quat = Rotation.from_matrix(curr_goal_tf[:3, :3]).as_quat()[[3, 0, 1, 2]]
    # 进行四元数的标准化
    curr_goal_quat = uniform_quaternion(curr_goal_quat)

    return curr_goal_pos, curr_goal_quat, curr_obj_bboxes


