import numpy as np


def transform_points(points: np.ndarray, T: np.ndarray):
    """Transform points given a rigid transformation.

    Args:
        points: [..., 3]
        T: [4, 4]

    Returns:
        np.ndarray: [..., 3]
    """
    assert T.shape == (4, 4)
    return points @ T[:3, :3].T + T[:3, 3]


def invert_transformation(T: np.ndarray):
    """Invert rigid transformation.

    Args:
        T: [4, 4]

    Returns:
        np.ndarray: [4, 4]
    """
    assert T.shape == (4, 4)
    invT = np.eye(4, dtype=T.dtype)
    invT[:3, :3] = T[:3, :3].T
    invT[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return invT

# 将角度限制在-2*pi到2*pi的范围以内
def wrap_angle(angle):
    """Wrap angle in radians to [−pi pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
