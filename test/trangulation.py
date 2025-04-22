import numpy as np
import math

def coords_normalize(coords, width, height):
    return (coords[0] / width) * 2 - 1, (coords[1] / height) * 2 - 1


def calculate_3d_position(camera1_coords, camera2_coords, camera_separation,
                          h_fov, v_fov, surface_pitch, surface_yaw):
    x1, y1 = camera1_coords
    x2, y2 = camera2_coords
    sep_x, sep_y = camera_separation

    h_fov_rad = math.radians(h_fov)
    v_fov_rad = math.radians(v_fov)

    angle_x1 = x1 * (h_fov_rad / 2)
    angle_y1 = y1 * (v_fov_rad / 2)
    angle_x2 = x2 * (h_fov_rad / 2)
    angle_y2 = y2 * (v_fov_rad / 2)

    dir1 = np.array([
        math.tan(angle_x1),
        math.tan(angle_y1),
        1.0
    ])

    dir2 = np.array([
        math.tan(angle_x2),
        math.tan(angle_y2),
        1.0
    ])

    cam1_pos = np.array([-sep_x / 2, -sep_y / 2, 0.0])
    cam2_pos = np.array([sep_x / 2, sep_y / 2, 0.0])

    a = np.column_stack((dir1, -dir2))
    b = cam2_pos - cam1_pos

    try:
        t1, t2 = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        print("kys")
        return None, None, None

    point1 = cam1_pos + t1 * dir1
    point2 = cam2_pos + t2 * dir2
    local_position = (point1 + point2) / 2

    pitch_rad = math.radians(surface_pitch)
    yaw_rad = math.radians(surface_yaw)

    pitch_matrix = np.array([
        [1, 0, 0],
        [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
        [0, math.sin(pitch_rad), math.cos(pitch_rad)]
    ])

    yaw_matrix = np.array([
        [math.cos(yaw_rad), 0, math.sin(yaw_rad)],
        [0, 1, 0],
        [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]
    ])

    rotation_matrix = np.dot(yaw_matrix, pitch_matrix)
    global_position = np.dot(rotation_matrix, local_position)

    return global_position


c1 = coords_normalize((320, 25), 800, 600)
c2 = coords_normalize((296, 64), 800, 600)
print(calculate_3d_position(c1, c2, (1, 1),
                              53, 45,
                              0, 0))
