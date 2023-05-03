import numpy as np
import torch

# reallybigname - auto-navigation functions in progress... 
# usage:
# if auto_rotation:
#    rot_mat = rotate_camera_towards_depth(depth_tensor, auto_rotation_steps, w, h, fov_deg, auto_rotation_depth_target)
def rotate_camera_towards_depth(depth_tensor, turn_weight, width, height, h_fov=60, target_depth=1):
    # Compute the depth at the target depth
    target_depth_index = int(target_depth * depth_tensor.shape[0])
    target_depth_values = depth_tensor[target_depth_index]
    max_depth_index = torch.argmax(target_depth_values).item()
    max_depth_index = (max_depth_index, target_depth_index)
    max_depth = target_depth_values[max_depth_index[0]].item()

    # Compute the normalized x and y coordinates
    x, y = max_depth_index
    x_normalized = (x / (width - 1)) * 2 - 1
    y_normalized = (y / (height - 1)) * 2 - 1

    # Calculate horizontal and vertical field of view (in radians)
    h_fov_rad = np.radians(h_fov)
    aspect_ratio = width / height
    v_fov_rad = h_fov_rad / aspect_ratio

    # Calculate the world coordinates (x, y) at the target depth
    x_world = np.tan(h_fov_rad / 2) * max_depth * x_normalized
    y_world = np.tan(v_fov_rad / 2) * max_depth * y_normalized

    # Compute the target position using the world coordinates and max_depth
    target_position = np.array([x_world, y_world, max_depth])

    # Assuming the camera is initially at the origin, and looking in the negative Z direction
    cam_position = np.array([0, 0, 0])
    current_direction = np.array([0, 0, -1])

    # Compute the direction vector and normalize it
    direction = target_position - cam_position
    direction = direction / np.linalg.norm(direction)

    # Compute the rotation angle based on the turn_weight (number of frames)
    axis = np.cross(current_direction, direction)
    axis = axis / np.linalg.norm(axis)
    angle = np.arcsin(np.linalg.norm(axis))
    max_angle = np.pi * (0.1 / turn_weight)  # Limit the maximum rotation angle to half of the visible screen
    rotation_angle = np.clip(np.sign(np.cross(current_direction, direction)) * angle / turn_weight, -max_angle, max_angle)

    # Compute the rotation matrix
    rotation_matrix = np.eye(3) + np.sin(rotation_angle) * np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ]) + (1 - np.cos(rotation_angle)) * np.outer(axis, axis)

    # Convert the NumPy array to a PyTorch tensor
    rotation_matrix_tensor = torch.from_numpy(rotation_matrix).float()

    # Add an extra dimension to match the expected shape (1, 3, 3)
    rotation_matrix_tensor = rotation_matrix_tensor.unsqueeze(0)

    return rotation_matrix_tensor

def rotation_matrix(axis, angle):
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
