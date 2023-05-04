import os
import numpy as np
import torch
from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize
from plyfile import PlyData, PlyElement


def init_image_coor(height, width, u0=None, v0=None):
    u0 = width / 2.0 if u0 is None else u0
    v0 = height / 2.0 if v0 is None else v0

    x_row = np.arange(0, width)
    x = np.tile(x_row, (height, 1))
    x = x.astype(np.float32)
    u_u0 = x - u0

    y_col = np.arange(0, height)
    y = np.tile(y_col, (width, 1)).T
    y = y.astype(np.float32)
    v_v0 = y - v0
    return u_u0, v_v0

def depth_to_pcd(depth, u_u0, v_v0, f, invalid_value=0):
    mask_invalid = depth <= invalid_value
    depth[mask_invalid] = 0.0
    x = u_u0 / f * depth
    y = v_v0 / f * depth
    z = depth
    pcd = np.stack([x, y, z], axis=2)
    return pcd, ~mask_invalid

def pcd_to_sparsetensor(pcd, mask_valid, voxel_size=0.01, num_points=100000):
    pcd_valid = pcd[mask_valid]
    block_ = pcd_valid
    block = np.zeros_like(block_)
    block[:, :3] = block_[:, :3]

    pc_ = np.round(block_[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)
    feat_ = block

    # transfer point cloud to voxels
    inds = sparse_quantize(pc_,
                           feat_,
                           return_index=True,
                           return_invs=False)
    if len(inds) > num_points:
        inds = np.random.choice(inds, num_points, replace=False)

    pc = pc_[inds]
    feat = feat_[inds]
    lidar = SparseTensor(feat, pc)
    feed_dict = [{'lidar': lidar}]
    inputs = sparse_collate_fn(feed_dict)
    return inputs

def pcd_uv_to_sparsetensor(pcd, u_u0, v_v0, mask_valid, f= 500.0, voxel_size=0.01, mask_side=None, num_points=100000):
    if mask_side is not None:
        mask_valid = mask_valid & mask_side
    pcd_valid = pcd[mask_valid]
    u_u0_valid = u_u0[mask_valid][:, np.newaxis] / f
    v_v0_valid = v_v0[mask_valid][:, np.newaxis] / f

    block_ = np.concatenate([pcd_valid, u_u0_valid, v_v0_valid], axis=1)
    block = np.zeros_like(block_)
    block[:, :] = block_[:, :]


    pc_ = np.round(block_[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)
    feat_ = block

    # transfer point cloud to voxels
    inds = sparse_quantize(pc_,
                           feat_,
                           return_index=True,
                           return_invs=False)
    if len(inds) > num_points:
        inds = np.random.choice(inds, num_points, replace=False)

    pc = pc_[inds]
    feat = feat_[inds]
    lidar = SparseTensor(feat, pc)
    feed_dict = [{'lidar': lidar}]
    inputs = sparse_collate_fn(feed_dict)
    return inputs


def refine_focal_one_step(depth, focal, model, u0, v0):
    # reconstruct PCD from depth
    u_u0, v_v0 = init_image_coor(depth.shape[0], depth.shape[1], u0=u0, v0=v0)
    pcd, mask_valid = depth_to_pcd(depth, u_u0, v_v0, f=focal, invalid_value=0)
    # input for the voxelnet
    feed_dict = pcd_uv_to_sparsetensor(pcd, u_u0, v_v0, mask_valid, f=focal, voxel_size=0.005, mask_side=None)
    inputs = feed_dict['lidar'].cuda()

    outputs = model(inputs)
    return outputs

def refine_shift_one_step(depth_wshift, model, focal, u0, v0):
    # reconstruct PCD from depth
    u_u0, v_v0 = init_image_coor(depth_wshift.shape[0], depth_wshift.shape[1], u0=u0, v0=v0)
    pcd_wshift, mask_valid = depth_to_pcd(depth_wshift, u_u0, v_v0, f=focal, invalid_value=0)
    # input for the voxelnet
    feed_dict = pcd_to_sparsetensor(pcd_wshift, mask_valid, voxel_size=0.01)
    inputs = feed_dict['lidar'].cuda()

    outputs = model(inputs)
    return outputs

def refine_focal(depth, focal, model, u0, v0):
    last_scale = 1
    focal_tmp = np.copy(focal)
    for i in range(1):
        scale = refine_focal_one_step(depth, focal_tmp, model, u0, v0)
        focal_tmp = focal_tmp / scale.item()
        last_scale = last_scale * scale
    return torch.tensor([[last_scale]])

def refine_shift(depth_wshift, model, focal, u0, v0):
    depth_wshift_tmp = np.copy(depth_wshift)
    last_shift = 0
    for i in range(1):
        shift = refine_shift_one_step(depth_wshift_tmp, model, focal, u0, v0)
        shift = shift if shift.item() < 0.7 else torch.tensor([[0.7]])
        depth_wshift_tmp -= shift.item()
        last_shift += shift.item()
    return torch.tensor([[last_shift]])

def reconstruct_3D(depth, f):
    """
    Reconstruct depth to 3D pointcloud with the provided focal length.
    Return:
        pcd: N X 3 array, point cloud
    """
    cu = depth.shape[1] / 2
    cv = depth.shape[0] / 2
    width = depth.shape[1]
    height = depth.shape[0]
    row = np.arange(0, width, 1)
    u = np.array([row for i in np.arange(height)])
    col = np.arange(0, height, 1)
    v = np.array([col for i in np.arange(width)])
    v = v.transpose(1, 0)

    if f > 1e5:
        print('Infinit focal length!!!')
        x = u - cu
        y = v - cv
        z = depth / depth.max() * x.max()
    else:
        x = (u - cu) * depth / f
        y = (v - cv) * depth / f
        z = depth

    x = np.reshape(x, (width * height, 1)).astype(np.float)
    y = np.reshape(y, (width * height, 1)).astype(np.float)
    z = np.reshape(z, (width * height, 1)).astype(np.float)
    pcd = np.concatenate((x, y, z), axis=1)
    pcd = pcd.astype(np.int)
    return pcd

def save_point_cloud(pcd, rgb, filename, binary=True):
    """Save an RGB point cloud as a PLY file.

    :paras
      @pcd: Nx3 matrix, the XYZ coordinates
      @rgb: NX3 matrix, the rgb colors for each 3D point
    """
    assert pcd.shape[0] == rgb.shape[0]

    if rgb is None:
        gray_concat = np.tile(np.array([128], dtype=np.uint8), (pcd.shape[0], 3))
        points_3d = np.hstack((pcd, gray_concat))
    else:
        points_3d = np.hstack((pcd, rgb))
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

         # Write
        PlyData([el]).write(filename)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property uchar red\n' \
                   'property uchar green\n' \
                   'property uchar blue\n' \
                   'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(filename, np.column_stack((x, y, z, r, g, b)), fmt="%d %d %d %d %d %d", header=ply_head, comments='')

def reconstruct_depth(depth, rgb, dir, pcd_name, focal):
    """
    para disp: disparity, [h, w]
    para rgb: rgb image, [h, w, 3], in rgb format
    """
    rgb = np.squeeze(rgb)
    depth = np.squeeze(depth)

    mask = depth < 1e-8
    depth[mask] = 0
    depth = depth / depth.max() * 10000

    pcd = reconstruct_3D(depth, f=focal)
    rgb_n = np.reshape(rgb, (-1, 3))
    save_point_cloud(pcd, rgb_n, os.path.join(dir, pcd_name + '.ply'))


def recover_metric_depth(pred, gt):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) & (pred > 1e-8)

    gt_mask = gt[mask]
    pred_mask = pred[mask]
    a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    pred_metric = a * pred + b
    return pred_metric
