import numpy as np
import cv2
from functools import reduce
import math
import py3d_tools as p3d
import torch
from einops import rearrange
from .prompt import check_is_number

# Webui
from modules.shared import state, opts

DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample

def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)

def construct_RotationMatrixHomogenous(rotation_angles):
    assert(type(rotation_angles)==list and len(rotation_angles)==3)
    RH = np.eye(4,4)
    cv2.Rodrigues(np.array(rotation_angles), RH[0:3, 0:3])
    return RH

# https://en.wikipedia.org/wiki/Rotation_matrix
def getRotationMatrixManual(rotation_angles):
	
    rotation_angles = [np.deg2rad(x) for x in rotation_angles]
    
    phi         = rotation_angles[0] # around x
    gamma       = rotation_angles[1] # around y
    theta       = rotation_angles[2] # around z
    
    # X rotation
    Rphi        = np.eye(4,4)
    sp          = np.sin(phi)
    cp          = np.cos(phi)
    Rphi[1,1]   = cp
    Rphi[2,2]   = Rphi[1,1]
    Rphi[1,2]   = -sp
    Rphi[2,1]   = sp
    
    # Y rotation
    Rgamma        = np.eye(4,4)
    sg            = np.sin(gamma)
    cg            = np.cos(gamma)
    Rgamma[0,0]   = cg
    Rgamma[2,2]   = Rgamma[0,0]
    Rgamma[0,2]   = sg
    Rgamma[2,0]   = -sg
    
    # Z rotation (in-image-plane)
    Rtheta      = np.eye(4,4)
    st          = np.sin(theta)
    ct          = np.cos(theta)
    Rtheta[0,0] = ct
    Rtheta[1,1] = Rtheta[0,0]
    Rtheta[0,1] = -st
    Rtheta[1,0] = st
    
    R           = reduce(lambda x,y : np.matmul(x,y), [Rphi, Rgamma, Rtheta]) 
    
    return R

def getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sidelength):
    
    ptsIn2D      =  ptsIn[0,:]
    ptsOut2D     =  ptsOut[0,:]
    ptsOut2Dlist =  []
    ptsIn2Dlist  =  []
    
    for i in range(0,4):
        ptsOut2Dlist.append([ptsOut2D[i,0], ptsOut2D[i,1]])
        ptsIn2Dlist.append([ptsIn2D[i,0], ptsIn2D[i,1]])
    
    pin  =  np.array(ptsIn2Dlist)   +  [W/2.,H/2.]
    pout = (np.array(ptsOut2Dlist)  +  [1.,1.]) * (0.5*sidelength)
    pin  = pin.astype(np.float32)
    pout = pout.astype(np.float32)
    
    return pin, pout


def warpMatrix(W, H, theta, phi, gamma, scale, fV):
    
    # M is to be estimated
    M          = np.eye(4, 4)
    
    fVhalf     = np.deg2rad(fV/2.)
    d          = np.sqrt(W*W+H*H)
    sideLength = scale*d/np.cos(fVhalf)
    h          = d/(2.0*np.sin(fVhalf))
    n          = h-(d/2.0)
    f          = h+(d/2.0)
    
    # Translation along Z-axis by -h
    T       = np.eye(4,4)
    T[2,3]  = -h
    
    # Rotation matrices around x,y,z
    R = getRotationMatrixManual([phi, gamma, theta])
    
    
    # Projection Matrix 
    P       = np.eye(4,4)
    P[0,0]  = 1.0/np.tan(fVhalf)
    P[1,1]  = P[0,0]
    P[2,2]  = -(f+n)/(f-n)
    P[2,3]  = -(2.0*f*n)/(f-n)
    P[3,2]  = -1.0
    
    # pythonic matrix multiplication
    F       = reduce(lambda x,y : np.matmul(x,y), [P, T, R]) 
    
    # shape should be 1,4,3 for ptsIn and ptsOut since perspectiveTransform() expects data in this way. 
    # In C++, this can be achieved by Mat ptsIn(1,4,CV_64FC3);
    ptsIn = np.array([[
                 [-W/2., H/2., 0.],[ W/2., H/2., 0.],[ W/2.,-H/2., 0.],[-W/2.,-H/2., 0.]
                 ]])
    ptsOut  = np.array(np.zeros((ptsIn.shape), dtype=ptsIn.dtype))
    ptsOut  = cv2.perspectiveTransform(ptsIn, F)
    
    ptsInPt2f, ptsOutPt2f = getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sideLength)
    
    # check float32 otherwise OpenCV throws an error
    assert(ptsInPt2f.dtype  == np.float32)
    assert(ptsOutPt2f.dtype == np.float32)
    M33 = cv2.getPerspectiveTransform(ptsInPt2f,ptsOutPt2f)

    return M33, sideLength

def get_flip_perspective_matrix(W, H, keys, frame_idx):
    perspective_flip_theta = keys.perspective_flip_theta_series[frame_idx]
    perspective_flip_phi = keys.perspective_flip_phi_series[frame_idx]
    perspective_flip_gamma = keys.perspective_flip_gamma_series[frame_idx]
    perspective_flip_fv = keys.perspective_flip_fv_series[frame_idx]
    M,sl = warpMatrix(W, H, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, 1., perspective_flip_fv);
    post_trans_mat = np.float32([[1, 0, (W-sl)/2], [0, 1, (H-sl)/2]])
    post_trans_mat = np.vstack([post_trans_mat, [0,0,1]])
    bM = np.matmul(M, post_trans_mat)
    return bM

def flip_3d_perspective(anim_args, prev_img_cv2, keys, frame_idx):
    W, H = (prev_img_cv2.shape[1], prev_img_cv2.shape[0])
    return cv2.warpPerspective(
        prev_img_cv2,
        get_flip_perspective_matrix(W, H, keys, frame_idx),
        (W, H),
        borderMode=cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE
    )

def anim_frame_warp(prev_img_cv2, args, anim_args, keys, frame_idx, depth_model=None, depth=None, device='cuda', half_precision = False):

    if anim_args.use_depth_warping:
        if depth is None and depth_model is not None:
            depth = depth_model.predict(prev_img_cv2, anim_args.midas_weight, half_precision)
            
    else:
        depth = None

    if anim_args.animation_mode == '2D':
        prev_img = anim_frame_warp_2d(prev_img_cv2, args, anim_args, keys, frame_idx)
    else: # '3D'
        prev_img = anim_frame_warp_3d(device, prev_img_cv2, depth, anim_args, keys, frame_idx)
                
    return prev_img, depth

def anim_frame_warp_2d(prev_img_cv2, args, anim_args, keys, frame_idx):
    angle = keys.angle_series[frame_idx]
    zoom = keys.zoom_series[frame_idx]
    translation_x = keys.translation_x_series[frame_idx]
    translation_y = keys.translation_y_series[frame_idx]
    transform_center_x = keys.transform_center_x_series[frame_idx]
    transform_center_y = keys.transform_center_y_series[frame_idx]
    center_point = (args.W * transform_center_x, args.H * transform_center_y)
    rot_mat = cv2.getRotationMatrix2D(center_point, angle, zoom)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    if anim_args.enable_perspective_flip:
        bM = get_flip_perspective_matrix(args.W, args.H, keys, frame_idx)
        rot_mat = np.matmul(bM, rot_mat, trans_mat)
    else:
        rot_mat = np.matmul(rot_mat, trans_mat)
    return cv2.warpPerspective(
        prev_img_cv2,
        rot_mat,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        borderMode=cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE
    )

def anim_frame_warp_3d(device, prev_img_cv2, depth, anim_args, keys, frame_idx):
    TRANSLATION_SCALE = 1.0/200.0 # matches Disco
    translate_xyz = [
        -keys.translation_x_series[frame_idx] * TRANSLATION_SCALE, 
        keys.translation_y_series[frame_idx] * TRANSLATION_SCALE, 
        -keys.translation_z_series[frame_idx] * TRANSLATION_SCALE
    ]
    rotate_xyz = [
        math.radians(keys.rotation_3d_x_series[frame_idx]), 
        math.radians(keys.rotation_3d_y_series[frame_idx]), 
        math.radians(keys.rotation_3d_z_series[frame_idx])
    ]
    if anim_args.enable_perspective_flip:
        prev_img_cv2 = flip_3d_perspective(anim_args, prev_img_cv2, keys, frame_idx)
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
    result = transform_image_3d_switcher(device if not device.type.startswith('mps') else torch.device('cpu'), prev_img_cv2, depth, rot_mat, translate_xyz, anim_args, keys, frame_idx)
    torch.cuda.empty_cache()
    return result

def transform_image_3d_switcher(device, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args, keys, frame_idx, depth_min_0=False, equalize=True, autocontrast=False, autocontrast_cutoff_low=0, autocontrast_cutoff_high=1):
    if anim_args.depth_algorithm.lower() in ['midas+adabins', 'zoe+adabins']:
        return transform_image_3d_legacy(device, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args, keys, frame_idx)
    else:
        return transform_image_3d_new(device, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args, keys, frame_idx, depth_min_0, equalize, autocontrast, autocontrast_cutoff_low, autocontrast_cutoff_high)
        
def transform_image_3d_legacy(device, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args, keys, frame_idx):
    # adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion 
    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    if anim_args.aspect_ratio_use_old_formula:
        aspect_ratio = float(w)/float(h)
    else:
        aspect_ratio = keys.aspect_ratio_series[frame_idx]
    
    near = keys.near_series[frame_idx]
    far = keys.far_series[frame_idx]
    fov_deg = keys.fov_series[frame_idx]
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, R=rot_mat, T=torch.tensor([translate]), device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))
    if depth_tensor is None:
        z = torch.ones_like(x)
    else:
        z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    image_tensor = rearrange(torch.from_numpy(prev_img_cv2.astype(np.float32)), 'h w c -> c h w').to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.add(1/512 - 0.0001).unsqueeze(0), 
        offset_coords_2d, 
        mode=anim_args.sampling_mode, 
        padding_mode=anim_args.padding_mode, 
        align_corners=False
    )

    # convert back to cv2 style numpy array
    result = rearrange(
        new_image.squeeze().clamp(0,255), 
        'c h w -> h w c'
    ).cpu().numpy().astype(prev_img_cv2.dtype)
    return result

def transform_image_3d_new(device, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args, keys, frame_idx, depth_min_0=False, equalize=True, autocontrast=False, autocontrast_cutoff_low=0, autocontrast_cutoff_high=1):
    '''
    originally an adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion
    modified by reallybigname to control various incoming tensors
    '''
    if anim_args.depth_algorithm.lower().startswith('midas'):
        depth = 1
        depth_factor = -1
        depth_offset = -2
    elif anim_args.depth_algorithm.lower() == "adabins":
        depth = 1
        depth_factor = 1
        depth_offset = 1
    elif anim_args.depth_algorithm.lower() == "leres":
        depth = 1
        depth_factor = 1
        depth_offset = 1
    elif anim_args.depth_algorithm.lower() == "zoe":
        depth = 1
        depth_factor = 1
        depth_offset = 1
    else:
        raise Exception(f"Unknown depth_algorithm passed to transform_image_3d function: {anim_args.depth_algorithm}")

    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    # depth stretching aspect ratio (has nothing to do with image dimensions - which is why the old formula was flawed)
    aspect_ratio = float(w)/float(h) if anim_args.aspect_ratio_use_old_formula else keys.aspect_ratio_series[frame_idx]
    
    # get projection keys
    near = keys.near_series[frame_idx]
    far = keys.far_series[frame_idx]
    fov_deg = keys.fov_series[frame_idx]

    # get perspective cams old (still) and new (transformed)
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, R=rot_mat, T=torch.tensor([translate]), device=device)

    # make xy meshgrid - range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))

    # test tensor for validity (some are corrupted for some reason)
    depth_tensor_invalid = depth_tensor is None or torch.isnan(depth_tensor).any() or torch.isinf(depth_tensor).any() or depth_tensor.min() == depth_tensor.max()

    # if invalid, create flat matrix for this frame
    if depth_tensor_invalid:
        print("Depth tensor invalid. Flat depth for this frame.")
        z = torch.ones_like(x)
    else:
        # console output vars
        con_txt = "\033[36mDepth"
        txt_depth_min = '{:.2f}'.format(float(depth_tensor.min()))
        txt_depth_max = '{:.2f}'.format(float(depth_tensor.max()))
        diff = '{:.2f}'.format(float(depth_tensor.max())-float(depth_tensor.min()))

        # prepare tensor between 0 and 1 with optional equalization and autocontrast
        depth_normalized = prepare_depth_tensor(depth_tensor, depth_min_0, equalize, autocontrast, autocontrast_cutoff_low, autocontrast_cutoff_high)

        # Rescale the depth values to depth with offset (depth 2 and offset -1 would be -1 to +11)
        depth_final = depth_normalized * depth + depth_offset

        # depth factor (1 is normal. -1 is inverted)
        if depth_factor != 1:
            depth_final *= depth_factor

        # console reporting of normalization
        con_txt += f" normalized to {depth_final.min()}/{depth_final.max()} from"

        # console reporting of depth normalization, min, max, diff
        print(con_txt + f" {txt_depth_min}/{txt_depth_max} diff {diff}\033[0m")

        # add z from depth
        z = torch.as_tensor(depth_final, dtype=torch.float32, device=device)

    # calculate offset_xy
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy

    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)

    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    # do the hyperdimensional remap
    image_tensor = rearrange(torch.from_numpy(prev_img_cv2.astype(np.float32)), 'h w c -> c h w').to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.unsqueeze(0),  # image_tensor.add(1/512 - 0.0001).unsqueeze(0), 
        offset_coords_2d, 
        mode=anim_args.sampling_mode, 
        padding_mode=anim_args.padding_mode, 
        align_corners=False
    )

    # convert back to cv2 style numpy array
    result = rearrange(
        new_image.squeeze().clamp(0,255), 
        'c h w -> h w c'
    ).cpu().numpy().astype(prev_img_cv2.dtype)
    return result
     
def prepare_depth_tensor(depth_tensor=None, depth_min_0=False, equalize=False, autocontrast=False, autocontrast_cutoff_low=0, autocontrast_cutoff_high=1):
    """
    Prepares a depth tensor with optional equalization and autocontrast.
    Normalizes the depth tensor values between 0 and 1.

    Args:
    depth_tensor (torch.Tensor): A 2D depth tensor (H, W)
    depth_min_0 (bool): If True, normalize using tensor floor of 0; otherwise, use tensor minimum.
    equalize (bool): If True, perform equalization on the depth tensor.
    autocontrast (bool): If True, apply autocontrast to the depth tensor.
    autocontrast_cutoff_low (float): The lower cutoff value for autocontrast (between 0.0 and 1.0).
    autocontrast_cutoff_high (float): The upper cutoff value for autocontrast (between 0.0 and 1.0).

    Returns:
    torch.Tensor: Prepared depth tensor (2D) with values between 0 and 1.
    """

    # normalization, zero based or min based
    if depth_min_0: # tensor floor = 0
        depth_tensor = 1 - (1 / depth_tensor.max())
        # Clip the equalized depth tensor values between 0 and 1
        equalized_depth_tensor = torch.clamp(equalized_depth_tensor, 0, 1)
    else: # tensor floor = tensor minimum
        depth_range = depth_tensor.max() - depth_tensor.min()
        depth_tensor = (depth_tensor - depth_tensor.min()) / depth_range

    # equalization of depth tensor (must be 0 to 1 at this stage)
    if equalize:
        depth_tensor = depth_equalization(depth_tensor=depth_tensor)
    
    # auto-contrast of depth tensor (must be 0 to 1 at this stage)
    if autocontrast:
        depth_tensor = depth_auto_contrast(depth_tensor=depth_tensor, low_cutoff=autocontrast_cutoff_low, high_cutoff=autocontrast_cutoff_high)
    
    return depth_tensor

def depth_equalization(depth_tensor):
    """
    Perform histogram equalization on a single-channel depth tensor.

    Args:
    depth_tensor (torch.Tensor): A 2D depth tensor (H, W).

    Returns:
    torch.Tensor: Equalized depth tensor (2D).
    """

    # Convert the depth tensor to a NumPy array for processing
    depth_array = depth_tensor.cpu().numpy()

    # Calculate the histogram of the depth values using a specified number of bins
    # Increase the number of bins for higher precision depth tensors
    hist, bin_edges = np.histogram(depth_array, bins=1024, range=(0, 1))

    # Calculate the cumulative distribution function (CDF) of the histogram
    cdf = hist.cumsum()

    # Normalize the CDF so that the maximum value is 1
    cdf = cdf / float(cdf[-1])

    # Perform histogram equalization by mapping the original depth values to the CDF values
    equalized_depth_array = np.interp(depth_array, bin_edges[:-1], cdf)

    # Convert the equalized depth array back to a PyTorch tensor and return it
    equalized_depth_tensor = torch.from_numpy(equalized_depth_array).to(depth_tensor.device)

    return equalized_depth_tensor

def depth_auto_contrast(depth_tensor=None, low_cutoff=0.0, high_cutoff=1.0):
    """
    Apply auto-contrast to a depth tensor by adjusting the range of depth values.

    Args:
    depth_tensor (torch.Tensor): A 2D depth tensor (H, W)
    low_cutoff (float): The lower cutoff value as a percentage (between 0.0 and 1.0) for clipping depth values.
    high_cutoff (float): The upper cutoff value as a percentage (between 0.0 and 1.0) for clipping depth values.

    Returns:
    torch.Tensor: Auto-contrasted depth tensor (2D)
    """

    # Calculate the lower and upper bounds based on the cutoff values
    depth_min = torch.quantile(depth_tensor, low_cutoff)
    depth_max = torch.quantile(depth_tensor, high_cutoff)

    # Apply auto-contrast by clipping and scaling the depth values
    depth_clipped = torch.clamp(depth_tensor, depth_min, depth_max)
    depth_auto_contrasted = (depth_clipped - depth_min) / (depth_max - depth_min)

    return depth_auto_contrasted