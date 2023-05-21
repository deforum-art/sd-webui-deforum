import os
import pathlib
import random
import cv2
import numpy as np
import PIL
from PIL import Image, ImageChops, ImageOps, ImageEnhance
from scipy.ndimage.filters import gaussian_filter
from .consistency_check import make_consistency
from .human_masking import video2humanmasks
from .load_images import load_image
from .video_audio_utilities import vid2frames, get_quick_vid_info, get_frame_name

def delete_all_imgs_in_folder(folder_path):
        files = list(pathlib.Path(folder_path).glob('*.jpg'))
        files.extend(list(pathlib.Path(folder_path).glob('*.png')))
        for f in files: os.remove(f)
    
def hybrid_generation(args, anim_args, root):
    video_in_frame_path = os.path.join(args.outdir, 'inputframes')
    hybrid_frame_path = os.path.join(args.outdir, 'hybridframes')
    human_masks_path = os.path.join(args.outdir, 'human_masks')

    # create hybridframes folder whether using init_image or inputframes
    os.makedirs(hybrid_frame_path, exist_ok=True)

    if anim_args.hybrid_generate_inputframes:
        # create folders for the video input frames and optional hybrid frames to live in
        os.makedirs(video_in_frame_path, exist_ok=True)
                
        # delete frames if overwrite = true
        if anim_args.overwrite_extracted_frames:
            delete_all_imgs_in_folder(hybrid_frame_path)

        # save the video frames from input video
        print(f"Video to extract: {anim_args.video_init_path}")
        print(f"Extracting video (1 every {anim_args.extract_nth_frame}) frames to {video_in_frame_path}...")
        video_fps = vid2frames(video_path=anim_args.video_init_path, video_in_frame_path=video_in_frame_path, n=anim_args.extract_nth_frame, overwrite=anim_args.overwrite_extracted_frames, extract_from_frame=anim_args.extract_from_frame, extract_to_frame=anim_args.extract_to_frame)
    
    # extract alpha masks of humans from the extracted input video imgs
    if anim_args.hybrid_generate_human_masks != "None":
        # create a folder for the human masks imgs to live in
        print(f"Checking /creating a folder for the human masks")
        os.makedirs(human_masks_path, exist_ok=True)
            
        # delete frames if overwrite = true
        if anim_args.overwrite_extracted_frames:
            delete_all_imgs_in_folder(human_masks_path)
        
        # in case that generate_input_frames isn't selected, we won't get the video fps rate as vid2frames isn't called, So we'll check the video fps in here instead
        if not anim_args.hybrid_generate_inputframes:
            _, video_fps, _ = get_quick_vid_info(anim_args.video_init_path)
            
        # calculate the correct fps of the masked video according to the original video fps and 'extract_nth_frame'
        output_fps = video_fps/anim_args.extract_nth_frame
        
        # generate the actual alpha masks from the input imgs
        print(f"Extracting alpha humans masks from the input frames")
        video2humanmasks(video_in_frame_path, human_masks_path, anim_args.hybrid_generate_human_masks, output_fps)
        
    # get sorted list of inputfiles
    inputfiles = sorted(pathlib.Path(video_in_frame_path).glob('*.jpg'))

    if not anim_args.hybrid_use_init_image:
        # determine max frames from length of input frames
        anim_args.max_frames = len(inputfiles)
        print(f"Using {anim_args.max_frames} input frames from {video_in_frame_path}...")

    # use first frame as init
    if anim_args.hybrid_use_first_frame_as_init_image:
        for f in inputfiles:
            args.init_image = str(f)
            args.use_init = True
            print(f"Using init_image from video: {args.init_image}")
            break

    return args, anim_args, inputfiles

def hybrid_composite(args, anim_args, frame_idx, prev_img, depth_model, hybrid_comp_schedules, root):
    video_frame = os.path.join(args.outdir, 'inputframes', get_frame_name(anim_args.video_init_path) + f"{frame_idx:09}.jpg")
    video_depth_frame = os.path.join(args.outdir, 'hybridframes', get_frame_name(anim_args.video_init_path) + f"_vid_depth{frame_idx:09}.jpg")
    depth_frame = os.path.join(args.outdir, f"{root.timestring}_depth_{frame_idx-1:09}.png")
    mask_frame = os.path.join(args.outdir, 'hybridframes', get_frame_name(anim_args.video_init_path) + f"_mask{frame_idx:09}.jpg")
    comp_frame = os.path.join(args.outdir, 'hybridframes', get_frame_name(anim_args.video_init_path) + f"_comp{frame_idx:09}.jpg")
    prev_frame = os.path.join(args.outdir, 'hybridframes', get_frame_name(anim_args.video_init_path) + f"_prev{frame_idx:09}.jpg")
    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
    prev_img_hybrid = Image.fromarray(prev_img)
    if anim_args.hybrid_use_init_image:
        video_image = load_image(args.init_image)
    else:
        video_image = Image.open(video_frame)
    video_image = video_image.resize((args.W, args.H), PIL.Image.LANCZOS)
    hybrid_mask = None

    # composite mask types
    if anim_args.hybrid_comp_mask_type == 'Depth': # get depth from last generation
        hybrid_mask = Image.open(depth_frame)
    elif anim_args.hybrid_comp_mask_type == 'Video Depth': # get video depth
        video_depth = depth_model.predict(np.array(video_image), anim_args.midas_weight, root.half_precision)
        depth_model.save(video_depth_frame, video_depth)
        hybrid_mask = Image.open(video_depth_frame)
    elif anim_args.hybrid_comp_mask_type == 'Blend': # create blend mask image
        hybrid_mask = Image.blend(ImageOps.grayscale(prev_img_hybrid), ImageOps.grayscale(video_image), hybrid_comp_schedules['mask_blend_alpha'])
    elif anim_args.hybrid_comp_mask_type == 'Difference': # create difference mask image
        hybrid_mask = ImageChops.difference(ImageOps.grayscale(prev_img_hybrid), ImageOps.grayscale(video_image))
        
    # optionally invert mask, if mask type is defined
    if anim_args.hybrid_comp_mask_inverse and anim_args.hybrid_comp_mask_type != "None":
        hybrid_mask = ImageOps.invert(hybrid_mask)

    # if a mask type is selected, make composition
    if hybrid_mask is None:
        hybrid_comp = video_image
    else:
        # ensure grayscale
        hybrid_mask = ImageOps.grayscale(hybrid_mask)
        # equalization before
        if anim_args.hybrid_comp_mask_equalize in ['Before', 'Both']:
            hybrid_mask = ImageOps.equalize(hybrid_mask)        
        # contrast
        hybrid_mask = ImageEnhance.Contrast(hybrid_mask).enhance(hybrid_comp_schedules['mask_contrast'])
        # auto contrast with cutoffs lo/hi
        if anim_args.hybrid_comp_mask_auto_contrast:
            hybrid_mask = autocontrast_grayscale(np.array(hybrid_mask), hybrid_comp_schedules['mask_auto_contrast_cutoff_low'], hybrid_comp_schedules['mask_auto_contrast_cutoff_high'])
            hybrid_mask = Image.fromarray(hybrid_mask)
            hybrid_mask = ImageOps.grayscale(hybrid_mask)   
        if anim_args.hybrid_comp_save_extra_frames:
            hybrid_mask.save(mask_frame)        
        # equalization after
        if anim_args.hybrid_comp_mask_equalize in ['After', 'Both']:
            hybrid_mask = ImageOps.equalize(hybrid_mask)        
        # do compositing and save
        hybrid_comp = Image.composite(prev_img_hybrid, video_image, hybrid_mask)            
        if anim_args.hybrid_comp_save_extra_frames:
            hybrid_comp.save(comp_frame)

    # final blend of composite with prev_img, or just a blend if no composite is selected
    hybrid_blend = Image.blend(prev_img_hybrid, hybrid_comp, hybrid_comp_schedules['alpha'])  
    if anim_args.hybrid_comp_save_extra_frames:
        hybrid_blend.save(prev_frame)

    prev_img = cv2.cvtColor(np.array(hybrid_blend), cv2.COLOR_RGB2BGR)

    # restore to np array and return
    return args, prev_img

def get_matrix_for_hybrid_motion(frame_idx, dimensions, inputfiles, hybrid_motion):
    print(f"Calculating {hybrid_motion} RANSAC matrix for frames {frame_idx} to {frame_idx+1}")
    img1 = cv2.cvtColor(get_resized_image_from_filename(str(inputfiles[frame_idx]), dimensions), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(get_resized_image_from_filename(str(inputfiles[frame_idx+1]), dimensions), cv2.COLOR_BGR2GRAY)
    M = get_transformation_matrix_from_images(img1, img2, hybrid_motion)
    return M

def get_matrix_for_hybrid_motion_prev(frame_idx, dimensions, inputfiles, prev_img, hybrid_motion):
    print(f"Calculating {hybrid_motion} RANSAC matrix for frames {frame_idx} to {frame_idx+1}")
    # first handle invalid images by returning default matrix
    height, width = prev_img.shape[:2]
    if height == 0 or width == 0 or prev_img != np.uint8:
        return get_hybrid_motion_default_matrix(hybrid_motion)
    else:
        prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(get_resized_image_from_filename(str(inputfiles[frame_idx+1]), dimensions), cv2.COLOR_BGR2GRAY)
        M = get_transformation_matrix_from_images(prev_img_gray, img, hybrid_motion)
        return M

def get_flow_for_hybrid_motion(frame_idx, dimensions, inputfiles, hybrid_frame_path, prev_flow, method, raft_model, consistency_check=True, consistency_blur=0, do_flow_visualization=False):
    print(f"Calculating {method} optical flow {'w/consistency mask' if consistency_check else ''} for frames {frame_idx} to {frame_idx+1}")
    i1 = get_resized_image_from_filename(str(inputfiles[frame_idx]), dimensions)
    i2 = get_resized_image_from_filename(str(inputfiles[frame_idx+1]), dimensions)
    if consistency_check:
        flow, reliable_flow = get_reliable_flow_from_images(i1, i2, method, raft_model, prev_flow, consistency_blur) # forward flow w/backward consistency check
        if do_flow_visualization: save_flow_mask_visualization(frame_idx, reliable_flow, hybrid_frame_path)
    else:
        flow = get_flow_from_images(i1, i2, method, raft_model, prev_flow) # old single flow forward
    if do_flow_visualization: save_flow_visualization(frame_idx, dimensions, flow, inputfiles, hybrid_frame_path)
    return flow

def get_flow_for_hybrid_motion_prev(frame_idx, dimensions, inputfiles, hybrid_frame_path, prev_flow, prev_img, method, raft_model, consistency_check=True, consistency_blur=0, do_flow_visualization=False):
    print(f"Calculating {method} optical flow {'w/consistency mask' if consistency_check else ''} for frames {frame_idx} to {frame_idx+1}")
    reliable_flow = None
    # first handle invalid images by returning default flow
    height, width = prev_img.shape[:2]   
    if height == 0 or width == 0:
        flow = get_hybrid_motion_default_flow(dimensions)
    else:
        i1 = prev_img.astype(np.uint8)
        i2 = get_resized_image_from_filename(str(inputfiles[frame_idx+1]), dimensions)
        if consistency_check:
            flow, reliable_flow = get_reliable_flow_from_images(i1, i2, method, raft_model, prev_flow, consistency_blur) # forward flow w/backward consistency check
            if do_flow_visualization: save_flow_mask_visualization(frame_idx, reliable_flow, hybrid_frame_path)
        else:
            flow = get_flow_from_images(i1, i2, method, raft_model, prev_flow)
    if do_flow_visualization: save_flow_visualization(frame_idx, dimensions, flow, inputfiles, hybrid_frame_path)
    return flow

def get_reliable_flow_from_images(i1, i2, method, raft_model, prev_flow, consistency_blur, reliability=0):
    flow_forward = get_flow_from_images(i1, i2, method, raft_model, prev_flow)
    flow_backward = get_flow_from_images(i2, i1, method, raft_model, None)
    reliable_flow = make_consistency(flow_forward, flow_backward, edges_unreliable=False)
    if consistency_blur > 0:
        reliable_flow = custom_gaussian_blur(reliable_flow.astype(np.float32), 1, consistency_blur)
    return filter_flow(flow_forward, reliable_flow, consistency_blur, reliability), reliable_flow

def custom_gaussian_blur(input_array, blur_size, sigma):
    return gaussian_filter(input_array, sigma=(sigma, sigma, 0), order=0, mode='constant', cval=0.0, truncate=blur_size)

def filter_flow(flow, reliable_flow, reliability=0.5, consistency_blur=0):
    # reliability from reliabile flow: -0.75 is bad, 0 is meh/outside, 1 is great
    # Create a mask from the first channel of the reliable_flow array
    mask = reliable_flow[..., 0]

    # to set everything to 1 or 0 based on reliability
    # mask = np.where(mask >= reliability, 1, 0)

    # Expand the mask to match the shape of the forward_flow array
    mask = np.repeat(mask[..., np.newaxis], flow.shape[2], axis=2)

    # Apply the mask to the flow
    return flow * mask

def image_transform_ransac(image_cv2, M, hybrid_motion, depth=None):
    if hybrid_motion == "Perspective":
        return image_transform_perspective(image_cv2, M, depth)
    else: # Affine
        return image_transform_affine(image_cv2, M, depth)

def image_transform_optical_flow(img, flow, flow_factor):
    # if flow factor not normal, calculate flow factor
    if flow_factor != 1:
        flow = flow * flow_factor
    # flow is reversed, so you need to reverse it:
    flow = -flow
    h, w = img.shape[:2]
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:,np.newaxis]
    return remap(img, flow)

def image_transform_affine(image_cv2, M, depth=None):
    if depth is None:
        return cv2.warpAffine(
            image_cv2,
            M,
            (image_cv2.shape[1],image_cv2.shape[0]),
            borderMode=cv2.BORDER_REFLECT_101
        )
    else:  # NEED TO IMPLEMENT THE FOLLOWING FUNCTION
        return depth_based_affine_warp(
            image_cv2,
            depth,
            M            
        )

def image_transform_perspective(image_cv2, M, depth=None):
    if depth is None:
        return cv2.warpPerspective(
            image_cv2,
            M,
            (image_cv2.shape[1], image_cv2.shape[0]),
            borderMode=cv2.BORDER_REFLECT_101
        )
    else:  # NEED TO IMPLEMENT THE FOLLOWING FUNCTION
        return render_3d_perspective(
            image_cv2,
            depth,
            M            
        )

def get_hybrid_motion_default_matrix(hybrid_motion):
    if hybrid_motion == "Perspective":
        arr = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    else:
        arr = np.array([[1., 0., 0.], [0., 1., 0.]])
    return arr

def get_hybrid_motion_default_flow(dimensions):
    cols, rows = dimensions
    flow = np.zeros((rows, cols, 2), np.float32)
    return flow

def get_transformation_matrix_from_images(img1, img2, hybrid_motion, confidence=0.75):
    # Create SIFT detector and feature extractor
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Create BFMatcher object and match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < confidence * n.distance:
            good_matches.append(m)

    if len(good_matches) <= 8:
        get_hybrid_motion_default_matrix(hybrid_motion)

    # Convert keypoints to numpy arrays
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    if len(src_pts) <= 8 or len(dst_pts) <= 8:
        return get_hybrid_motion_default_matrix(hybrid_motion)
    elif hybrid_motion == "Perspective": # Perspective transformation (3x3)
        transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return transformation_matrix
    else: # Affine - rigid transformation (no skew 3x2)
        transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        return transformation_rigid_matrix

def get_flow_from_images(i1, i2, method, raft_model, prev_flow=None):
    if method == "RAFT":
        if raft_model is None:
            raise Exception("RAFT Model not provided to get_flow_from_images function, cannot continue.")
        return get_flow_from_images_RAFT(i1, i2, raft_model)
    elif method == "DIS Medium":
        return get_flow_from_images_DIS(i1, i2, 'medium', prev_flow)
    elif method == "DIS Fine":
        return get_flow_from_images_DIS(i1, i2, 'fine', prev_flow)
    elif method == "DenseRLOF": # Unused - requires running opencv-contrib-python (full opencv) INSTEAD of opencv-python
        return get_flow_from_images_Dense_RLOF(i1, i2, prev_flow)
    elif method == "SF": # Unused - requires running opencv-contrib-python (full opencv) INSTEAD of opencv-python
        return get_flow_from_images_SF(i1, i2, prev_flow)
    elif method == "DualTVL1": # Unused - requires running opencv-contrib-python (full opencv) INSTEAD of opencv-python
        return get_flow_from_images_DualTVL1(i1, i2, prev_flow)
    elif method == "DeepFlow": # Unused - requires running opencv-contrib-python (full opencv) INSTEAD of opencv-python
        return get_flow_from_images_DeepFlow(i1, i2, prev_flow)
    elif method == "PCAFlow": # Unused - requires running opencv-contrib-python (full opencv) INSTEAD of opencv-python
        return get_flow_from_images_PCAFlow(i1, i2, prev_flow)
    elif method == "Farneback": # Farneback Normal:
        return get_flow_from_images_Farneback(i1, i2, prev_flow)
    # if we reached this point, something went wrong. raise an error:
    raise RuntimeError(f"Invald flow method name: '{method}'")

def get_flow_from_images_RAFT(i1, i2, raft_model):
    flow = raft_model.predict(i1, i2)
    return flow

def get_flow_from_images_DIS(i1, i2, preset, prev_flow):
    # DIS PRESETS CHART KEY: finest scale, grad desc its, patch size
    # DIS_MEDIUM: 1, 25, 8 | DIS_FAST: 2, 16, 8 | DIS_ULTRAFAST: 2, 12, 8
    if preset == 'medium': preset_code = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM    
    elif preset == 'fast': preset_code = cv2.DISOPTICAL_FLOW_PRESET_FAST    
    elif preset == 'ultrafast': preset_code = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST   
    elif preset in ['slow','fine']: preset_code = None
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(preset_code)
    # custom presets
    if preset == 'slow':
        dis.setGradientDescentIterations(192)
        dis.setFinestScale(1)
        dis.setPatchSize(8)
        dis.setPatchStride(4)
    if preset == 'fine':
        dis.setGradientDescentIterations(192)
        dis.setFinestScale(0)
        dis.setPatchSize(8)
        dis.setPatchStride(4)
    return dis.calc(i1, i2, prev_flow)

def get_flow_from_images_Dense_RLOF(i1, i2, last_flow=None):
    return cv2.optflow.calcOpticalFlowDenseRLOF(i1, i2, flow = last_flow)

def get_flow_from_images_SF(i1, i2, last_flow=None, layers = 3, averaging_block_size = 2, max_flow = 4):
    return cv2.optflow.calcOpticalFlowSF(i1, i2, layers, averaging_block_size, max_flow)

def get_flow_from_images_DualTVL1(i1, i2, prev_flow):
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    f = cv2.optflow.DualTVL1OpticalFlow_create()
    return f.calc(i1, i2, prev_flow)

def get_flow_from_images_DeepFlow(i1, i2, prev_flow):
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    f = cv2.optflow.createOptFlow_DeepFlow()
    return f.calc(i1, i2, prev_flow)

def get_flow_from_images_PCAFlow(i1, i2, prev_flow):
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    f = cv2.optflow.createOptFlow_PCAFlow()
    return f.calc(i1, i2, prev_flow)

def get_flow_from_images_Farneback(i1, i2, preset="normal", last_flow=None, pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0):
    flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN         # Specify the operation flags
    pyr_scale = 0.5   # The image scale (<1) to build pyramids for each image
    if preset == "fine":
        levels = 13       # The number of pyramid layers, including the initial image
        winsize = 77      # The averaging window size
        iterations = 13   # The number of iterations at each pyramid level
        poly_n = 15       # The size of the pixel neighborhood used to find polynomial expansion in each pixel
        poly_sigma = 0.8  # The standard deviation of the Gaussian used to smooth derivatives used as a basis for the polynomial expansion
    else: # "normal"
        levels = 5        # The number of pyramid layers, including the initial image
        winsize = 21      # The averaging window size
        iterations = 5    # The number of iterations at each pyramid level
        poly_n = 7        # The size of the pixel neighborhood used to find polynomial expansion in each pixel
        poly_sigma = 1.2  # The standard deviation of the Gaussian used to smooth derivatives used as a basis for the polynomial expansion
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    flags = 0 # flags = cv2.OPTFLOW_USE_INITIAL_FLOW    
    flow = cv2.calcOpticalFlowFarneback(i1, i2, last_flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    return flow

def save_flow_visualization(frame_idx, dimensions, flow, inputfiles, hybrid_frame_path):
    flow_img_file = os.path.join(hybrid_frame_path, f"flow{frame_idx:09}.jpg")
    flow_img = cv2.imread(str(inputfiles[frame_idx]))
    flow_img = cv2.resize(flow_img, (dimensions[0], dimensions[1]), cv2.INTER_AREA)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2GRAY)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_GRAY2BGR)
    flow_img = draw_flow_lines_in_grid_in_color(flow_img, flow)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(flow_img_file, flow_img)
    print(f"Saved optical flow visualization: {flow_img_file}")

def save_flow_mask_visualization(frame_idx, reliable_flow, hybrid_frame_path, color=True):
    flow_mask_img_file = os.path.join(hybrid_frame_path, f"flow_mask{frame_idx:09}.jpg")
    if color:
        # Normalize the reliable_flow array to the range [0, 255]
        normalized_reliable_flow = (reliable_flow - reliable_flow.min()) / (reliable_flow.max() - reliable_flow.min()) * 255
        # Change the data type to np.uint8
        mask_image = normalized_reliable_flow.astype(np.uint8)
    else:
        # Extract the first channel of the reliable_flow array
        first_channel = reliable_flow[..., 0]
        # Normalize the first channel to the range [0, 255]
        normalized_first_channel = (first_channel - first_channel.min()) / (first_channel.max() - first_channel.min()) * 255
        # Change the data type to np.uint8
        grayscale_image = normalized_first_channel.astype(np.uint8)
        # Replicate the grayscale channel three times to form a BGR image
        mask_image = np.stack((grayscale_image, grayscale_image, grayscale_image), axis=2)
    cv2.imwrite(flow_mask_img_file, mask_image)
    print(f"Saved mask flow visualization: {flow_mask_img_file}")

def reliable_flow_to_image(reliable_flow):
    # Extract the first channel of the reliable_flow array
    first_channel = reliable_flow[..., 0]
    # Normalize the first channel to the range [0, 255]
    normalized_first_channel = (first_channel - first_channel.min()) / (first_channel.max() - first_channel.min()) * 255
    # Change the data type to np.uint8
    grayscale_image = normalized_first_channel.astype(np.uint8)
    # Replicate the grayscale channel three times to form a BGR image
    bgr_image = np.stack((grayscale_image, grayscale_image, grayscale_image), axis=2)
    return bgr_image

def draw_flow_lines_in_grid_in_color(img, flow, step=8, magnitude_multiplier=1, min_magnitude = 0, max_magnitude = 10000):
    flow = flow * magnitude_multiplier
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    vis = cv2.add(vis, bgr)

    # Iterate through the lines
    for (x1, y1), (x2, y2) in lines:
        # Calculate the magnitude of the line
        magnitude = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Only draw the line if it falls within the magnitude range
        if min_magnitude <= magnitude <= max_magnitude:
            b = int(bgr[y1, x1, 0])
            g = int(bgr[y1, x1, 1])
            r = int(bgr[y1, x1, 2])
            color = (b, g, r)
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, thickness=1, tipLength=0.1)    
    return vis

def draw_flow_lines_in_color(img, flow, threshold=3, magnitude_multiplier=1, min_magnitude = 0, max_magnitude = 10000):
    # h, w = img.shape[:2]
    vis = img.copy()  # Create a copy of the input image
    
    # Find the locations in the flow field where the magnitude of the flow is greater than the threshold
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    idx = np.where(mag > threshold)

    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV image to BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Add color from bgr 
    vis = cv2.add(vis, bgr)

    # Draw an arrow at each of these locations to indicate the direction of the flow
    for i, (y, x) in enumerate(zip(idx[0], idx[1])):
        # Calculate the magnitude of the line
        x2 = x + magnitude_multiplier * int(flow[y, x, 0])
        y2 = y + magnitude_multiplier * int(flow[y, x, 1])
        magnitude = np.sqrt((x2 - x)**2 + (y2 - y)**2)

        # Only draw the line if it falls within the magnitude range
        if min_magnitude <= magnitude <= max_magnitude:
            if i % random.randint(100, 200) == 0:
                b = int(bgr[y, x, 0])
                g = int(bgr[y, x, 1])
                r = int(bgr[y, x, 2])
                color = (b, g, r)
                cv2.arrowedLine(vis, (x, y), (x2, y2), color, thickness=1, tipLength=0.25)

    return vis

def autocontrast_grayscale(image, low_cutoff=0, high_cutoff=100):
    # Perform autocontrast on a grayscale np array image.
    # Find the minimum and maximum values in the image
    min_val = np.percentile(image, low_cutoff)
    max_val = np.percentile(image, high_cutoff)

    # Scale the image so that the minimum value is 0 and the maximum value is 255
    image = 255 * (image - min_val) / (max_val - min_val)

    # Clip values that fall outside the range [0, 255]
    image = np.clip(image, 0, 255)

    return image

def get_resized_image_from_filename(im, dimensions):
    img = cv2.imread(im)
    return cv2.resize(img, (dimensions[0], dimensions[1]), cv2.INTER_AREA)

def remap(img, flow):
    border_mode = cv2.BORDER_REFLECT_101
    h, w = img.shape[:2]
    displacement = int(h * 0.25), int(w * 0.25)
    larger_img = cv2.copyMakeBorder(img, displacement[0], displacement[0], displacement[1], displacement[1], border_mode)
    lh, lw = larger_img.shape[:2]
    larger_flow = extend_flow(flow, lw, lh)
    remapped_img = cv2.remap(larger_img, larger_flow, None, cv2.INTER_LINEAR, border_mode)
    output_img = center_crop_image(remapped_img, w, h)
    return output_img

def center_crop_image(img, w, h):
    y, x, _ = img.shape
    width_indent = int((x - w) / 2)
    height_indent = int((y - h) / 2)
    cropped_img = img[height_indent:y-height_indent, width_indent:x-width_indent]
    return cropped_img

def extend_flow(flow, w, h):
    # Get the shape of the original flow image
    flow_h, flow_w = flow.shape[:2]
    # Calculate the position of the image in the new image
    x_offset = int((w - flow_w) / 2)
    y_offset = int((h - flow_h) / 2)
    # Generate the X and Y grids
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    # Create the new flow image and set it to the X and Y grids
    new_flow = np.dstack((x_grid, y_grid)).astype(np.float32)
    # Shift the values of the original flow by the size of the border
    flow[:,:,0] += x_offset
    flow[:,:,1] += y_offset
    # Overwrite the middle of the grid with the original flow
    new_flow[y_offset:y_offset+flow_h, x_offset:x_offset+flow_w, :] = flow
    # Return the extended image
    return new_flow

def abs_flow_to_rel_flow(flow, width, height):
    fx, fy = flow[:,:,0], flow[:,:,1]
    max_flow_x = np.max(np.abs(fx))
    max_flow_y = np.max(np.abs(fy))
    max_flow = max(max_flow_x, max_flow_y)

    rel_fx = fx / (max_flow * width)
    rel_fy = fy / (max_flow * height)
    return np.dstack((rel_fx, rel_fy))

def rel_flow_to_abs_flow(rel_flow, width, height):
    rel_fx, rel_fy = rel_flow[:,:,0], rel_flow[:,:,1]
    
    max_flow_x = np.max(np.abs(rel_fx * width))
    max_flow_y = np.max(np.abs(rel_fy * height))
    max_flow = max(max_flow_x, max_flow_y)

    fx = rel_fx * (max_flow * width)
    fy = rel_fy * (max_flow * height)
    return np.dstack((fx, fy))
