import cv2
import os
import pathlib
import numpy as np
from PIL import Image, ImageChops, ImageOps, ImageEnhance
from .vid2frames import vid2frames, get_frame_name, get_next_frame
from .human_masking import video2humanmasks

def hybrid_generation(args, anim_args, root):
    video_in_frame_path = os.path.join(args.outdir, 'inputframes')
    hybrid_frame_path = os.path.join(args.outdir, 'hybridframes')
    human_masks_path = os.path.join(args.outdir, 'human_masks')

    if anim_args.hybrid_generate_inputframes:
        # create folders for the video input frames and optional hybrid frames to live in
        os.makedirs(video_in_frame_path, exist_ok=True)
        os.makedirs(hybrid_frame_path, exist_ok=True)

        if anim_args.overwrite_extracted_frames:
            # delete hybridframes (they will now be generated again anyway)
            files = pathlib.Path(hybrid_frame_path).glob('*.jpg')
            for f in files: os.remove(f)
            files = pathlib.Path(hybrid_frame_path).glob('*.png')
            for f in files: os.remove(f)

        # save the video frames from input video
        print(f"Video to extract: {anim_args.video_init_path}")
        print(f"Extracting video (1 every {anim_args.extract_nth_frame}) frames to {video_in_frame_path}...")
        vid2frames(anim_args.video_init_path, video_in_frame_path, anim_args.extract_nth_frame, anim_args.overwrite_extracted_frames)
    
    # extract alpha masks of humans from the extracted input video imgs
    if anim_args.hybrid_generate_human_masks:
        # create a folder for the human masks imgs to live in
        print(f"Checking /creating a folder for the human masks")
        os.makedirs(human_masks_path, exist_ok=True)
        
        if anim_args.overwrite_extracted_frames:
            # delete human alpha masks (they will now be generated again anyway)
            files = pathlib.Path(human_masks_path).glob('*.jpg')
            for f in files: os.remove(f)
            files = pathlib.Path(human_masks_path).glob('*.png')
            for f in files: os.remove(f)
        
        # generate the actual alpha masks from the input imgs
        print(f"Extracting alpha humans masks from the input frames")
        video2humanmasks(video_in_frame_path, human_masks_path)
        
    # determine max frames from length of input frames
    anim_args.max_frames = len([f for f in pathlib.Path(video_in_frame_path).glob('*.jpg')])
    print(f"Using {anim_args.max_frames} input frames from {video_in_frame_path}...")

    # get sorted list of inputfiles
    inputfiles = sorted(pathlib.Path(video_in_frame_path).glob('*.jpg'))

    # use first frame as init
    if anim_args.hybrid_use_first_frame_as_init_image:
        for f in inputfiles:
            args.init_image = str(f)
            args.use_init = True
            print(f"Using init_image from video: {args.init_image}")
            break

    return args, anim_args, inputfiles

def hybrid_composite(args, anim_args, frame_idx, prev_img, depth_model, hybrid_comp_schedules, root):
    video_frame = os.path.join(args.outdir, 'inputframes', get_frame_name(anim_args.video_init_path) + f"{frame_idx:05}.jpg")
    video_depth_frame = os.path.join(args.outdir, 'hybridframes', get_frame_name(anim_args.video_init_path) + f"_vid_depth{frame_idx:05}.jpg")
    depth_frame = os.path.join(args.outdir, f"{args.timestring}_depth_{frame_idx-1:05}.png")
    mask_frame = os.path.join(args.outdir, 'hybridframes', get_frame_name(anim_args.video_init_path) + f"_mask{frame_idx:05}.jpg")
    comp_frame = os.path.join(args.outdir, 'hybridframes', get_frame_name(anim_args.video_init_path) + f"_comp{frame_idx:05}.jpg")
    prev_frame = os.path.join(args.outdir, 'hybridframes', get_frame_name(anim_args.video_init_path) + f"_prev{frame_idx:05}.jpg")
    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
    prev_img_hybrid = Image.fromarray(prev_img)
    video_image = Image.open(video_frame)
    video_image = video_image.resize((args.W, args.H), Image.Resampling.LANCZOS)
    hybrid_mask = None

    # composite mask types
    if anim_args.hybrid_comp_mask_type == 'Depth': # get depth from last generation
        hybrid_mask = Image.open(depth_frame)
    elif anim_args.hybrid_comp_mask_type == 'Video Depth': # get video depth
        video_depth = depth_model.predict(np.array(video_image), anim_args, root.half_precision)
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
    if hybrid_mask == None:
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
    matrix = get_translation_matrix_from_images(str(inputfiles[frame_idx]), str(inputfiles[frame_idx+1]), dimensions, hybrid_motion)
    print(f"Calculating {hybrid_motion} RANSAC matrix for frames {frame_idx} to {frame_idx+1}")
    return matrix

def get_flow_for_hybrid_motion(frame_idx, dimensions, inputfiles, hybrid_frame_path, method, save_flow_visualization=False):
    print(f"Calculating optical flow for frames {frame_idx} to {frame_idx+1}")
    flow = get_flow_from_images(str(inputfiles[frame_idx]), str(inputfiles[frame_idx+1]), dimensions, method)
    if save_flow_visualization:
        flow_img_file = os.path.join(hybrid_frame_path, f"flow{frame_idx:05}.jpg")
        flow_cv2 = cv2.imread(str(inputfiles[frame_idx]))
        flow_cv2 = cv2.resize(flow_cv2, (dimensions[0], dimensions[1]), cv2.INTER_AREA)
        flow_cv2 = cv2.cvtColor(flow_cv2,cv2.COLOR_BGR2RGB)
        flow_cv2 = draw_flow_lines_in_grid_in_color(flow_cv2, flow)
        flow_PIL = Image.fromarray(np.uint8(flow_cv2))
        flow_PIL.save(flow_img_file)
        print(f"Saved optical flow visualization: {flow_img_file}")
    return flow

def image_transform_ransac(image_cv2, xform, hybrid_motion, border_mode=cv2.BORDER_REPLICATE):
    if hybrid_motion == "Perspective":
        return image_transform_perspective(image_cv2, xform, border_mode=border_mode)
    else: # Affine
        return image_transform_affine(image_cv2, xform, border_mode=border_mode)

def image_transform_optical_flow(img, flow, border_mode=cv2.BORDER_REPLICATE, flow_reverse=False):
    h, w = flow.shape[:2]
    if not flow_reverse:
        flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:,np.newaxis]
    r = cv2.remap(
        img,
        flow,
        None,
        cv2.INTER_LINEAR,
        border_mode
    )
    return r

def image_transform_affine(image_cv2, xform, border_mode=cv2.BORDER_REPLICATE):
    return cv2.warpAffine(
        image_cv2,
        xform,
        (image_cv2.shape[1],image_cv2.shape[0]),
        borderMode=border_mode
    )

def image_transform_perspective(image_cv2, xform, border_mode=cv2.BORDER_REPLICATE):
    return cv2.warpPerspective(
        image_cv2,
        xform,
        (image_cv2.shape[1], image_cv2.shape[0]),
        borderMode=border_mode
    )

def get_hybrid_motion_default_matrix(hybrid_motion):
    if hybrid_motion == "Perspective":
        arr = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    else:
        arr = np.array([[1., 0., 0.], [0., 1., 0.]])
    return arr

def get_translation_matrix_from_images(i1, i2, dimensions, hybrid_motion, max_corners=200, quality_level=0.01, min_distance=30, block_size=3):
    img1 = cv2.imread(i1, 0)
    img2 = cv2.imread(i2, 0)
    img1 = cv2.resize(img1, (dimensions[0], dimensions[1]), cv2.INTER_AREA)
    img2 = cv2.resize(img2, (dimensions[0], dimensions[1]), cv2.INTER_AREA)
    
    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(img1,
                                        maxCorners=max_corners,
                                        qualityLevel=quality_level,
                                        minDistance=min_distance,
                                        blockSize=block_size)

    if prev_pts is None or len(prev_pts) < 8:
        return get_hybrid_motion_default_matrix(hybrid_motion)

    # Get optical flow
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, prev_pts, None) 
   
    # Filter only valid points
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    if len(prev_pts) < 8 or len(curr_pts) < 8:
        return get_hybrid_motion_default_matrix(hybrid_motion)
    
    if hybrid_motion == "Perspective":  # Perspective - Find the transformation between points
        transformation_matrix, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
        return transformation_matrix
    else: # Affine - Compute a rigid transformation (without depth, only scale + rotation + translation)
        transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        return transformation_rigid_matrix

def get_flow_from_images(img1, img2, dimensions, method):
    i1 = cv2.imread(img1)
    i2 = cv2.imread(img2)
    i1 = cv2.resize(i1, (dimensions[0], dimensions[1]), cv2.INTER_AREA)
    i2 = cv2.resize(i2, (dimensions[0], dimensions[1]), cv2.INTER_AREA)
    if method == "DenseRLOF":
        r = get_flow_from_images_Dense_RLOF(i1, i2)
    elif method == "SF":
        r = get_flow_from_images_SF(i1, i2)
    elif method =="Farneback":
        r = get_flow_from_images_Farneback(i1, i2)
    return r
        
def get_flow_from_images_Farneback(img1, img2, last_flow=None, pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2):
    i1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flags = 0 # flags = cv2.OPTFLOW_USE_INITIAL_FLOW    
    flow = cv2.calcOpticalFlowFarneback(i1, i2, last_flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    return flow

def get_flow_from_images_Dense_RLOF(i1, i2, last_flow=None):
    return cv2.optflow.calcOpticalFlowDenseRLOF(i1, i2, flow = last_flow)

def get_flow_from_images_SF(i1, i2, last_flow=None):
    layers = 3
    averaging_block_size = 2
    max_flow = 4
    return cv2.optflow.calcOpticalFlowSF(i1, i2, layers, averaging_block_size, max_flow)

def draw_flow_lines_in_grid_in_color(img, flow, step=8, magnitude_multiplier=1, min_magnitude = 1, max_magnitude = 100):
    flow = flow * magnitude_multiplier
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = img.copy()  # Create a copy of the input image
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

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
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, thickness=1, tipLength=0.2)    

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