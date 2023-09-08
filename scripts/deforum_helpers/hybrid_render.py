from .hybrid_video import (get_matrix_for_hybrid_motion_prev, get_matrix_for_hybrid_motion,
                            get_flow_for_hybrid_motion_prev, get_flow_for_hybrid_motion,
                            image_transform_ransac, image_transform_optical_flow)

# hybrid video motion in cadence loop goes right after animation warping, flow from last frame to current frame (skips first frame)
def hybrid_motion_for_cadence(turbo_prev_image, turbo_next_image, advance_prev, advance_next, args, anim_args, tween_frame_idx, prev_img, prev_flow, hybrid_flow_factor, inputfiles, hybrid_frame_path, raft_model):
    idx_modifier = -1 if anim_args.hybrid_motion_behavior == "Before Generation" else 0
    if anim_args.hybrid_motion in ['Affine', 'Perspective']:
        if anim_args.hybrid_motion_use_prev_img:
            matrix = get_matrix_for_hybrid_motion_prev(tween_frame_idx + idx_modifier, (args.W, args.H), inputfiles, prev_img, anim_args.hybrid_motion)
        else:
            matrix = get_matrix_for_hybrid_motion(tween_frame_idx + idx_modifier, (args.W, args.H), inputfiles, anim_args.hybrid_motion)

        if advance_prev:
            turbo_prev_image = image_transform_ransac(turbo_prev_image, matrix, anim_args.hybrid_motion)
        if advance_next:
            turbo_next_image = image_transform_ransac(turbo_next_image, matrix, anim_args.hybrid_motion)
    elif anim_args.hybrid_motion in ['Optical Flow']:
        if anim_args.hybrid_motion_use_prev_img:
            flow = get_flow_for_hybrid_motion_prev(tween_frame_idx + idx_modifier, (args.W, args.H), inputfiles, hybrid_frame_path, prev_flow, prev_img, anim_args.hybrid_flow_method, raft_model,
                                                    anim_args.hybrid_flow_consistency, anim_args.hybrid_consistency_blur, anim_args.hybrid_comp_save_extra_frames)
        else:
            flow = get_flow_for_hybrid_motion(tween_frame_idx + idx_modifier, (args.W, args.H), inputfiles, hybrid_frame_path, prev_flow, anim_args.hybrid_flow_method, raft_model,
                                                anim_args.hybrid_flow_consistency, anim_args.hybrid_consistency_blur, anim_args.hybrid_comp_save_extra_frames)
        if advance_prev:
            turbo_prev_image = image_transform_optical_flow(turbo_prev_image, flow, hybrid_flow_factor)
        if advance_next:
            turbo_next_image = image_transform_optical_flow(turbo_next_image, flow, hybrid_flow_factor)
        prev_flow = flow
    return turbo_prev_image, turbo_next_image, prev_flow

def hybrid_motion_before_generation(prev_img, prev_flow, frame_idx, args, anim_args, inputfiles, hybrid_frame_path, hybrid_flow_factor, raft_model):
    if anim_args.hybrid_motion in ['Affine', 'Perspective']:
        if anim_args.hybrid_motion_use_prev_img:
            matrix = get_matrix_for_hybrid_motion_prev(frame_idx - 1, (args.W, args.H), inputfiles, prev_img, anim_args.hybrid_motion)
        else:
            matrix = get_matrix_for_hybrid_motion(frame_idx - 1, (args.W, args.H), inputfiles, anim_args.hybrid_motion)
        prev_img = image_transform_ransac(prev_img, matrix, anim_args.hybrid_motion)
    if anim_args.hybrid_motion in ['Optical Flow']:
        if anim_args.hybrid_motion_use_prev_img:
            flow = get_flow_for_hybrid_motion_prev(frame_idx - 1, (args.W, args.H), inputfiles, hybrid_frame_path, prev_flow, prev_img, anim_args.hybrid_flow_method, raft_model,
                                                    anim_args.hybrid_flow_consistency, anim_args.hybrid_consistency_blur, anim_args.hybrid_comp_save_extra_frames)
        else:
            flow = get_flow_for_hybrid_motion(frame_idx - 1, (args.W, args.H), inputfiles, hybrid_frame_path, prev_flow, anim_args.hybrid_flow_method, raft_model,
                                                anim_args.hybrid_flow_consistency, anim_args.hybrid_consistency_blur, anim_args.hybrid_comp_save_extra_frames)
        prev_img = image_transform_optical_flow(prev_img, flow, hybrid_flow_factor)
        prev_flow = flow
    return prev_img, prev_flow

def hybrid_motion_after_generation(image, prev_img, prev_flow, frame_idx, args, anim_args, inputfiles, hybrid_frame_path, hybrid_flow_factor, raft_model):
    if anim_args.hybrid_motion in ['Affine', 'Perspective']:
        if anim_args.hybrid_motion_use_prev_img and prev_img is not None:
            matrix = get_matrix_for_hybrid_motion_prev(frame_idx, (args.W, args.H), inputfiles, prev_img, anim_args.hybrid_motion)
        else:
            matrix = get_matrix_for_hybrid_motion(frame_idx, (args.W, args.H), inputfiles, anim_args.hybrid_motion)
        image = image_transform_ransac(image, matrix, anim_args.hybrid_motion)
    if anim_args.hybrid_motion in ['Optical Flow']:
        if anim_args.hybrid_motion_use_prev_img and prev_img is not None:
            flow = get_flow_for_hybrid_motion_prev(frame_idx, (args.W, args.H), inputfiles, hybrid_frame_path, prev_flow, prev_img, anim_args.hybrid_flow_method, raft_model,
                                                    anim_args.hybrid_flow_consistency, anim_args.hybrid_consistency_blur, anim_args.hybrid_comp_save_extra_frames)
        else:
            flow = get_flow_for_hybrid_motion(frame_idx, (args.W, args.H), inputfiles, hybrid_frame_path, prev_flow, anim_args.hybrid_flow_method, raft_model,
                                                anim_args.hybrid_flow_consistency, anim_args.hybrid_consistency_blur, anim_args.hybrid_comp_save_extra_frames)
        image = image_transform_optical_flow(image, flow, hybrid_flow_factor)
        prev_flow = flow
    return image, prev_flow

