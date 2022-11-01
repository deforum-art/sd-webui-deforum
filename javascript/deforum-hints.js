// mouseover tooltips for various UI elements

deforum_titles = {
    "animation_mode": "selects the type of animation",
        "2D": "only 2D motion parameters will be used, but this mode uses the least amount of VRAM. You can optionally enable flip_2d_perspective to enable some psuedo-3d animation parameters while in 2D mode.",
        "3D": "enables all 3D motion parameters.",
        "Video Input": "will ignore all motion parameters and attempt to reference a video loaded into the runtime, specified by the video_init_path. Max_frames is ignored during video_input mode, and instead, follows the number of frames pulled from the video’s length. Resume_from_timestring is NOT available with Video_Input mode.",
    "max_frames": "the maximum number of output images to be created",
    "border": "controls handling method of pixels to be generated when the image is smaller than the frame.",
        "wrap": "pulls pixels from the opposite edge of the image",
        "replicate": "repeats the edge of the pixels, and extends them. Animations with quick motion may yield lines where this border function was attempting to populate pixels into the empty space created.",
    "angle": "2D operator to rotate canvas clockwise/anticlockwise in degrees per frame",
    "zoom": "2D operator that scales the canvas size, multiplicatively. [static = 1.0]",
    "translation_x": "2D & 3D operator to move canvas left/right in pixels per frame",
    "translation_y": "2D & 3D operator to move canvas up/down in pixels per frame", 
    "translation_z": "3D operator to move canvas towards/away from view [speed set by FOV]",
    "rotation_3d_x": "3D operator to tilt canvas up/down in degrees per frame",
    "rotation_3d_y": "3D operator to pan canvas left/right in degrees per frame",
    "rotation_3d_z": "3D operator to roll canvas clockwise/anticlockwise",
    "flip_2d_perspective": "enables 2D mode functions to simulate faux 3D movement",
    "perspective_flip_theta": "the roll effect angle",
    "perspective_flip_phi": "the tilt effect angle",
    "perspective_flip_gamma": "the pan effect angle",
    "perspective_flip_fv": "the 2D vanishing point of perspective (recommended range 30-160)",
    "noise_schedule": "amount of graininess to add per frame for diffusion diversity",
    "strength_schedule": "amount of presence of previous frame to influence next frame, also controls steps in the following formula [steps - (strength_schedule * steps)]",
    "contrast_schedule": "adjusts the overall contrast per frame [default neutral at 1.0]",
    "cfg_scale_schedule": "how closely the image should conform to the prompt. Lower values produce more creative results. (recommended range 5-15)",
    "fov_schedule": "adjusts the scale at which the canvas is moved in 3D by the translation_z value. [maximum range -180 to +180, with 0 being undefined. Values closer to 180 will make the image have less depth, while values closer to 0 will allow more depth]",
    "near_schedule": "",
    "far_schedule":  "",
    "seed_schedule": "allows you to specify seeds at a specific schedule, if seed_behavior is set to schedule.",
    "color_coherence": "The color coherence will attempt to sample the overall pixel color information, and trend those values analyzed in the first frame to be applied to future frames.",
        "HSV": "HSV is a good method for balancing presence of vibrant colors, but may produce unrealistic results - (ie.blue apples)",
        "LAB": "LAB is a more linear approach to mimic human perception of color space - a good default setting for most users.",
        "RGB": "RGB is good for enforcing unbiased amounts of color in each red, green and blue channel - some images may yield colorized artifacts if sampling is too low.",
    "diffusion_cadence": "The default setting of 1 will cause every frame to receive diffusion in the sequence of image outputs. A setting of 2 will only diffuse on every other frame, yet motion will still be in effect. The output of images during the cadence sequence will be automatically blended, additively and saved to the specified drive. This may improve the illusion of coherence in some workflows as the content and context of an image will not change or diffuse during frames that were skipped. Higher values of 4-8 cadence will skip over a larger amount of frames and only diffuse the “Nth” frame as set by the diffusion_cadence value. This may produce more continuity in an animation, at the cost of little opportunity to add more diffused content. In extreme examples, motion within a frame will fail to produce diverse prompt context, and the space will be filled with lines or approximations of content - resulting in unexpected animation patterns and artifacts. Video Input & Interpolation modes are not affected by diffusion_cadence.",
    "use_depth_warping": "enables instructions to warp an image dynamically in 3D mode only.",
    "midas_weight": "sets a midpoint at which a depthmap is to be drawn: range [-1 to +1]",
    "padding_mode": "instructs the handling of pixels outside the field of view as they come into the scene.",
        "border": "Border will attempt to use the edges of the canvas as the pixels to be drawn.",
        "reflection": "Reflection will attempt to approximate the image and tile/repeat pixels",
        "zeros": "Zeros will not add any new pixel information",
    "sampling_mode": "choose from Bicubic, Bilinear or Nearest modes. (Recommended: Bicubic)",
    "save_depth_maps": "will output a greyscale depth map image alongside the output images.",
	
    "video_init_path": "the directory at which your video file is located for Video Input mode only.",
    "extract_nth_frame": "during the run sequence, only frames specified by this value will be extracted, saved, and diffused upon. A value of 1 indicates that every frame is to be accounted for. Values of 2 will use every other frame for the sequence. Higher values will skip that number of frames respectively.",
    "overwrite_extracted_frames": "when enabled, will re-extract video frames each run. When using video_input mode, the run will be instructed to write video frames to the drive. If you’ve already populated the frames needed, uncheck this box to skip past redundant extraction, and immediately start the render. If you have not extracted frames, you must run at least once with this box checked to write the necessary frames.",
    "use_mask_video": "video_input mode only, enables the extraction and use of a separate video file intended for use as a mask. White areas of the extracted video frames will not be affected by diffusion, while black areas will be fully effected. Lighter/darker areas are affected dynamically.",
    "video_mask_path": "the directory in which your mask video is located.",

    "interpolate_key_frames": "selects whether to ignore prompt schedule or _x_frames.",
    "interpolate_x_frames": "the number of frames to transition thru between prompts (when interpolate_key_frames = true, then the numbers in front of the animation prompts will dynamically guide the images based on their value. If set to false, will ignore the prompt numbers and force interpole_x_frames value regardless of prompt number)",
    
    "resume_from_timestring": "instructs the run to start from a specified point",
    "resume_timestring": "the required timestamp to reference when resuming. Currently only available in 2D & 3D mode, the timestamp is saved as the settings .txt file name as well as images produced during your previous run. The format follows: yyyymmddhhmmss - a timestamp of when the run was started to diffuse."
}


onUiUpdate(function(){
	gradioApp().querySelectorAll('span, button, select, p').forEach(function(span){
		tooltip = deforum_titles[span.textContent];

		if(!tooltip){
		    tooltip = deforum_titles[span.value];
		}

		if(!tooltip){
			for (const c of span.classList) {
				if (c in deforum_titles) {
					tooltip = deforum_titles[c];
					break;
				}
			}
		}

		if(tooltip){
			span.title = tooltip;
		}
	})

	gradioApp().querySelectorAll('select').forEach(function(select){
	    if (select.onchange != null) return;

	    select.onchange = function(){
            select.title = deforum_titles[select.value] || "";
	    }
	})
})
