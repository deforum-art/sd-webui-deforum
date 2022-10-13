import scripts.deforum.args as deforum_args

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from types import SimpleNamespace

class Script(scripts.Script):

    def title(self):
        return "Deforum v0.5-webui-beta"

    def ui(self, is_img2img):
        d = SimpleNamespace(**deforum_args.DeforumArgs())
        da = SimpleNamespace(**deforum_args.DeforumAnimArgs()) #default args
        gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Deforum v0.5-webui-beta</p>")
        gr.HTML("<p style=\"margin-bottom:0.75em\">Made by deforum.github.io</p>")
        gr.HTML("<p style=\"margin-bottom:0.75em\">Original Deforum Github repo  github.com/deforum/stable-diffusion</p>")
        gr.HTML("<p style=\"margin-bottom:0.75em\">This WIP fork for auto1111's webui github.com/kabachuha/stable-diffusion/tree/automatic1111-webui</p>")
        gr.HTML("<p style=\"margin-bottom:0.75em\">Join the official Deforum Discord discord.gg/deforum to share your creations and suggestions</p>")
        gr.HTML("<p style=\"margin-bottom:0.75em\">User guide for v0.5 docs.google.com/document/d/1pEobUknMFMkn8F5TMsv8qRzamXX_75BShMMXV8IFslI/edit</p>")
        gr.HTML("<p style=\"margin-bottom:0.75em\">Math keyframing explanation docs.google.com/document/d/1pfW1PwbDIuW0cv-dnuyYj1UzPqe23BlSLTJsqazffXM/edit?usp=sharing</p>")
        
        
        gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Import settings from file</p>")
        with gr.Row():
            override_settings_with_file = gr.Checkbox(label="Override settings", value=False, interactive=True)
            custom_settings_file = gr.Textbox(label="Custom settings file", lines=1, interactive=True)
            #TODO make a button
            
        # Animation settings START
        #TODO make a some sort of the original dictionary parsing
        gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Animation settings</p>")
        with gr.Row():
            animation_mode = gr.Dropdown(label="animation_mode", choices=['None', '2D', '3D', 'Video Input', 'Interpolation'], value=da.animation_mode, type="index", elem_id="animation_mode", interactive=True)
            max_frames = gr.Number(label="max_frames", value=da.max_frames, interactive=True)
            border = gr.Dropdown(label="border", choices=['replicate', 'wrap'], value=da.border, type="index", elem_id="border", interactive=True)
        
        
        gr.HTML("<p style=\"margin-bottom:0.75em\">Motion parameters:</p>")
        gr.HTML("<p style=\"margin-bottom:0.75em\">2D and 3D settings</p>")
        with gr.Row():
            angle = gr.Textbox(label="angle", lines=1, value = da.angle, interactive=True)
        with gr.Row():
            zoom = gr.Textbox(label="zoom", lines=1, value = da.zoom, interactive=True)
        with gr.Row():
            translation_x = gr.Textbox(label="translation_x", lines=1, value = da.translation_x, interactive=True)
        with gr.Row():
            translation_y = gr.Textbox(label="translation_y", lines=1, value = da.translation_y, interactive=True)
        gr.HTML("<p style=\"margin-bottom:0.75em\">3D settings</p>")
        with gr.Row():
            translation_z = gr.Textbox(label="translation_z", lines=1, value = da.translation_z, interactive=True)
        with gr.Row():
            rotation_3d_x = gr.Textbox(label="rotation_3d_x", lines=1, value = da.rotation_3d_x, interactive=True)
        with gr.Row():
            rotation_3d_y = gr.Textbox(label="rotation_3d_y", lines=1, value = da.rotation_3d_y, interactive=True)
        with gr.Row():
            rotation_3d_z = gr.Textbox(label="rotation_3d_z", lines=1, value = da.rotation_3d_z, interactive=True)
        gr.HTML("<p style=\"margin-bottom:0.75em\">Prespective flip — Low VRAM pseudo-3D mode:</p>")
        with gr.Row():
            flip_2d_perspective = gr.Checkbox(label="flip_2d_perspective", value=da.flip_2d_perspective, interactive=True)
        with gr.Row():
            perspective_flip_theta = gr.Textbox(label="perspective_flip_theta", lines=1, value = da.perspective_flip_theta, interactive=True)
        with gr.Row():
            perspective_flip_phi = gr.Textbox(label="perspective_flip_phi", lines=1, value = da.perspective_flip_phi, interactive=True)
        with gr.Row():
            perspective_flip_gamma = gr.Textbox(label="perspective_flip_gamma", lines=1, value = da.perspective_flip_gamma, interactive=True)
        with gr.Row():
            perspective_flip_fv = gr.Textbox(label="perspective_flip_fv", lines=1, value = da.perspective_flip_fv, interactive=True)
        gr.HTML("<p style=\"margin-bottom:0.75em\">Generation settings:</p>")
        with gr.Row():
            noise_schedule = gr.Textbox(label="noise_schedule", lines=1, value = da.noise_schedule, interactive=True)
        with gr.Row():
            strength_schedule = gr.Textbox(label="strength_schedule", lines=1, value = da.strength_schedule, interactive=True)
        with gr.Row():
            contrast_schedule = gr.Textbox(label="contrast_schedule", lines=1, value = da.contrast_schedule, interactive=True)
#TODO
#        with gr.Row():
#            seed_schedule = gr.Textbox(label="seed_schedule", lines=1, value = da.seed_schedule, interactive=True)
#        with gr.Row():
#            scale_schedule = gr.Textbox(label="scale_schedule", lines=1, value = da.scale_schedule, interactive=True)
        
        gr.HTML("<p style=\"margin-bottom:0.75em\">Coherence:</p>")
        with gr.Row():
            color_coherence = gr.Dropdown(label="color_coherence", choices=['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'], value=da.color_coherence, type="index", elem_id="color_coherence", interactive=True)
            diffusion_cadence = gr.Slider(label="diffusion_cadence", minimum=1, maximum=8, step=1, value=1, interactive=True)
            
        gr.HTML("<p style=\"margin-bottom:0.75em\">3D Depth Warping:</p>")
        with gr.Row():
            use_depth_warping = gr.Checkbox(label="use_depth_warping", value=False, interactive=True)
        with gr.Row():
            midas_weight = gr.Number(label="midas_weight", value=da.midas_weight, interactive=True)
            near_plane = gr.Number(label="near_plane", value=da.near_plane, interactive=True)
            far_plane = gr.Number(label="far_plane", value=da.far_plane, interactive=True)
            fov = gr.Number(label="fov", value=da.fov, interactive=True)
            padding_mode = gr.Dropdown(label="padding_mode", choices=['border', 'reflection', 'zeros'], value=da.padding_mode, type="index", elem_id="padding_mode", interactive=True)
            sampling_mode = gr.Dropdown(label="sampling_mode", choices=['bicubic', 'bilinear', 'nearest'], value=da.sampling_mode, type="index", elem_id="sampling_mode", interactive=True)
            save_depth_maps = gr.Checkbox(label="save_depth_maps", value=da.save_depth_maps, interactive=True)
        
        gr.HTML("<p style=\"margin-bottom:0.75em\">Video Input:</p>")
        with gr.Row():
            video_init_path = gr.Textbox(label="video_init_path", lines=1, value = da.video_init_path, interactive=True)
        with gr.Row():
            extract_nth_frame = gr.Number(label="extract_nth_frame", value=da.extract_nth_frame, interactive=True)
            overwrite_extracted_frames = gr.Checkbox(label="overwrite_extracted_frames", value=False, interactive=True)
            use_mask_video = gr.Checkbox(label="use_mask_video", value=False, interactive=True)
        with gr.Row():
            video_mask_path = gr.Textbox(label="video_mask_path", lines=1, value = da.video_mask_path, interactive=True)
        
        gr.HTML("<p style=\"margin-bottom:0.75em\">Interpolation:</p>")
        with gr.Row():
            interpolate_key_frames = gr.Checkbox(label="interpolate_key_frames", value=da.interpolate_key_frames, interactive=True)
            interpolate_x_frames = gr.Number(label="interpolate_x_frames", value=da.interpolate_x_frames, interactive=True)
        
        gr.HTML("<p style=\"margin-bottom:0.75em\">Resume animation:</p>")
        with gr.Row():
            resume_from_timestring = gr.Checkbox(label="resume_from_timestring", value=da.resume_from_timestring, interactive=True)
            resume_timestring = gr.Textbox(label="resume_timestring", lines=1, value = da.resume_timestring, interactive=True)
        # Animation settings END
        
        # Prompts settings START
        
        gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Prompts</p>")
        gr.HTML("<p style=\"margin-bottom:0.75em\">`animation_mode: None` batches on list of *prompts*.</p>")
        gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">*Important change!*</p>")
        gr.HTML("<p style=\"font-weight:italic;margin-bottom:0.75em\">This script uses the built-in webui weighting settings.</p>")
        gr.HTML("<p style=\"font-weight:italic;margin-bottom:0.75em\">So if you want to use math functions as prompt weights,</p>")
        gr.HTML("<p style=\"font-weight:italic;margin-bottom:0.75em\">keep the values above zero in both parts</p>")
        gr.HTML("<p style=\"font-weight:italic;margin-bottom:0.75em\">Negative prompt part can be specified with --negative</p>")
        with gr.Row():
            prompts = gr.Textbox(label="prompts", lines=8, interactive=True, value = deforum_args.prompts)
        with gr.Row():
            animation_prompts = gr.Textbox(label="animation_prompts", lines=8, interactive=True, value = deforum_args.animation_prompts)
        
        # Prompts settings END
        
        gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Run settings</p>")
        
        # Sampling settings START
        gr.HTML("<p style=\"margin-bottom:0.75em\">Sampling settings</p>")
        gr.HTML("<p style=\"margin-bottom:0.75em\">The following settings have already been set up in the webui</p>")
        gr.HTML("<p style=\"margin-bottom:0.75em\">Do you want to override them?</p>")
        gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">WIP *Doesn't do anything at the moment!*</p>") #TODO
        with gr.Row():
            override_webui_with_these = gr.Checkbox(label="override_webui_with_these", value=False, interactive=True)
        gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">W, H, seed, sampler, steps, scale, ddim_eta, n_batch, make_grid, grid_rows</p>")
        # Sampling settings END
        
        # Batch settings START
        gr.HTML("<p style=\"margin-bottom:0.75em\">Batch settings</p>")
        with gr.Row():
            batch_name = gr.Textbox(label="batch_name", lines=1, interactive=True, value = d.batch_name)
        with gr.Row():    
            filename_format = gr.Textbox(label="filename_format", lines=1, interactive=True, value = d.filename_format)
        with gr.Row():
            seed_behavior = gr.Dropdown(label="seed_behavior", choices=['iter', 'fixed', 'random'], value=d.seed_behavior, type="index", elem_id="seed_behavior", interactive=True)
        # output - made in run
        # Batch settings END
        
        # Init settings START
        gr.HTML("<p style=\"margin-bottom:0.75em\">Init settings</p>")
        with gr.Row():
            use_init = gr.Checkbox(label="use_init", value=is_img2img, interactive=True, visible=True)
            from_img2img_instead_of_link = gr.Checkbox(label="from_img2img_instead_of_link", value=is_img2img, interactive=True, visible=is_img2img)
        with gr.Row():
            strength_0_no_init = gr.Checkbox(label="strength_0_no_init", value=True, interactive=True)
            strength = gr.Slider(label="strength", minimum=0, maximum=1, step=0.02, value=0, interactive=True)
        with gr.Row():
            init_image = gr.Textbox(label="init_image", lines=1, interactive=True, value = d.init_image)
        with gr.Row():
            use_mask = gr.Checkbox(label="use_mask", value=d.use_mask, interactive=True)
            use_alpha_as_mask = gr.Checkbox(label="use_alpha_as_mask", value=d.use_alpha_as_mask, interactive=True)
            invert_mask = gr.Checkbox(label="invert_mask", value=d.invert_mask, interactive=True)
            overlay_mask = gr.Checkbox(label="overlay_mask", value=d.overlay_mask, interactive=True)
        with gr.Row():
            mask_file = gr.Textbox(label="mask_file", lines=1, interactive=True, value = d.mask_file)
        with gr.Row():
            mask_brightness_adjust = gr.Number(label="mask_brightness_adjust", value=d.mask_brightness_adjust, interactive=True)
            mask_overlay_blur = gr.Number(label="mask_overlay_blur", value=d.mask_overlay_blur, interactive=True)
        # Init settings END
        
        return [interpolate_x_frames, batch_name]


    def run(self, p, interpolate_x_frames, batch_name):
        print('Hello, deforum!')
        
        outdir =  os.path.join(p.outpath_samples, batch_name)
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        display_result_data = ["Hello, deforum!"]

        return Processed(p, display_result_data)
    
    