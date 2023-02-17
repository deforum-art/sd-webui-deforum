import gradio as gr

def change_visibility_from_skip_video(choice):
    return gr.update(visible=False) if choice else gr.update(visible=True) 

def update_r_upscale_factor(choice):
    return gr.update(value='x4', choices = ['x4']) if choice != 'realesr-animevideov3' else gr.update(value='x2', choices = ['x2', 'x3', 'x4'])

def change_perlin_visibility(choice):
    return gr.update(visible=choice=="perlin")

def change_color_coherence_video_every_N_frames_visibility(choice):
    return gr.update(visible=choice=="Video Input")

def change_seed_iter_visibility(choice):
    return gr.update(visible=choice=="iter")
    
def change_seed_schedule_visibility(choice):
    return gr.update(visible=choice=="schedule")

def disable_pers_flip_accord(choice):
    return gr.update(visible=True) if choice in ['2D','3D'] else gr.update(visible=False)

def change_max_frames_visibility(choice):
    return gr.update(visible=choice != "Video Input")
    
def change_diffusion_cadence_visibility(choice):
    return gr.update(visible=choice not in ['Video Input', 'Interpolation'])
    
def disble_3d_related_stuff(choice):
    return gr.update(visible=False) if choice != '3D' else gr.update(visible=True)
    
def enable_2d_related_stuff(choice):
    return gr.update(visible=True) if choice == '2D' else gr.update(visible=False)
    
def disable_by_interpolation(choice):
    return gr.update(visible=False) if choice in ['Interpolation'] else gr.update(visible=True)
    
def disable_by_video_input(choice):
    return gr.update(visible=False) if choice in ['Video Input'] else gr.update(visible=True)
    
def change_comp_mask_x_visibility(choice):
    return gr.update(visible=choice != "None")
    
def change_gif_button_visibility(choice):
    return gr.update(visible=False, value=False) if int(choice) > 30 else gr.update(visible=True)
    
def disable_by_hybrid_composite(choice):
    return gr.update(visible=True) if choice else gr.update(visible=False)
        
def disable_by_hybrid_composite_dynamic(choice, comp_mask_type):
    if choice == True:
        if comp_mask_type != 'None':
            return gr.update(visible=True)
    return gr.update(visible=False)
    
def disable_by_comp_mask(choice):
    return gr.update(visible=False) if choice == 'None' else gr.update(visible=True)
        
def disable_by_non_optical_flow(choice):
    return gr.update(visible=False) if choice != 'Optical Flow' else gr.update(visible=True)