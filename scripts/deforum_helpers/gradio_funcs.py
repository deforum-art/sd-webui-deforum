import gradio as gr

def change_visibility_from_skip_video(choice):
    return gr.update(visible=False) if choice else gr.update(visible=True) 

# def hide_by_gif(choice):
    # return gr.update(visible=False) if choice == 'PIL gif' else gr.update(visible=True)

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