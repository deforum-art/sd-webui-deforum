import gradio as gr

def change_visibility_from_skip_video(choice):
    if choice:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True) 

def hide_by_gif(choice):
    if choice == 'PIL gif':
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)
        
def change_color_coherence_video_every_N_frames_visibility(choice):
    return gr.update(visible=choice=="Video Input")

def change_seed_iter_visibility(choice):
    return gr.update(visible=choice=="iter")
def change_seed_schedule_visibility(choice):
    return gr.update(visible=choice=="schedule")

def update_motion_accord_name(choice):
    if choice == '2D':
        return gr.update(label = '2D Motion')
    elif choice == '3D':
        return gr.update(label = '3D Motion, Depth & FOV')
    else:
        return gr.update()

def disable_motion_accord(choice):
    if choice in ['2D','3D']:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def change_max_frames_visibility(choice):
    return gr.update(visible=choice != "Video Input")
def change_diffusion_cadence_visibility(choice):
    return gr.update(visible=choice not in ['Video Input', 'Interpolation'])
def disble_3d_related_stuff(choice):
    if choice != '3D':
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)
def enable_2d_related_stuff(choice):
    if choice == '2D':
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
def disable_by_interpolation(choice):
    if choice in ['Interpolation']:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)
def disable_by_video_input(choice):
    if choice in ['Video Input']:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)
def disable_when_not_in_2d_or_3d_modes(choice):
    if choice not in ['2D','3D']:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)

def change_comp_mask_x_visibility(choice):
    return gr.update(visible=choice != "None")