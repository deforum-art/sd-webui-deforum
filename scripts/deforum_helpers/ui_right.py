# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

from .args import DeforumOutputArgs, get_component_names, get_settings_component_names
from modules.shared import opts, state
from modules.ui import create_output_panel, wrap_gradio_call
from modules.call_queue import wrap_gradio_gpu_call
from .run_deforum import run_deforum
from .settings import save_settings, load_all_settings, load_video_settings, check_file_exists, overwrite_settings
from .general_utils import get_deforum_version
from .ui_left import setup_deforum_left_side_ui
from scripts.deforum_extend_paths import deforum_sys_extend
import gradio as gr

def on_ui_tabs():
    # extend paths using sys.path.extend so we can access all of our files and folders
    deforum_sys_extend()
    # set text above generate button
    i1_store_backup = f"<p style=\"text-align:center;font-weight:bold;margin-bottom:0em\">Deforum extension for auto1111 — version 3.0 | Git commit: {get_deforum_version()}</p>"
    i1_store = i1_store_backup

    with gr.Blocks(analytics_enabled=False) as deforum_interface:
        components = {}
        dummy_component = gr.Label(visible=False)
        with gr.Row(elem_id='deforum_progress_row').style(equal_height=False, variant='compact'):
            with gr.Column(scale=1, variant='panel'):
                # setting the left side of the ui:
                components = setup_deforum_left_side_ui()
            with gr.Column(scale=1, variant='compact'):
                with gr.Row(variant='compact'):
                    btn = gr.Button("Click here after the generation to show the video")
                    components['btn'] = btn
                    close_btn = gr.Button("Close the video", visible=False)
                with gr.Row(variant='compact'):
                    i1 = gr.HTML(i1_store, elem_id='deforum_header')
                    components['i1'] = i1
                    def show_vid(): # Show video button related func
                        from .run_deforum import last_vid_data # get latest vid preview data (this import needs to stay inside the function!)
                        return {
                            i1: gr.update(value=last_vid_data, visible=True),
                            close_btn: gr.update(visible=True),
                            btn: gr.update(value="Update the video", visible=True),
                        }
                    btn.click(
                        fn=show_vid,
                        inputs=[],
                        outputs=[i1, close_btn, btn],
                        )
                    def close_vid(): # Close video button related func
                        return {
                            i1: gr.update(value=i1_store_backup, visible=True),
                            close_btn: gr.update(visible=False),
                            btn: gr.update(value="Click here after the generation to show the video", visible=True),
                        }
                    
                    close_btn.click(
                        fn=close_vid,
                        inputs=[],
                        outputs=[i1, close_btn, btn],
                        )
                id_part = 'deforum'
                with gr.Row(elem_id=f"{id_part}_generate_box", variant='compact'):
                    skip = gr.Button('Pause/Resume', elem_id=f"{id_part}_skip", visible=False)
                    interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt", visible=True)
                    interrupting = gr.Button('Interrupting...', elem_id=f"{id_part}_interrupting", elem_classes="generate-box-interrupting", tooltip="Interrupting generation...")
                    submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                    skip.click(
                        fn=lambda: state.skip(),
                        inputs=[],
                        outputs=[],
                    )

                    interrupt.click(
                        fn=lambda: state.interrupt(),
                        inputs=[],
                        outputs=[],
                    )
                    
                    interrupting.click(
                        fn=lambda: state.interrupt(),
                        inputs=[],
                        outputs=[],
                    )
                
                output_panel = create_output_panel("deforum", opts.outdir_img2img_samples)
                if isinstance(output_panel, tuple):
                    deforum_gallery = output_panel[0]
                    generation_info = output_panel[1]
                    html_info = output_panel[2]
                else:
                    deforum_gallery = output_panel.gallery
                    generation_info = output_panel.generation_info
                    html_info = output_panel.infotext

                with gr.Row(variant='compact'):
                    settings_path = gr.Textbox("deforum_settings.txt", elem_id='deforum_settings_path', label="Settings File", info="settings file path can be relative to webui folder OR full - absolute")
                with gr.Row(variant="compact"):
                    html_msg = gr.HTML('html component', visible=False)
                with gr.Row(variant='compact'):
                    overwrite_settings_btn = gr.Button('Overwrite', visible=False, variant="secondary", elem_id='deforum_overwrite_settings_btn')
                    cancel_save_btn = gr.Button('Cancel Save', visible=False, variant="stop", elem_id='deforum_cancel_save_btn')
                    save_settings_btn = gr.Button('Save Settings', elem_id='deforum_save_settings_btn')
                    load_settings_btn = gr.Button('Load All Settings', elem_id='deforum_load_settings_btn')
                    load_video_settings_btn = gr.Button('Load Video Settings', elem_id='deforum_load_video_settings_btn')

        component_list = [components[name] for name in get_component_names()]

        submit.click(
                    fn=wrap_gradio_gpu_call(run_deforum),
                    _js="submit_deforum",
                    inputs=[dummy_component, dummy_component] + component_list,
                    outputs=[
                         deforum_gallery,
                         components["resume_timestring"],
                         generation_info,
                         html_info                 
                    ],
                )
        
        settings_component_list = [components[name] for name in get_settings_component_names()]
        video_settings_component_list = [components[name] for name in list(DeforumOutputArgs().keys())]

        save_settings_btn.click(
            fn=wrap_gradio_call(check_file_exists), 
            inputs=[settings_path] + settings_component_list + video_settings_component_list,
            outputs=[save_settings_btn, overwrite_settings_btn, cancel_save_btn, html_msg, html_msg]
        )

        overwrite_settings_btn.click(
            fn=wrap_gradio_call(overwrite_settings), 
            inputs=[settings_path] + settings_component_list + video_settings_component_list, 
            outputs=[save_settings_btn, overwrite_settings_btn, cancel_save_btn, html_msg, html_msg]
        )
        
        cancel_save_btn.click(
            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)], 
            None, 
            [save_settings_btn, overwrite_settings_btn, cancel_save_btn, html_msg]
        )
        
        load_settings_btn.click(
            fn=wrap_gradio_call(lambda *args, **kwargs: load_all_settings(*args, ui_launch=False, **kwargs)),
            inputs=[settings_path] + settings_component_list,
            outputs=settings_component_list,
        )

        load_video_settings_btn.click(
            fn=wrap_gradio_call(load_video_settings),
            inputs=[settings_path] + video_settings_component_list,
            outputs=video_settings_component_list,
        )
        
    # handle persistent settings - load the persistent file upon UI launch
    def trigger_load_general_settings():
        print("Loading general settings...")
        wrapped_fn = wrap_gradio_call(lambda *args, **kwargs: load_all_settings(*args, ui_launch=True, **kwargs))
        inputs = [settings_path.value] + [component.value for component in settings_component_list]
        outputs = settings_component_list
        updated_values = wrapped_fn(*inputs, *outputs)[0]
        settings_component_name_to_obj = {name: component for name, component in zip(get_settings_component_names(), settings_component_list)}
        for key, value in updated_values.items():
            settings_component_name_to_obj[key].value = value['value']
    # actually check persistent setting status
    if opts.data.get("deforum_enable_persistent_settings", False):
        trigger_load_general_settings()
        
    return [(deforum_interface, "Deforum", "deforum_interface")]
