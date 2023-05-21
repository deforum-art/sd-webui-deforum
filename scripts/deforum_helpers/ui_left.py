from types import SimpleNamespace
import gradio as gr
from .defaults import get_gradio_html
from .gradio_funcs import change_css, handle_change_functions
from .args import DeforumArgs, DeforumAnimArgs, ParseqArgs, DeforumOutputArgs, RootArgs, LoopArgs
from .deforum_controlnet import setup_controlnet_ui
from .ui_elements import get_tab_run, get_tab_keyframes, get_tab_prompts, get_tab_init, get_tab_hybrid, get_tab_output

def set_arg_lists():
    # convert dicts to NameSpaces for easy working (args.param instead of args['param']
    d = SimpleNamespace(**DeforumArgs())  # default args
    da = SimpleNamespace(**DeforumAnimArgs())  # default anim args
    dp = SimpleNamespace(**ParseqArgs())  # default parseq ars
    dv = SimpleNamespace(**DeforumOutputArgs())  # default video args
    dr = SimpleNamespace(**RootArgs())  # ROOT args
    dloopArgs = SimpleNamespace(**LoopArgs())  # Guided imgs args
    return d, da, dp, dv, dr, dloopArgs

def setup_deforum_left_side_ui():
    d, da, dp, dv, dr, dloopArgs = set_arg_lists()
    # set up main info accordion on top of the UI
    with gr.Accordion("Info, Links and Help", open=False, elem_id='main_top_info_accord'):
        gr.HTML(value=get_gradio_html('main'))
    # show button to hide/ show gradio's info texts for each element in the UI
    with gr.Row(variant='compact'):
        show_info_on_ui = gr.Checkbox(label="Show more info", value=d.show_info_on_ui, interactive=True)
    with gr.Blocks():
        with gr.Tabs():
            # Get main tab contents:
            tab_run_params = get_tab_run(d, da)  # Run tab
            tab_keyframes_params = get_tab_keyframes(d, da, dloopArgs)  # Keyframes tab
            tab_prompts_params = get_tab_prompts(da)  # Prompts tab
            tab_init_params = get_tab_init(d, da, dp)  # Init tab
            controlnet_dict = setup_controlnet_ui()  # ControlNet tab
            tab_hybrid_params = get_tab_hybrid(da)  # Hybrid tab
            tab_output_params = get_tab_output(da, dv)  # Output tab
            # add returned gradio elements from main tabs to locals()
            for key, value in {**tab_run_params, **tab_keyframes_params, **tab_prompts_params, **tab_init_params, **controlnet_dict, **tab_hybrid_params, **tab_output_params}.items():
                locals()[key] = value

    # Gradio's Change functions - hiding and renaming elements based on other elements
    show_info_on_ui.change(fn=change_css, inputs=show_info_on_ui, outputs=gr.outputs.HTML())
    handle_change_functions(locals())

    return locals()
