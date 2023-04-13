import gradio as gr

    # *** TODO: re-enable table printing! disabled only temp! 13-04-23 ***
    # table = Table(title="ControlNet params",padding=0, box=box.ROUNDED)

    # TODO: auto infer the names and the values for the table
    # field_names = []
    # field_names += ["module", "model", "weight", "inv", "guide_start", "guide_end", "guess", "resize", "rgb_bgr", "proc res", "thr a", "thr b"]
    # for field_name in field_names:
        # table.add_column(field_name, justify="center")
    
    # cn_model_name = str(controlnet_args.cn_1_model)

    # rows = []
    # rows += [controlnet_args.cn_1_module, cn_model_name[len('control_'):] if 'control_' in cn_model_name else cn_model_name, controlnet_args.cn_1_weight, controlnet_args.cn_1_invert_image, controlnet_args.cn_1_guidance_start, controlnet_args.cn_1_guidance_end, controlnet_args.cn_1_guess_mode, controlnet_args.cn_1_resize_mode, controlnet_args.cn_1_rgbbgr_mode, controlnet_args.cn_1_processor_res, controlnet_args.cn_1_threshold_a, controlnet_args.cn_1_threshold_b]
    # rows = [str(x) for x in rows]

    # table.add_row(*rows)
    # console.print(table)
       
class ToolButton(gr.Button, gr.components.FormComponent):
        """Small button with single emoji as text, fits inside gradio forms"""
        def __init__(self, **kwargs):
            super().__init__(variant="tool", **kwargs)

        def get_block_name(self):
            return "button"

def build_sliders(module):
    if module == "canny":
        return [
            gr.update(label="Annotator resolution", value=512, minimum=64, maximum=2048, step=1, interactive=True),
            gr.update(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1, interactive=True),
            gr.update(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1, interactive=True),
            gr.update(visible=True)
        ]
    elif module == "mlsd": #Hough
        return [
            gr.update(label="Hough Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
            gr.update(label="Hough value threshold (MLSD)", minimum=0.01, maximum=2.0, value=0.1, step=0.01, interactive=True),
            gr.update(label="Hough distance threshold (MLSD)", minimum=0.01, maximum=20.0, value=0.1, step=0.01, interactive=True),
            gr.update(visible=True)
        ]
    elif module in ["hed", "fake_scribble"]:
        return [
            gr.update(label="HED Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]
    elif module in ["openpose", "openpose_hand", "segmentation"]:
        return [
            gr.update(label="Annotator Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]
    elif module == "depth":
        return [
            gr.update(label="Midas Resolution", minimum=64, maximum=2048, value=384, step=1, interactive=True),
            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]
    elif module in ["depth_leres", "depth_leres_boost"]:
        return [
            gr.update(label="LeReS Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
            gr.update(label="Remove Near %", value=0, minimum=0, maximum=100, step=0.1, interactive=True),
            gr.update(label="Remove Background %", value=0, minimum=0, maximum=100, step=0.1, interactive=True),
            gr.update(visible=True)
        ]
    elif module == "normal_map":
        return [
            gr.update(label="Normal Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
            gr.update(label="Normal background threshold", minimum=0.0, maximum=1.0, value=0.4, step=0.01, interactive=True),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]
    elif module == "binary":
        return [
            gr.update(label="Annotator resolution", value=512, minimum=64, maximum=2048, step=1, interactive=True),
            gr.update(label="Binary threshold", minimum=0, maximum=255, value=0, step=1, interactive=True),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]
    elif module == "color":
        return [
            gr.update(label="Annotator Resolution", value=512, minimum=64, maximum=2048, step=8, interactive=True),
            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]
    elif module == "none":
        return [
            gr.update(label="Normal Resolution", value=64, minimum=64, maximum=2048, interactive=False),
            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=False)
        ]
    else:
        return [
            gr.update(label="Annotator resolution", value=512, minimum=64, maximum=2048, step=1, interactive=True),
            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]