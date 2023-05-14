import gradio as gr
# print (cnet_1.get_modules())

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

def hide_ui_by_cn_status(choice):
    return gr.update(visible=True) if choice else gr.update(visible=False)
    
def hide_file_textboxes(choice):
    return gr.update(visible=False) if choice else gr.update(visible=True)
    
class ToolButton(gr.Button, gr.components.FormComponent):
        """Small button with single emoji as text, fits inside gradio forms"""
        def __init__(self, **kwargs):
            super().__init__(variant="tool", **kwargs)

        def get_block_name(self):
            return "button"

model_free_preprocessors = ["reference_only"]
flag_preprocessor_resolution = "Preprocessor Resolution"
preprocessor_sliders_config = {
    "none": [],
    "inpaint": [],
    "reference_only": [],
    "canny": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        },
        {
            "name": "Canny Low Threshold",
            "value": 100,
            "min": 1,
            "max": 255
        },
        {
            "name": "Canny High Threshold",
            "value": 200,
            "min": 1,
            "max": 255
        },
    ],
    "mlsd": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        },
        {
            "name": "MLSD Value Threshold",
            "min": 0.01,
            "max": 2.0,
            "value": 0.1,
            "step": 0.01
        },
        {
            "name": "MLSD Distance Threshold",
            "min": 0.01,
            "max": 20.0,
            "value": 0.1,
            "step": 0.01
        }
    ],
    "hed": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "scribble_hed": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "hed_safe": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "openpose": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "openpose_full": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "segmentation": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "depth": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "depth_leres": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        },
        {
            "name": "Remove Near %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1,
        },
        {
            "name": "Remove Background %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1,
        }
    ],
    "depth_leres++": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        },
        {
            "name": "Remove Near %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1,
        },
        {
            "name": "Remove Background %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1,
        }
    ],
    "normal_map": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        },
        {
            "name": "Normal Background Threshold",
            "min": 0.0,
            "max": 1.0,
            "value": 0.4,
            "step": 0.01
        }
    ],
    "threshold": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        },
        {
            "name": "Binarization Threshold",
            "min": 0,
            "max": 255,
            "value": 127
        }
    ],

    "scribble_xdog": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        },
        {
            "name": "XDoG Threshold",
            "min": 1,
            "max": 64,
            "value": 32,
        }
    ],
    "tile_resample": [
        None,
        {
            "name": "Down Sampling Rate",
            "value": 1.0,
            "min": 1.0,
            "max": 8.0,
            "step": 0.01
        }
    ],
    "color": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048,
        }
    ],
    "mediapipe_face": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048,
        },
        {
            "name": "Max Faces",
            "value": 1,
            "min": 1,
            "max": 10,
            "step": 1
        },
        {
            "name": "Min Face Confidence",
            "value": 0.5,
            "min": 0.01,
            "max": 1.0,
            "step": 0.01
        }
    ],
}
def build_sliders(module, pp):
    

    grs = []
    if module not in preprocessor_sliders_config:
        grs += [
            gr.update(label=flag_preprocessor_resolution, value=512, minimum=64, maximum=2048, step=1, visible=not pp, interactive=not pp),
            gr.update(visible=False, interactive=False),
            gr.update(visible=False, interactive=False),
            gr.update(visible=True)
        ]
    else:
        for slider_config in preprocessor_sliders_config[module]:
            if isinstance(slider_config, dict):
                visible = True
                if slider_config['name'] == flag_preprocessor_resolution:
                    visible = not pp
                grs.append(gr.update(
                    label=slider_config['name'],
                    value=slider_config['value'],
                    minimum=slider_config['min'],
                    maximum=slider_config['max'],
                    step=slider_config['step'] if 'step' in slider_config else 1,
                    visible=visible,
                    interactive=visible))
            else:
                grs.append(gr.update(visible=False, interactive=False))
        while len(grs) < 3:
            grs.append(gr.update(visible=False, interactive=False))
        grs.append(gr.update(visible=True))
    if module in model_free_preprocessors:
        grs += [gr.update(visible=False, value='None'), gr.update(visible=False)]
    else:
        grs += [gr.update(visible=True), gr.update(visible=True)]
    return grs