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