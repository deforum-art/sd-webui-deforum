# TODO HACK FIXME HARDCODE — as using the scripts doesn't seem to work for some reason
deforum_latest_network = None
deforum_latest_params = (None, 'placeholder to trigger the model loading')
deforum_input_image = None
from scripts.processor import unload_hed, unload_mlsd, unload_midas, unload_leres, unload_pidinet, unload_openpose, unload_uniformer, HWC3
import modules.shared as shared
import modules.devices as devices
import modules.processing as processing
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
import numpy as np
from scripts.controlnet import update_cn_models, cn_models, cn_models_names
import os
import modules.scripts as scrpts
import torch
from scripts.cldm import PlugableControlModel
from scripts.adapter import PlugableAdapter
from scripts.utils import load_state_dict
from torchvision.transforms import Resize, InterpolationMode, CenterCrop, Compose
from einops import rearrange
cn_models_dir = os.path.join(scrpts.basedir(), "models")
default_conf_adapter = os.path.join(cn_models_dir, "sketch_adapter_v14.yaml")
default_conf = os.path.join(cn_models_dir, "cldm_v15.yaml")
unloadable = {
    "hed": unload_hed,
    "fake_scribble": unload_hed,
    "mlsd": unload_mlsd,
    "depth": unload_midas,
    "depth_leres": unload_leres,
    "normal_map": unload_midas,
    "pidinet": unload_pidinet,
    "openpose": unload_openpose,
    "openpose_hand": unload_openpose,
    "segmentation": unload_uniformer,
}
deforum_latest_model_hash = ""

def restore_networks(unet):
    global deforum_latest_network
    global deforum_latest_params
    if deforum_latest_network is not None:
        print("restoring last networks")
        deforum_input_image = None
        deforum_latest_network.restore(unet)
        deforum_latest_network = None

    last_module = deforum_latest_params[0]
    if last_module is not None:
        unloadable.get(last_module, lambda:None)()

def process(p, *args):

    global deforum_latest_network
    global deforum_latest_params
    global deforum_input_image
    global deforum_latest_model_hash
    
    unet = p.sd_model.model.diffusion_model

    enabled, module, model, weight, image, scribble_mode, \
        resize_mode, rgbbgr_mode, lowvram, pres, pthr_a, pthr_b, guidance_strength = args

    if not enabled:
        restore_networks(unet)
        return

    models_changed = deforum_latest_params[1] != model \
        or deforum_latest_model_hash != p.sd_model.sd_model_hash or deforum_latest_network == None \
        or (deforum_latest_network is not None and deforum_latest_network.lowvram != lowvram)

    deforum_latest_params = (module, model)
    deforum_latest_model_hash = p.sd_model.sd_model_hash
    if models_changed:
        restore_networks(unet)
        model_path = cn_models.get(model, None)

        if model_path is None:
            raise RuntimeError(f"model not found: {model}")

        # trim '"' at start/end
        if model_path.startswith("\"") and model_path.endswith("\""):
            model_path = model_path[1:-1]

        if not os.path.exists(model_path):
            raise ValueError(f"file not found: {model_path}")

        print(f"Loading preprocessor: {module}, model: {model}")
        state_dict = load_state_dict(model_path)
        network_module = PlugableControlModel
        network_config = shared.opts.data.get("control_net_model_config", default_conf)
        if any([k.startswith("body.") for k, v in state_dict.items()]):
            # adapter model     
            network_module = PlugableAdapter
            network_config = shared.opts.data.get("control_net_model_adapter_config", default_conf_adapter)

        network = network_module(
            state_dict=state_dict, 
            config_path=network_config, 
            weight=weight, 
            lowvram=lowvram,
            base_model=unet,
        )
        network.to(p.sd_model.device, dtype=p.sd_model.dtype)
        network.hook(unet, p.sd_model)

        print(f"ControlNet model {model} loaded.")
        deforum_latest_network = network
        
    if image is not None:
        deforum_input_image = HWC3(image['image'])
        if 'mask' in image and image['mask'] is not None and not ((image['mask'][:, :, 0]==0).all() or (image['mask'][:, :, 0]==255).all()):
            print("using mask as input")
            deforum_input_image = HWC3(image['mask'][:, :, 0])
            scribble_mode = True
    else:
        # use img2img init_image as default
        deforum_input_image = getattr(p, "init_images", [None])[0]
        if deforum_input_image is None:
            raise ValueError('controlnet is enabled but no input image is given')
        deforum_input_image = HWC3(np.asarray(deforum_input_image))
            
    if scribble_mode:
        detected_map = np.zeros_like(deforum_input_image, dtype=np.uint8)
        detected_map[np.min(deforum_input_image, axis=2) < 127] = 255
        deforum_input_image = detected_map
    
    from scripts.processor import canny, midas, midas_normal, leres, hed, mlsd, openpose, pidinet, simple_scribble, fake_scribble, uniformer
    
    preprocessor = {
        "none": lambda x, *args, **kwargs: x,
        "canny": canny,
        "depth": midas,
        "depth_leres": leres,
        "hed": hed,
        "mlsd": mlsd,
        "normal_map": midas_normal,
        "openpose": openpose,
        # "openpose_hand": openpose_hand,
        "pidinet": pidinet,
        "scribble": simple_scribble,
        "fake_scribble": fake_scribble,
        "segmentation": uniformer,
    }
            
    preprocessor = preprocessor[deforum_latest_params[0]]
    h, w, bsz = p.height, p.width, p.batch_size
    if pres > 64:
        detected_map = preprocessor(deforum_input_image, res=pres, thr_a=pthr_a, thr_b=pthr_b)
    else:
        detected_map = preprocessor(deforum_input_image)
    detected_map = HWC3(detected_map)
    
    if module == "normal_map" or rgbbgr_mode:
        control = torch.from_numpy(detected_map[:, :, ::-1].copy()).float().to(devices.get_device_for("controlnet")) / 255.0
    else:
        control = torch.from_numpy(detected_map.copy()).float().to(devices.get_device_for("controlnet")) / 255.0
    
    control = rearrange(control, 'h w c -> c h w')
    detected_map = rearrange(torch.from_numpy(detected_map), 'h w c -> c h w')
    if resize_mode == "Scale to Fit (Inner Fit)":
        transform = Compose([
            Resize(h if h<w else w, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size=(h, w))
        ]) 
        control = transform(control)
        detected_map = transform(detected_map)
    elif resize_mode == "Envelope (Outer Fit)":
        transform = Compose([
            Resize(h if h>w else w, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size=(h, w))
        ]) 
        control = transform(control)
        detected_map = transform(detected_map)
    else:
        control = Resize((h,w), interpolation=InterpolationMode.BICUBIC)(control)
        detected_map = Resize((h,w), interpolation=InterpolationMode.BICUBIC)(detected_map)
        
    # for log use
    detected_map = rearrange(detected_map, 'c h w -> h w c').numpy().astype(np.uint8)
        
    # control = torch.stack([control for _ in range(bsz)], dim=0)
    deforum_latest_network.notify(control, weight, guidance_strength)

    if shared.opts.data.get("control_net_skip_img2img_processing") and hasattr(p, "init_images"):
        swap_img2img_pipeline(p)

def swap_img2img_pipeline(p: processing.StableDiffusionProcessingImg2Img):
    p.__class__ = processing.StableDiffusionProcessingTxt2Img
    dummy = processing.StableDiffusionProcessingTxt2Img()
    for k,v in dummy.__dict__.items():
        if hasattr(p, k):
            continue
        setattr(p, k, v)

