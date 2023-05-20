# TODO: deduplicate upscaling/interp/vid2depth code

import os, gc
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps, ImageChops
from modules.shared import cmd_opts, device as sh_device
from modules import devices
import shutil
from .frame_interpolation import clean_folder_name
from rife.inference_video import duplicate_pngs_from_folder
from .video_audio_utilities import get_quick_vid_info, vid2frames, ffmpeg_stitch_video

def process_depth_vid_upload_logic(file, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth, vid_file_name, keep_imgs, f_location, f_crf, f_preset, f_models_path):
    print("got a request to *vid2depth* an existing video.")

    in_vid_fps, _, _ = get_quick_vid_info(file.name)
    folder_name = clean_folder_name(Path(vid_file_name).stem)
    outdir_no_tmp = os.path.join(os.getcwd(), 'outputs', 'frame-depth', folder_name)
    i = 1
    while os.path.exists(outdir_no_tmp):
        outdir_no_tmp = os.path.join(os.getcwd(), 'outputs', 'frame-depth', folder_name + '_' + str(i))
        i += 1

    outdir = os.path.join(outdir_no_tmp, 'tmp_input_frames')
    os.makedirs(outdir, exist_ok=True)
    
    vid2frames(video_path=file.name, video_in_frame_path=outdir, overwrite=True, extract_from_frame=0, extract_to_frame=-1, numeric_files_output=True, out_img_format='png')
    
    process_video_depth(mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth, orig_vid_fps=in_vid_fps, real_audio_track=file.name, raw_output_imgs_path=outdir, img_batch_id=None, ffmpeg_location=f_location, ffmpeg_crf=f_crf, ffmpeg_preset=f_preset, f_models_path=f_models_path, keep_depth_imgs=keep_imgs, orig_vid_name=folder_name)

def process_video_depth(mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth, orig_vid_fps, real_audio_track, raw_output_imgs_path, img_batch_id, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, f_models_path, keep_depth_imgs, orig_vid_name):
    devices.torch_gc()

    print("Vid2depth progress (it's OK if it finishes before 100%):")

    upscaled_path = os.path.join(raw_output_imgs_path, 'depth_frames')
    if orig_vid_name is not None: # upscaling a video (deforum or unrelated)
        custom_upscale_path = "{}_{}".format(upscaled_path, orig_vid_name)
    else: # upscaling after a deforum run:
        custom_upscale_path = "{}_{}".format(upscaled_path, img_batch_id)
    
    temp_convert_raw_png_path = os.path.join(raw_output_imgs_path, "tmp_depth_folder")
    duplicate_pngs_from_folder(raw_output_imgs_path, temp_convert_raw_png_path, img_batch_id, orig_vid_name)

    videogen = []
    for f in os.listdir(temp_convert_raw_png_path):
        # double check for old _depth_ files, not really needed probably but keeping it for now
        if '_depth_' not in f:
            videogen.append(f)
            
    videogen.sort(key= lambda x:int(x.split('.')[0]))
    vid_out = None

    if not os.path.exists(custom_upscale_path):
        os.mkdir(custom_upscale_path)
    
    # Loading the chosen model
    if 'Mixed' in mode:
        model = (load_depth_model(f_models_path, midas_weight_vid2depth), load_anime_model())
    elif 'Depth' in mode:
        model = load_depth_model(f_models_path, midas_weight_vid2depth)
    elif 'Anime' in mode:
        model = load_anime_model()
    else:
        model = None

    # Upscaling is a slow and demanding operation, so we don't need as much parallelization here
    for i in tqdm(range(len(videogen)), desc="Vid2depth"):
        lastframe = videogen[i]
        img_path = os.path.join(temp_convert_raw_png_path, lastframe)
        image = process_frame(model, Image.open(img_path).convert("RGB"), mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth)
        filename = '{}/{:0>9d}.png'.format(custom_upscale_path, i)
        image.save(filename)
    
    # Cleaning up and freeing the memory before stitching
    model = None
    gc.collect()
    devices.torch_gc()

    shutil.rmtree(temp_convert_raw_png_path)
    # stitch video from upscaled frames, and add audio if needed
    try:
        print (f"*Passing depth frames to ffmpeg...*")
        vid_out_path = stitch_video(img_batch_id, orig_vid_fps, custom_upscale_path, real_audio_track, ffmpeg_location, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth, ffmpeg_crf, ffmpeg_preset, keep_depth_imgs, orig_vid_name)
        # remove folder with raw (non-upscaled) vid input frames in case of input VID and not PNGs
        if orig_vid_name is not None:
            shutil.rmtree(raw_output_imgs_path)
    except Exception as e:
        print(f'Video stitching gone wrong. *Vid2depth frames were saved to HD as backup!*. Actual error: {e}')

    gc.collect()
    devices.torch_gc()

def process_frame(model, image, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth):
    # Get grayscale foreground map
    if 'None' in mode:
        depth = process_depth(image, 'None', thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur)
    elif not 'Mixed' in mode:
        depth = process_frame_depth(model, np.array(image), midas_weight_vid2depth) if 'Depth' in mode else process_frame_anime(model, np.array(image))
        depth = process_depth(depth, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur)
    else:
        if thresholding == 'None':
            raise "Mixed mode doesn't work with no thresholding!"
        depth_depth = process_frame_depth(model[0], np.array(image), midas_weight_vid2depth)
        depth_depth = process_depth(depth_depth, 'Depth', thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur)
        anime_depth = process_frame_anime(model[1], np.array(image))
        anime_depth = process_depth(anime_depth, 'Anime', 'Simple', 32, 255, adapt_block_size, adapt_c, invert, end_blur)
        depth = ImageChops.logical_or(depth_depth.convert('1'), anime_depth.convert('1'))

    return depth

def process_depth(depth, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur):
    depth = depth.convert('L')
    # Depth mode need inverting whereas Anime mode doesn't
    # (invert and 'Depth' in mode) or (not invert and not 'Depth' in mode)
    if (invert and 'None' in mode) or (invert is ('Depth' in mode)):
        depth = ImageOps.invert(depth)
    
    depth = np.array(depth)
    
    # Apply thresholding
    if thresholding == 'Simple':
        _, depth = cv2.threshold(depth, threshold_value, threshold_value_max, cv2.THRESH_BINARY)
    elif thresholding == 'Simple (Auto-value)':
        _, depth = cv2.threshold(depth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresholding == 'Adaptive (Mean)':
        depth = cv2.adaptiveThreshold(depth, threshold_value_max, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, int(adapt_block_size), adapt_c)
    elif thresholding == 'Adaptive (Gaussian)':
        depth = cv2.adaptiveThreshold(depth, threshold_value_max, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, int(adapt_block_size), adapt_c)

    # Apply slight blur in the end to smoothen the edges after initial thresholding
    if end_blur > 0:
        depth = cv2.GaussianBlur(depth, (5, 5), end_blur)

    if thresholding == 'None' or end_blur == 0:
        # Return a graymap
        return Image.fromarray(depth).convert('L')
    else:
        # This commits thresholding again, but on the already processed image, so we don't need to set it up as much
        return Image.fromarray(cv2.threshold(depth, 127, 255, cv2.THRESH_BINARY)[1]).convert('L')
    
def stitch_video(img_batch_id, fps, img_folder_path, audio_path, ffmpeg_location, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth, f_crf, f_preset, keep_imgs, orig_vid_name):        
    parent_folder = os.path.dirname(img_folder_path)
    grandparent_folder = os.path.dirname(parent_folder)
    mode = str(mode).replace('\\', '_').replace(' ', '_').replace('(', '_').replace(')', '_')
    mp4_path = os.path.join(grandparent_folder, str(orig_vid_name if orig_vid_name is not None else img_batch_id) +'_depth_'+f"{thresholding}")
    
    mp4_path = mp4_path + '.mp4'

    t = os.path.join(img_folder_path, "%09d.png")
    add_soundtrack = 'None'
    if not audio_path is None:
        add_soundtrack = 'File'
        
    exception_raised = False
    try:
        ffmpeg_stitch_video(ffmpeg_location=ffmpeg_location, fps=fps, outmp4_path=mp4_path, stitch_from_frame=0, stitch_to_frame=1000000, imgs_path=t, add_soundtrack=add_soundtrack, audio_path=audio_path, crf=f_crf, preset=f_preset)
    except Exception as e:
        exception_raised = True
        print(f"An error occurred while stitching the video: {e}")

    if not exception_raised and not keep_imgs:
        shutil.rmtree(img_folder_path)

    if (keep_imgs and orig_vid_name is not None) or (orig_vid_name is not None and exception_raised is True):
        shutil.move(img_folder_path, grandparent_folder)

    return mp4_path

# Midas/Adabins Depth mode with the usual workflow
def load_depth_model(models_path, midas_weight_vid2depth):
    from .depth import DepthModel
    device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else sh_device)
    keep_in_vram = False # TODO: Future  - handle this too?
    print('Loading Depth Model')
    depth_model = DepthModel(models_path, device, not cmd_opts.no_half, keep_in_vram=keep_in_vram)
    return depth_model

# Anime Remove Background by skytnt and onnx model
# https://huggingface.co/spaces/skytnt/anime-remove-background/blob/main/app.py
def load_anime_model():
    # Installing its deps on demand
    print('Checking ARB dependencies')
    from launch import is_installed, run_pip
    libs = ["onnx", "onnxruntime-gpu", "huggingface_hub"]
    for lib in libs:
        if not is_installed(lib):
            run_pip(f"install {lib}", lib)
    
    try:
        import onnxruntime as rt
        import huggingface_hub
    except Exception as e:
        raise f"onnxruntime has not been installed correctly! Anime Remove Background mode is unable to function. The actual exception is: {e}. Note, that you'll need internet connection for the first run!"
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    model_path = huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
    return rt.InferenceSession(model_path, providers=providers)
    
def get_mask(rmbg_model, img, s=1024):
    img = (img / 255).astype(np.float32)
    h, w = h0, w0 = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = rmbg_model.run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))
    # TODO: pass in batches
    mask = (mask * 255).astype(np.uint8)
    return mask

def process_frame_depth(depth_model, image, midas_weight):
    opencv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    depth = depth_model.predict(opencv_image, midas_weight, not cmd_opts.no_half)
    return depth_model.to_image(depth)

def process_frame_anime(model, image):
    return Image.fromarray(get_mask(model, image), 'L')
