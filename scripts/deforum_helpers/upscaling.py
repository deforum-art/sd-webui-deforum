import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from modules.scripts_postprocessing import PostprocessedImage
from modules import devices
import shutil
from queue import Queue, Empty
import modules.scripts as scr
from .frame_interpolation import clean_folder_name
from .general_utils import duplicate_pngs_from_folder, checksum
# TODO: move some funcs to this file?
from .video_audio_utilities import get_quick_vid_info, vid2frames, ffmpeg_stitch_video, extract_number, media_file_has_audio
from basicsr.utils.download_util import load_file_from_url
from .rich import console
import time
import subprocess

def process_upscale_vid_upload_logic(file, selected_tab, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, vid_file_name, keep_imgs, f_location, f_crf, f_preset):
    print("Got a request to *upscale* an existing video.")

    in_vid_fps, _, _ = get_quick_vid_info(file.name)
    folder_name = clean_folder_name(Path(vid_file_name).stem)
    outdir_no_tmp = os.path.join(os.getcwd(), 'outputs', 'frame-upscaling', folder_name)
    i = 1
    while os.path.exists(outdir_no_tmp):
        outdir_no_tmp = os.path.join(os.getcwd(), 'outputs', 'frame-upscaling', folder_name + '_' + str(i))
        i += 1

    outdir = os.path.join(outdir_no_tmp, 'tmp_input_frames')
    os.makedirs(outdir, exist_ok=True)
    
    vid2frames(video_path=file.name, video_in_frame_path=outdir, overwrite=True, extract_from_frame=0, extract_to_frame=-1, numeric_files_output=True, out_img_format='png')
    
    process_video_upscaling(selected_tab, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, orig_vid_fps=in_vid_fps, real_audio_track=file.name, raw_output_imgs_path=outdir, img_batch_id=None, ffmpeg_location=f_location, ffmpeg_crf=f_crf, ffmpeg_preset=f_preset, keep_upscale_imgs=keep_imgs, orig_vid_name=folder_name)

def process_video_upscaling(resize_mode, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, orig_vid_fps, real_audio_track, raw_output_imgs_path, img_batch_id, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, keep_upscale_imgs, orig_vid_name):
    devices.torch_gc()

    print("Upscaling progress (it's OK if it finishes before 100%):")

    upscaled_path = os.path.join(raw_output_imgs_path, 'upscaled_frames')
    if orig_vid_name is not None: # upscaling a video (deforum or unrelated)
        custom_upscale_path = "{}_{}".format(upscaled_path, orig_vid_name)
    else: # upscaling after a deforum run:
        custom_upscale_path = "{}_{}".format(upscaled_path, img_batch_id)
    
    temp_convert_raw_png_path = os.path.join(raw_output_imgs_path, "tmp_upscale_folder")
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

    # Upscaling is a slow and demanding operation, so we don't need as much parallelization here
    for i in tqdm(range(len(videogen)), desc="Upscaling"):
        lastframe = videogen[i]
        img_path = os.path.join(temp_convert_raw_png_path, lastframe)
        image = process_frame(resize_mode, Image.open(img_path).convert("RGB"), upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility)
        filename = '{}/{:0>7d}.png'.format(custom_upscale_path, i)
        image.save(filename)

    shutil.rmtree(temp_convert_raw_png_path)
    # stitch video from upscaled frames, and add audio if needed
    try:
        print (f"*Passing upsc frames to ffmpeg...*")
        vid_out_path = stitch_video(img_batch_id, orig_vid_fps, custom_upscale_path, real_audio_track, ffmpeg_location, resize_mode, upscaling_resize, upscaling_resize_w, upscaling_resize_h, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, ffmpeg_crf, ffmpeg_preset, keep_upscale_imgs, orig_vid_name)
        # remove folder with raw (non-upscaled) vid input frames in case of input VID and not PNGs
        if orig_vid_name is not None:
            shutil.rmtree(raw_output_imgs_path)
    except Exception as e:
        print(f'Video stitching gone wrong. *Upscaled frames were saved to HD as backup!*. Actual error: {e}')

    devices.torch_gc()

def process_frame(resize_mode, image, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility):
    pp = PostprocessedImage(image)
    postproc = scr.scripts_postproc
    upscaler_script = next(s for s in postproc.scripts if s.name == "Upscale")
    upscaler_script.process(pp, resize_mode, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility)
    return pp.image

def stitch_video(img_batch_id, fps, img_folder_path, audio_path, ffmpeg_location, resize_mode, upscaling_resize, upscaling_resize_w, upscaling_resize_h, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, f_crf, f_preset, keep_imgs, orig_vid_name):        
    parent_folder = os.path.dirname(img_folder_path)
    grandparent_folder = os.path.dirname(parent_folder)
    if orig_vid_name is not None:
        mp4_path = os.path.join(grandparent_folder, str(orig_vid_name) +'_upscaled_' + (('by_' + str(upscaling_resize).replace('.', '-')) if resize_mode == 0 else f"to_{upscaling_resize_w}_{upscaling_resize_h}")) + f"_with_{extras_upscaler_1}" + (f"_then_{extras_upscaler_2}" if extras_upscaler_2_visibility > 0 else "")
    else:
        mp4_path = os.path.join(parent_folder, str(img_batch_id) +'_upscaled_' + (('by_' + str(upscaling_resize).replace('.', '-')) if resize_mode == 0 else f"to_{upscaling_resize_w}_{upscaling_resize_h}")) + f"_with_{extras_upscaler_1}_then_{extras_upscaler_2}"
    
    mp4_path = mp4_path + '.mp4'

    t = os.path.join(img_folder_path, "%07d.png")
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

# NCNN Upscale section START
def process_ncnn_upscale_vid_upload_logic(vid_path, in_vid_fps, in_vid_res, out_vid_res, models_path, upscale_model, upscale_factor, keep_imgs, f_location, f_crf, f_preset, current_user_os):
    print(f"Got a request to *upscale* a video using {upscale_model} at {upscale_factor}")

    folder_name = clean_folder_name(Path(vid_path.orig_name).stem)
    outdir_no_tmp = os.path.join(os.getcwd(), 'outputs', 'frame-upscaling', folder_name)
    i = 1
    while os.path.exists(outdir_no_tmp):
        outdir_no_tmp = os.path.join(os.getcwd(), 'outputs', 'frame-upscaling', folder_name + '_' + str(i))
        i += 1

    outdir = os.path.join(outdir_no_tmp, 'tmp_input_frames')
    os.makedirs(outdir, exist_ok=True)
    
    vid2frames(video_path=vid_path.name, video_in_frame_path=outdir, overwrite=True, extract_from_frame=0, extract_to_frame=-1, numeric_files_output=True, out_img_format='png')
    
    process_ncnn_video_upscaling(vid_path, outdir, in_vid_fps, in_vid_res, out_vid_res, models_path, upscale_model, upscale_factor, keep_imgs, f_location, f_crf, f_preset, current_user_os)
    
def process_ncnn_video_upscaling(vid_path, outdir, in_vid_fps, in_vid_res, out_vid_res, models_path, upscale_model, upscale_factor, keep_imgs, f_location, f_crf, f_preset, current_user_os):
    # get clean number from 'x2, x3' etc
    clean_num_r_up_factor = extract_number(upscale_factor)
    # set paths
    realesrgan_ncnn_location = os.path.join(models_path, 'realesrgan_ncnn', 'realesrgan-ncnn-vulkan' + ('.exe' if current_user_os == 'Windows' else ''))
    upscaled_folder_path = os.path.join(os.path.dirname(outdir), 'Upscaled_frames')
    # create folder for upscaled imgs to live in. this folder will stay alive if keep_imgs=True, otherwise get deleted at the end
    os.makedirs(upscaled_folder_path, exist_ok=True)
    out_upscaled_mp4_path = os.path.join(os.path.dirname(outdir), f"{vid_path.orig_name}_Upscaled_{upscale_factor}.mp4")
    # download upscaling model if needed
    check_and_download_realesrgan_ncnn(models_path, current_user_os)
    # set cmd command
    cmd = [realesrgan_ncnn_location, '-i', outdir, '-o', upscaled_folder_path, '-s', str(clean_num_r_up_factor), '-n', upscale_model]
    # msg to print - need it to hide that text later on (!)
    msg_to_print = f"Upscaling raw PNGs using {upscale_model} at {upscale_factor}..."
    # blink the msg in the cli until action is done
    console.print(msg_to_print, style="blink yellow", end="") 
    start_time = time.time()
    # make call to ncnn upscaling executble
    process = subprocess.run(cmd, capture_output=True, check=True, text=True)
    print("\r" + " " * len(msg_to_print), end="", flush=True)
    print(f"\r{msg_to_print}", flush=True)
    print(f"\rUpscaling \033[0;32mdone\033[0m in {time.time() - start_time:.2f} seconds!", flush=True)
    # set custom path for ffmpeg func below
    upscaled_imgs_path_for_ffmpeg = os.path.join(upscaled_folder_path, "%05d.png")
    add_soundtrack = 'None'
    # don't pass add_soundtrack to ffmpeg if orig video doesn't contain any audio, so we won't get a message saying audio couldn't be added :)
    if media_file_has_audio(vid_path.name, f_location):
        add_soundtrack = 'File'
    # stitch video from upscaled pngs 
    ffmpeg_stitch_video(ffmpeg_location=f_location, fps=in_vid_fps, outmp4_path=out_upscaled_mp4_path, stitch_from_frame=0, stitch_to_frame=-1, imgs_path=upscaled_imgs_path_for_ffmpeg, add_soundtrack=add_soundtrack, audio_path=vid_path.name, crf=f_crf, preset=f_preset)
    # delete the raw video pngs
    shutil.rmtree(outdir)
    # delete upscaled imgs if user requested
    if not keep_imgs:
        shutil.rmtree(upscaled_folder_path)
        
def check_and_download_realesrgan_ncnn(models_folder, current_user_os):
    import zipfile
    if current_user_os == 'Windows':
        zip_file_name = 'realesrgan-ncnn-windows.zip'
        executble_name = 'realesrgan-ncnn-vulkan.exe'
        zip_checksum_value = '1d073f520a4a3f6438a500fea88407964da6d4a87489719bedfa7445b76c019fdd95a5c39576ca190d7ac22c906b33d5250a6f48cb7eda2b6af3e86ec5f09dfc'
        download_url = 'https://github.com/hithereai/Real-ESRGAN/releases/download/real-esrgan-ncnn-windows/realesrgan-ncnn-windows.zip'
    elif current_user_os == 'Linux':
        zip_file_name = 'realesrgan-ncnn-linux.zip'
        executble_name = 'realesrgan-ncnn-vulkan'
        zip_checksum_value = 'df44c4e9a1ff66331079795f018a67fbad8ce37c4472929a56b5a38440cf96982d6e164a086b438c3d26d269025290dd6498bd50846bda8691521ecf8f0fafdf'
        download_url = 'https://github.com/hithereai/Real-ESRGAN/releases/download/real-esrgan-ncnn-linux/realesrgan-ncnn-linux.zip'
    elif current_user_os == 'Mac':
        zip_file_name = 'realesrgan-ncnn-mac.zip'
        executble_name = 'realesrgan-ncnn-vulkan'
        zip_checksum_value = '65f09472025b55b18cf6ba64149ede8cded90c20e18d35a9edb1ab60715b383a6ffbf1be90d973fc2075cf99d4cc1411fbdc459411af5c904f544b8656111469'
        download_url = 'https://github.com/hithereai/Real-ESRGAN/releases/download/real-esrgan-ncnn-mac/realesrgan-ncnn-mac.zip'
    else: # who are you then?
        raise Exception(f"No support for OS type: {current_user_os}")

    # set paths
    realesrgan_ncnn_folder = os.path.join(models_folder, 'realesrgan_ncnn')
    realesrgan_exec_path = os.path.join(realesrgan_ncnn_folder, executble_name)
    realesrgan_zip_path = os.path.join(realesrgan_ncnn_folder, zip_file_name)
    # return if exec file already exist
    if os.path.exists(realesrgan_exec_path):
        return
    try:
        os.makedirs(realesrgan_ncnn_folder, exist_ok=True)
        # download exec and model files from url
        load_file_from_url(download_url, realesrgan_ncnn_folder)
        # check downloaded zip's hash
        with open(realesrgan_zip_path, 'rb') as f:
            file_hash = checksum(realesrgan_zip_path)
        # wrong hash, file is probably broken/ download interrupted 
        if file_hash != zip_checksum_value:
            raise Exception(f"Error while downloading {realesrgan_zip_path}. Please download from: {download_url}, and extract its contents into: {models_folder}/realesrgan_ncnn")
        # hash ok, extract zip contents into our folder
        with zipfile.ZipFile(realesrgan_zip_path, 'r') as zip_ref:
            zip_ref.extractall(realesrgan_ncnn_folder)
        # delete the zip file
        os.remove(realesrgan_zip_path)
        # chmod 755 the exec if we're in a linux machine, otherwise we'd get permission errors
        if current_user_os in ('Linux', 'Mac'):
            os.chmod(realesrgan_exec_path, 0o755)
            # enable running the exec for mac users
            if current_user_os == 'Mac':
                os.system(f'xattr -d com.apple.quarantine "{realesrgan_exec_path}"')

    except Exception as e:
        raise Exception(f"Error while downloading {realesrgan_zip_path}. Please download from: {download_url}, and extract its contents into: {models_folder}/realesrgan_ncnn")

def make_upscale_v2(upscale_factor, upscale_model, keep_imgs, imgs_raw_path, imgs_batch_id, deforum_models_path, current_user_os, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, fps, stitch_from_frame, stitch_to_frame, audio_path, add_soundtrack):
    # get clean number from 'x2, x3' etc
    clean_num_r_up_factor = extract_number(upscale_factor)
    # set paths
    realesrgan_ncnn_location = os.path.join(deforum_models_path, 'realesrgan_ncnn', 'realesrgan-ncnn-vulkan' + ('.exe' if current_user_os == 'Windows' else ''))
    upscaled_folder_path = os.path.join(imgs_raw_path, f"{imgs_batch_id}_upscaled")
    temp_folder_to_keep_raw_ims = os.path.join(upscaled_folder_path, 'temp_raw_imgs_to_upscale')
    out_upscaled_mp4_path = os.path.join(imgs_raw_path, f"{imgs_batch_id}_Upscaled_{upscale_factor}.mp4")
    # download upscaling model if needed
    check_and_download_realesrgan_ncnn(deforum_models_path, current_user_os)
    # make a folder with only the imgs we need to duplicate so we can call the ncnn with the folder syntax (quicker!)
    duplicate_pngs_from_folder(from_folder=imgs_raw_path, to_folder=temp_folder_to_keep_raw_ims, img_batch_id=imgs_batch_id, orig_vid_name='Dummy')
    # set dynamic cmd command
    cmd = [realesrgan_ncnn_location, '-i', temp_folder_to_keep_raw_ims, '-o', upscaled_folder_path, '-s', str(clean_num_r_up_factor), '-n', upscale_model]
    # msg to print - need it to hide that text later on (!)
    msg_to_print = f"Upscaling raw output PNGs using {upscale_model} at {upscale_factor}..."
    # blink the msg in the cli until action is done
    console.print(msg_to_print, style="blink yellow", end="") 
    start_time = time.time()
    # make call to ncnn upscaling executble
    process = subprocess.run(cmd, capture_output=True, check=True, text=True, cwd=(os.path.join(deforum_models_path, 'realesrgan_ncnn') if current_user_os == 'Mac' else None))
    print("\r" + " " * len(msg_to_print), end="", flush=True)
    print(f"\r{msg_to_print}", flush=True)
    print(f"\rUpscaling \033[0;32mdone\033[0m in {time.time() - start_time:.2f} seconds!", flush=True)
    # set custom path for ffmpeg func below
    upscaled_imgs_path_for_ffmpeg = os.path.join(upscaled_folder_path, f"{imgs_batch_id}_%05d.png")
    # stitch video from upscaled pngs 
    ffmpeg_stitch_video(ffmpeg_location=ffmpeg_location, fps=fps, outmp4_path=out_upscaled_mp4_path, stitch_from_frame=stitch_from_frame, stitch_to_frame=stitch_to_frame, imgs_path=upscaled_imgs_path_for_ffmpeg, add_soundtrack=add_soundtrack, audio_path=audio_path, crf=ffmpeg_crf, preset=ffmpeg_preset)

    # delete the duplicated raw imgs
    shutil.rmtree(temp_folder_to_keep_raw_ims)

    if not keep_imgs:
        shutil.rmtree(upscaled_folder_path)
# NCNN Upscale section END