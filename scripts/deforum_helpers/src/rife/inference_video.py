# thanks to https://github.com/n00mkrad for the inspiration and a bit of code. Also thanks for https://github.com/XmYx for the initial reorganization of this script
import os, sys
from types import SimpleNamespace
import cv2
import torch
import argparse
import shutil
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
from queue import Queue, Empty
import subprocess
import time
from .model.pytorch_msssim import ssim_matlab

sys.path.append('../../')
from deforum_helpers.video_audio_utilities import ffmpeg_stitch_video
from deforum_helpers.general_utils import duplicate_pngs_from_folder

warnings.filterwarnings("ignore")

def run_rife_new_video_infer(
        output=None,
        model=None,
        fp16=False,
        UHD=False, # *Will be received as *True* if imgs/vid resolution is 2K or higher*
        scale=1.0,
        fps=None,
        deforum_models_path=None,
        raw_output_imgs_path=None,
        img_batch_id=None,
        ffmpeg_location=None,
        audio_track=None,
        interp_x_amount=2,
        slow_mo_enabled=False,
        slow_mo_x_amount=2,
        ffmpeg_crf=17,
        ffmpeg_preset='veryslow',
        keep_imgs=False,
        orig_vid_name = None):

    args = SimpleNamespace()
    args.output = output
    args.modelDir = model
    args.fp16 = fp16
    args.UHD = UHD
    args.scale = scale
    args.fps = fps
    args.deforum_models_path = deforum_models_path
    args.raw_output_imgs_path = raw_output_imgs_path
    args.img_batch_id = img_batch_id
    args.ffmpeg_location = ffmpeg_location
    args.audio_track = audio_track
    args.interp_x_amount = interp_x_amount
    args.slow_mo_enabled = slow_mo_enabled
    args.slow_mo_x_amount = slow_mo_x_amount
    args.ffmpeg_crf = ffmpeg_crf
    args.ffmpeg_preset = ffmpeg_preset
    args.keep_imgs = keep_imgs
    args.orig_vid_name = orig_vid_name

    if args.UHD and args.scale == 1.0:
        args.scale = 0.5
        
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # TODO: Can/ need to handle this? currently it's always False and give errors if True but faster speeds on tensortcore equipped gpus?
        if (args.fp16):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
    if args.modelDir is not None:
        try:
            from .rife_new_gen.RIFE_HDv3 import Model
        except ImportError as e:
            raise ValueError(f"{args.modelDir} could not be found. Please contact deforum support {e}")
        except Exception as e:
            raise ValueError(f"An error occured while trying to import {args.modelDir}: {e}")
    else:
        print("Got a request to frame-interpolate but no valid frame interpolation engine value provided. Doing... nothing")
        return
   
    model = Model()
    if not hasattr(model, 'version'):
        model.version = 0
    model.load_model(args.modelDir, -1, deforum_models_path)
    model.eval()
    model.device()
    
    print(f"{args.modelDir}.pkl model successfully loaded into memory")
    print("Interpolation progress (it's OK if it finishes before 100%):")
   
    interpolated_path = os.path.join(args.raw_output_imgs_path, 'interpolated_frames_rife')
    # set custom name depending on if we interpolate after a run, or interpolate a video (related/unrelated to deforum, we don't know) directly from within the RIFE tab
    if args.orig_vid_name is not None: # interpolating a video (deforum or unrelated)
        custom_interp_path = "{}_{}".format(interpolated_path, args.orig_vid_name)
    else: # interpolating after a deforum run:
        custom_interp_path = "{}_{}".format(interpolated_path, args.img_batch_id)

    # In this folder we temporarily keep the original frames (converted/ copy-pasted and img format depends on scenario)
    # the convertion case is done to avert a problem with 24 and 32 mixed outputs from the same animation run
    temp_convert_raw_png_path = os.path.join(args.raw_output_imgs_path, "tmp_rife_folder")
    
    duplicate_pngs_from_folder(args.raw_output_imgs_path, temp_convert_raw_png_path, args.img_batch_id, args.orig_vid_name)
    
    videogen = []
    for f in os.listdir(temp_convert_raw_png_path):
        # double check for old _depth_ files, not really needed probably but keeping it for now
        if '_depth_' not in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x.split('.')[0]))
    img_path = os.path.join(temp_convert_raw_png_path, videogen[0])
    lastframe = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]    
    h, w, _ = lastframe.shape
    vid_out = None

    if not os.path.exists(custom_interp_path):
        os.mkdir(custom_interp_path)

    tmp = max(128, int(128 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)

    write_buffer = Queue(maxsize=500)
    read_buffer  = Queue(maxsize=500)
    
    _thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen, temp_convert_raw_png_path))
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer, custom_interp_path))

    I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1, args.fp16, padding)
    temp = None  # save lastframe when processing static frame
    
    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1, args.fp16, padding)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996:
            frame = read_buffer.get()  # read a new frame
            if frame is None:
                break_flag = True
                frame = lastframe
            else:
                temp = frame
            I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = pad_image(I1, args.fp16, padding)
            I1 = model.inference(I0, I1, args.scale)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

        if ssim < 0.2:
            output = []
            for i in range(args.interp_x_amount - 1):
                output.append(I0)
        else:
            output = make_inference(model, I0, I1, args.interp_x_amount - 1, scale)
        
        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame
        if break_flag:
            break

    write_buffer.put(lastframe)

    while (not write_buffer.empty()):
        time.sleep(0.1)
    pbar.close()
    shutil.rmtree(temp_convert_raw_png_path)
    
    print(f"Interpolation \033[0;32mdone\033[0m in {time.time()-start_time:.2f} seconds!")
    # stitch video from interpolated frames, and add audio if needed
    try:
        print (f"*Passing interpolated frames to ffmpeg...*")
        vid_out_path = stitch_video(args.img_batch_id, args.fps, custom_interp_path, args.audio_track, args.ffmpeg_location, args.interp_x_amount, args.slow_mo_enabled, args.slow_mo_x_amount, args.ffmpeg_crf, args.ffmpeg_preset, args.keep_imgs, args.orig_vid_name)
        # remove folder with raw (non-interpolated) vid input frames in case of input VID and not PNGs
        if orig_vid_name is not None:
            shutil.rmtree(raw_output_imgs_path)
    except Exception as e:
        print(f'Video stitching gone wrong. *Interpolated frames were saved to HD as backup!*. Actual error: {e}')
    
def clear_write_buffer(user_args, write_buffer, custom_interp_path):
    cnt = 0

    while True:
        item = write_buffer.get()
        if item is None:
            break
        filename = '{}/{:0>9d}.png'.format(custom_interp_path, cnt)

        cv2.imwrite(filename, item[:, :, ::-1])

        cnt += 1

def build_read_buffer(user_args, read_buffer, videogen, temp_convert_raw_png_path):
    for frame in videogen:
        if not temp_convert_raw_png_path is None:
            img_path = os.path.join(temp_convert_raw_png_path, frame)
            frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        read_buffer.put(frame)
    read_buffer.put(None)

def make_inference(model, I0, I1, n, scale):
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i + 1) * 1. / (n + 1), scale))
        return res
    else:
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(model, I0, middle, n=n // 2, scale=scale)
        second_half = make_inference(model, middle, I1, n=n // 2, scale=scale)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

def pad_image(img, fp16, padding):
    if (fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

# TODO: move to fream_interpolation and add FILM to it!
def stitch_video(img_batch_id, fps, img_folder_path, audio_path, ffmpeg_location, interp_x_amount, slow_mo_enabled, slow_mo_x_amount, f_crf, f_preset, keep_imgs, orig_vid_name):        
    parent_folder = os.path.dirname(img_folder_path)
    grandparent_folder = os.path.dirname(parent_folder)
    if orig_vid_name is not None:
        mp4_path = os.path.join(grandparent_folder, str(orig_vid_name) +'_RIFE_' + 'x' + str(interp_x_amount))
    else:
        mp4_path = os.path.join(parent_folder, str(img_batch_id) +'_RIFE_' + 'x' + str(interp_x_amount))

    if slow_mo_enabled:
        mp4_path = mp4_path + '_slomo_x' + str(slow_mo_x_amount)
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