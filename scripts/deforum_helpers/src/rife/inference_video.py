# thanks to https://github.com/n00mkrad for the inspiration and a bit of code. Also thanks for https://github.com/XmYx for the initial reorganization of this script
import os
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
from .model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

def run_rife_new_video_infer(
        output=None,
        model=None,
        fp16=False,
        UHD=False,
        scale=1.0,
        fps=None,
        png=True,
        deforum_models_path=None,
        raw_output_imgs_path=None,
        img_batch_id=None,
        ffmpeg_location='ffmpeg',
        audio_track=None,
        interp_x_amount=2,
        slow_mo_x_amount=-1,
        ffmpeg_crf=17,
        ffmpeg_preset='veryslow',
        keep_imgs=False):

    args = SimpleNamespace()
    args.output = output
    args.modelDir = model
    args.fp16 = fp16
    args.UHD = UHD
    args.scale = scale
    args.fps = fps
    args.png = png
    args.deforum_models_path = deforum_models_path
    args.raw_output_imgs_path = raw_output_imgs_path
    args.img_batch_id = img_batch_id
    args.ffmpeg_location = ffmpeg_location
    args.audio_track = audio_track
    args.interp_x_amount = interp_x_amount
    args.slow_mo_x_amount = slow_mo_x_amount
    args.ffmpeg_crf = ffmpeg_crf
    args.ffmpeg_preset = ffmpeg_preset
    args.keep_imgs = keep_imgs
   
    if args.UHD and args.scale == 1.0:
        args.scale = 0.5

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
    
    if not args.audio_track is None and args.slow_mo_x_amount >= 2:
        print("Got a request to add audio. The audio will be added to the interpolated video as it is!")
    
    # TODO: add options to not move audio if slow mode is enabled + add option to slow-down the audio 

    interpolated_path = os.path.join(args.raw_output_imgs_path, 'interpolated_frames')
    custom_interp_path = "{}_{}".format(interpolated_path, args.img_batch_id)
    temp_convert_raw_png_path = os.path.join(args.raw_output_imgs_path, "tmp_rife_folder")
    
    duplicate_pngs_from_folder(args.raw_output_imgs_path, temp_convert_raw_png_path, args.img_batch_id)
    
    videogen = []
    for f in os.listdir(temp_convert_raw_png_path):
        videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))
    img_path = os.path.join(temp_convert_raw_png_path, videogen[0])
    lastframe = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]    
    h, w, _ = lastframe.shape
    vid_out = None

    if args.png:
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
    import time

    while (not write_buffer.empty()):
        time.sleep(0.1)
    pbar.close()
    shutil.rmtree(temp_convert_raw_png_path)
    
    # stitch video from interpolated frames, and add audio if needed
    try:
        print (f"Trying to stitch video from interpolated PNG frames...")
        stitch_video(args.img_batch_id, args.fps, custom_interp_path, args.audio_track, args.ffmpeg_location, args.interp_x_amount, args.slow_mo_x_amount, args.ffmpeg_crf, args.ffmpeg_preset, args.keep_imgs)
        print("Interpolated video created!")
    except Exception as e:
        print(f'Video stitching gone wrong. Error: {e}')

def duplicate_pngs_from_folder(from_folder, to_folder, img_batch_id):
    temp_convert_raw_png_path = os.path.join(from_folder, to_folder) #"tmp_rife_folder")
    if not os.path.exists(temp_convert_raw_png_path):
                os.makedirs(temp_convert_raw_png_path)
                
    for f in os.listdir(from_folder):
        if ('png' in f or 'jpg' in f) and '-' not in f and f.startswith(img_batch_id):
            original_img_path = os.path.join(from_folder, f)
            image = cv2.imread(original_img_path)
            new_path = os.path.join(temp_convert_raw_png_path, f)
            cv2.imwrite(new_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
def clear_write_buffer(user_args, write_buffer, custom_interp_path):
    cnt = 0

    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            filename = '{}/{:0>7d}.png'.format(custom_interp_path, cnt)

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
    #global model
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

def get_filename(i, path):
    s = str(i)
    while len(s) < 7:
        s = '0' + s
    #return path + '/' + s + '.png'
    return path + s + '.png'

def stitch_video(img_batch_id, fps, img_folder_path, audio_path, ffmpeg_location, interp_x_amount, slow_mo_x_amount, f_crf, f_preset, keep_imgs):
    parent_folder = os.path.dirname(img_folder_path)
    mp4_path = os.path.join(parent_folder, str(img_batch_id) +'_RIFE_' + 'x' + str(interp_x_amount))
    if slow_mo_x_amount != -1:
        mp4_path = mp4_path + '_slomo_x' + str(slow_mo_x_amount)
    mp4_path = mp4_path + '.mp4'

    t = os.path.join(img_folder_path, "%07d.png")
    try:
        cmd = [
                ffmpeg_location,
                '-y',
                '-vcodec', 'png',
                '-r', str(int(fps)),
                '-start_number', str(0),
                '-i', t,
                '-frames:v', str(1000000),
                '-c:v', 'libx264',
                '-vf',
                f'fps={int(fps)}',
                '-pix_fmt', 'yuv420p',
                '-crf', str(f_crf),
                '-preset', f_preset,
                '-pattern_type', 'sequence',
                mp4_path
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr)
    except Exception as e:
        print(f'Error stitching interpolation video. Actual error: {e}')

    if not audio_path is None:
        try:
            cmd = [
                ffmpeg_location,
                '-i',
                mp4_path, 
                '-i',
                audio_path,
                '-map', '0:v',
                '-map', '1:a',
                '-c:v', 'copy',
                '-shortest',
                mp4_path+'.temp.mp4'
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(stderr)
                raise RuntimeError(stderr)
            os.replace(mp4_path+'.temp.mp4', mp4_path)
        except Exception as e:
            print(f'Error adding audio to interpolated video. Actual error: {e}')
    # delete temp folder with interpolated frames if requested
    if not keep_imgs:
        shutil.rmtree(img_folder_path)
