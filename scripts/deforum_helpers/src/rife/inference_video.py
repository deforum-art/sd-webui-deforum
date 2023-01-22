# Special thanks for https://github.com/XmYx for the initial reorganization of this script
import os
from types import SimpleNamespace
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from .model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

def run_rife_new_video_infer(video=None,
        output=None,
        model=None,
        fp16=False,
        UHD=False,
        scale=1.0,
        fps=None,
        png=True,
        ext='mp4',
        exp=1,
        multi=2,
        deforum_models_path=None,
        add_soundtrack=None,
        raw_output_imgs_path=None,
        img_batch_id=None):

    args = SimpleNamespace()
    args.video = video
    args.output = output
    args.modelDir = model
    args.fp16 = fp16
    args.UHD = UHD
    args.scale = scale
    args.fps = fps
    args.png = png
    args.ext = ext
    args.exp = exp
    args.multi = multi
    args.deforum_models_path = deforum_models_path
    args.add_soundtrack = add_soundtrack
    args.raw_output_imgs_path = raw_output_imgs_path
    args.img_batch_id = img_batch_id
   
    if args.exp != 1:
        args.multi = (2 ** args.exp)
    if args.UHD and args.scale == 1.0:
        args.scale = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # TODO: Can handle this? currently it's always False and give errors if True but faster speeds on tensortcore equipped gpus?
        if (args.fp16):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
    if args.modelDir is not None:
        try:
            from .rife_new_gen.RIFE_HDv3 import Model
            print(f"{args.modelDir} has been successfully imported.")
        except ImportError as e:
            raise ValueError(f"{args.modelDir} could not be found. Please contact deforum support. {e}")
        except Exception as e:
            raise ValueError(f"An error occured while trying to import {args.modelDir}: {e}")
    else:
        print("Got a request to frame-interpolate but no valid frame interpolation engine value provided. Doing... nothing.")
        return
   
    model = Model()
    if not hasattr(model, 'version'):
        model.version = 0
    model.load_model(args.modelDir, -1, deforum_models_path)
    model.eval()
    model.device()
    
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = args.multi
    else:
        fpsNotAssigned = False
    
    if args.add_soundtrack != 'None' and fpsNotAssigned:
        print("The audio will be transferred from source video to interpolated video *after* the interpolation process")
    if not fpsNotAssigned and args.add_soundtrack != 'None':
        print("Will not transfer audio because Slow-Mo mode is activated!")
        
    interpolated_path = os.path.join(args.raw_output_imgs_path, 'interpolated_frames')
    custom_interp_path = "{}_{}".format(interpolated_path, args.img_batch_id)

    videogen = []
    for f in os.listdir(args.raw_output_imgs_path):
        if ('png' in f or 'jpg' in f) and '-' not in f and f.startswith(args.img_batch_id):
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))
    img_path = os.path.join(args.raw_output_imgs_path, videogen[0])
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
    
    _thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
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
            for i in range(args.multi - 1):
                output.append(I0)
        else:
            output = make_inference(model, I0, I1, args.multi - 1, scale)
        
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
    
    # move audio to new video file if appropriate
    if args.add_soundtrack != 'None' and args.png == False and fpsNotAssigned == True and not args.video is None:
        try:
            print (f"Trying to transfer audio from source video to final interpolated video")
            #transferAudio(args.video, vid_out_name)
        except:
            print("Audio transfer failed. Interpolated video will have no audio")
            #targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
            #os.rename(targetNoAudio, vid_out_name)
    
    #print(f"Frame interpolation *DONE*. Interpolated video name/ path: {vid_out_name}")
    stitch_video(args.fps, custom_interp_path, 'testtt.mp4', None)
    
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

def build_read_buffer(user_args, read_buffer, videogen):
    for frame in videogen:
        if not user_args.raw_output_imgs_path is None:
            img_path = os.path.join(user_args.raw_output_imgs_path, frame)
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


import os
import numpy as np
import cv2
import av

def get_filename(i, path):
    s = str(i)
    while len(s) < 7:
        s = '0' + s
    #return path + '/' + s + '.png'
    return path + s + '.png'
  
def stitch_video(fps, img_folder_path, video_name, audio_path):
    print("DO SOME STUFF HERE")
    # print("RIFE successfully transferred audio from source video to interpolated video")