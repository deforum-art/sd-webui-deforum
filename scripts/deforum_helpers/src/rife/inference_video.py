# Special thanks for https://github.com/XmYx for fixing this chaotic script
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

# TODO: FIX fp16 issues if this file fails and fp16 is enabled
def run_rife_new_video_infer(video=None,
        output=None,
        img=None,
        montage=False,
        model='RIFE46',
        fp16=False,
        UHD=False,
        scale=1.0,
        skip=False,
        fps=None,
        png=False,
        ext='mp4',
        exp=1,
        multi=2,
        deforum_models_path='models/Deforum',
        add_soundtrack=None):

    args = SimpleNamespace()
    args.video = video
    args.output = output
    args.img = img
    args.montage = montage
    args.modelDir = model
    args.fp16 = fp16
    args.UHD = UHD
    args.scale = scale
    args.skip = skip
    args.fps = fps
    args.png = png
    args.ext = ext
    args.ext = ext
    args.exp = exp
    args.multi = multi
    args.deforum_models_path = deforum_models_path
    args.add_soundtrack = add_soundtrack
   
    if args.exp != 1:
        args.multi = (2 ** args.exp)
    assert (not args.video is None or not args.img is None)
    if args.skip:
        print("skip flag is abandoned, please refer to issue #207.")
    if args.UHD and args.scale == 1.0:
        args.scale = 0.5
    assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
    if not args.img is None:
        args.png = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # TODO: Can handle this? currently it's always False and give errors if True but faster speeds on tensortcore equipped gpus?
        if (args.fp16):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    if args.modelDir == "RIFE31":
        try:
            from .RIFE31.RIFE_HDv3 import Model
            print("RIFE v3.1 has been successfully imported.")
        except ImportError as e:
            raise ValueError(f"RIFE v3.1 could not be found. Please contact deforum support. {e}")
        except Exception as e:
            raise ValueError(f"An error occured while trying to import RIFE v3.1: {e}")
    elif args.modelDir == "RIFE43":
        try:
            from .RIFE43.RIFE_HDv3 import Model
            print("RIFE v4.3 has been successfully imported.")
        except ImportError as e:
            raise ValueError(f"RIFE v4.3 could not be found. Please contact deforum support. {e}")
        except Exception as e:
            raise ValueError(f"An error occured while trying to import RIFE v4.3: {e}")
    elif args.modelDir == "RIFE46":
        try:
            from .RIFE46.RIFE_HDv3 import Model
            print("RIFE v4.6 has been successfully imported.")
        except ImportError as e:
            raise ValueError(f"RIFE v4.6 could not be found. Please contact deforum support. {e}")
        except Exception as e:
            raise ValueError(f"An error occured while trying to import RIFE v4.6: {e}")
    else:
        print("Got a request to frame-interpolate but no valid frame interpolation engine value provided. Doing... nothing.")
        return
    model = Model()
    if not hasattr(model, 'version'):
        model.version = 0
    model.load_model(args.modelDir, -1, deforum_models_path)
    model.eval()
    model.device()

    if not args.video is None:
        videoCapture = cv2.VideoCapture(args.video)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        videoCapture.release()
        if args.fps is None:
            fpsNotAssigned = True
            args.fps = fps * args.multi
        else:
            fpsNotAssigned = False
        videogen = skvideo.io.vreader(args.video)
        lastframe = next(videogen)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_path_wo_ext, ext = os.path.splitext(args.video)
        print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
        if args.png == False and fpsNotAssigned == True and args.add_soundtrack != 'None':
            print("The audio will be merged after the interpolation process")
        #else:
        #    print("Will not merge audio because using png or fps flag!")
    else:
        videogen = []
        for f in os.listdir(args.img):
            if 'png' in f:
                videogen.append(f)
        tot_frame = len(videogen)
        videogen.sort(key=lambda x: int(x[:-4]))
        lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        videogen = videogen[1:]
    h, w, _ = lastframe.shape
    vid_out_name = None
    vid_out = None
    if args.png:
        if not os.path.exists('vid_out'):
            os.mkdir('vid_out')
    else:
        if args.output is not None:
            vid_out_name = args.output
        else:
            vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
        vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

    if args.montage:
        left = w // 4
        w = w // 2
    tmp = max(128, int(128 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)
    if args.montage:
        lastframe = lastframe[:, left: left + w]
    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    
    _thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer, vid_out))

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

        if args.montage:
            write_buffer.put(np.concatenate((lastframe, lastframe), 1))
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
        else:
            write_buffer.put(lastframe)
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame
        if break_flag:
            break

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
    else:
        write_buffer.put(lastframe)
    import time

    while (not write_buffer.empty()):
        time.sleep(0.1)
    pbar.close()
    if not vid_out is None:
        vid_out.release()
    
    # move audio to new video file if appropriate
    if args.add_soundtrack != 'None' and args.png == False and fpsNotAssigned == True and not args.video is None:
        try:
            print (f"Trying to transfer audio from source video to final interpolated video")
            transferAudio(args.video, vid_out_name)
        except:
            print("Audio transfer failed. Interpolated video will have no audio")
            targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
            os.rename(targetNoAudio, vid_out_name)
            
    print(f"Frame interpolation *DONE*. Interpolated video name/ path: {vid_out_name}")

def clear_write_buffer(user_args, write_buffer, vid_out):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])


def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            if not user_args.img is None:
                frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            read_buffer.put(frame)
    except:
        pass
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


def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(
            targetVideo) == 0:  # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0):  # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")