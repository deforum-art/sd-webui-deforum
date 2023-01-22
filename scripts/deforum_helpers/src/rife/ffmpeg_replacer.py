import os
import cv2
import av
from pydub import AudioSegment

def create_video(imgs_folder_path, fps, bitrate, full_audio_path = None):

    audio_on = False
    if not full_audio_path is None:
        audio_on = True
    
    # only process PNGs and JPGs
    allowed_file_extensions = ['png', 'jpg']
    images = []
    for file in os.listdir(imgs_folder_path):
        if any(file.endswith(ext) for ext in allowed_file_extensions): # add all png/ jpg files from folder
            image = cv2.imread(os.path.join(imgs_folder_path, file))
            images.append(image)

    video_duration = len(images) / fps # so we can print it at the end
    
    # Get the width and height of the first image
    height, width, channels = images[0].shape
    
    # Open audio file
    if audio_on:
        # convert input audio file to mp4 format to be compatible with output mp4 video
        pydub_audio = AudioSegment.from_file(full_audio_path)
        pydub_audio.export(os.path.join(imgs_folder_path,"temp_audio.mp4"), format="mp4")
        # open converted mp4 *audio* file
        audio_input = av.open(os.path.join(imgs_folder_path,"temp_audio.mp4")) # TODO: delete temp when done?
        audio_input_stream = audio_input.streams.audio[0] 
        audio_input_timebase = audio_input.streams.audio[0].time_base

    # open/ prepare output video file
    output_path = os.path.join(imgs_folder_path, 'output_av.mp4')
    output = av.open(output_path, 'w')
    # Add video stream
    video_stream = output.add_stream('h264', fps, thread_type = 'AUTO')
    video_stream.bit_rate = bitrate
    
    # Set the width and height of the video to the width and height of the first image
    video_stream.width = width
    video_stream.height = height
    
    # Add audio stream
    if audio_on:
        audio_stream = output.add_stream("aac", 44100)    
        # audio_stream = output.add_stream("mp3", 44100)

    # Encode and mux video frames
    for i, img in enumerate(images):
        frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        packet = video_stream.encode(frame)
        output.mux(packet)

    # Encode and mux audio frames if needed
    if audio_on:
        for audio_frame in audio_input.decode(audio_input_stream): # tests for mp3?
            audio_pts_in_video_timebase = audio_frame.pts * audio_input_timebase
            if audio_pts_in_video_timebase < (i + 1) / fps:
                packet = audio_stream.encode(audio_frame)
                output.mux(packet)
          
    # flush
    packet = video_stream.encode(None)
    output.mux(packet)

    output.close()
    
    
# Usage examples:
    
#create_video('D:/D-SD/autopt2NEW/stable-diffusion-webui/outputs/img2img-images/Deforum', 10, 8000000, "D:/D-SD/autopt2NEW/stable-diffusion-webui/outputs/img2img-images/Deforum/20221010021115audio.mp4")
create_video('D:/D-SD/autopt2NEW/stable-diffusion-webui/outputs/img2img-images/Deforum', 12, 8000000, "C:/Users/jklgj/AppData/Local/Packages/TelegramFZ-LLC.Unigram_1vfw5zm9jmzqy/LocalState/0/music/TenWallsCut.mp3")

# create_video('D:/D-SD/autopt2NEW/stable-diffusion-webui/outputs/img2img-images/Deforum', 10, 8000000, "D:/D-SD/OLD/outputs/zoom_maker_NEW4/outputwithaudio.mp4")


