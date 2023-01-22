import os
import cv2
import av

def create_video(imgs_folder_path, fps, bitrate, full_audio_path = None):
    
    audio_on = False
    if not full_audio_path is None:
        audio_on = True
        
    file_extension = '.png'
    images = []
    for file in os.listdir(imgs_folder_path):
        if file.endswith(file_extension):
            image = cv2.imread(os.path.join(imgs_folder_path, file))
            images.append(image)

    # Needed?
    video_duration = len(images) / fps
    
    # Get the width and height of the first image
    height, width, channels = images[0].shape
    
    # Open audio file
    if audio_on:
        audio_input = av.open(full_audio_path)
        audio_input_stream = audio_input.streams.audio[0] # Do i need this?
        audio_input_timebase = audio_input.streams.audio[0].time_base # Do i need this?

    # Open output file
    output_path = os.path.join(imgs_folder_path, 'output_av.mp4')
    output = av.open(output_path, 'w')
    # Add video stream
    stream = output.add_stream('h264', fps, thread_type = 'AUTO')
    stream.bit_rate = bitrate
    
    # Set the width and height of the video to the width and height of the first image
    stream.width = width
    stream.height = height
    
    # Add audio stream
    if audio_on:
        audio_stream = output.add_stream("aac", 44100)    

    # Encode and mux video frames
    for i, img in enumerate(images):
        frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        packet = stream.encode(frame)
        output.mux(packet)

    
    # Encode and mux audio frames
    if audio_on:
        for audio_frame in audio_input.decode(audio_input_stream): # tests for mp3?
            audio_pts_in_video_timebase = audio_frame.pts * audio_input_timebase
            if audio_pts_in_video_timebase < (i + 1) / fps:
                packet = audio_stream.encode(audio_frame)
                output.mux(packet)
          
    # flush
    packet = stream.encode(None)
    output.mux(packet)

    output.close()
    
#create_video('D:/D-SD/autopt2NEW/stable-diffusion-webui/outputs/img2img-images/Deforum', 15, 8000000, "D:/D-SD/autopt2NEW/stable-diffusion-webui/outputs/img2img-images/Deforum/20221010021115audio.mp4")
create_video('D:/D-SD/autopt2NEW/stable-diffusion-webui/outputs/img2img-images/Deforum', 14, 8000000, None)

#create_video('D:/D-SD/autopt2NEW/stable-diffusion-webui/outputs/img2img-images/Deforum', 14, 8000000, "D:\Downloads-D\music.mp3")


