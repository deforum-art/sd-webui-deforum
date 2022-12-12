# Deforum Stable Diffusion looper
The idea is to allow the looping of deforum videos

This addon to the deforum tool is still in early beta - aka it works for me.

# Getting started

To enable the loop make sure to check `init_image` set your strength_schedule, and modify your `init_image`. This should get you started.

As for suggestions on values for a strength schedule, id suggest higher values for the keyframe frames.

The web ui changes I have implemented for this version are in the `init_image`. There is no visual change from the original deforum, however it now uses json that is parsed within [the generate script](scripts/deforum_helpers/generate.py#L160-L174). This will allow for the init image to be like the `animation_prompts` from the prompts tab, and should be a simple interface for injecting keyframes at specific keyframe intervals.

So the `init_image` should look something like the following 
```json
{
    "0":  "location of initial image",
    "50": "2nd image location",
    "100": "3rd image location",
    "150": "4th image location",
    "200": "location of initial image"
}
```
#### this has only been tested with web images, local images may work, not sure. Let me know if they do!

# Notes

### 1
the math is super early still, and has mostly been tested and optimized for 50 frame injection cycles. Ill work on making this more general later. For now expected frames are 0,50,100,... 

### 2
Also note currently the last frame injected (say 200) needs to be 20 frames before the end of the video to allow the previous prompts and images to go back to the original. So if you inject 5 images where the 1st and last one are the same you will need to run the animation for 220 runs

### 3
The strength schedule and amount of 1st image used are super important but up to taste how close you want these to be. I generally have been setting the strength schedule from betweeen .5 and .8 depending on how close to a frame change I am. For the initial image strength Ive been setting mine between .68 and .83 with great results depending on the size of the jump I want for the last frame. Ill have more on this later as I clean it up.

### 4
You need to set your seed to schedule and schedule the seed to move in a specific way. I will try to clean this up later but what I have found is its best (using the example of 5 insertions at 220 frames) to have a schedule that starts and ends on the same seed.

For instance I use the following schedule sometimes

`0:(5), 1:(-1), 219:(-1), 220:(5)`

I am not positive about this, however I know the first and last frame in the seed should be the same to get the most simular images

### 5
There are a few hidden variables that I will break out in later versions, like how fast the colors change from one image to the other and the default blend factor, also possibly the blend formula if I can get a clean way to do that.

# Math

The method used is somewhat simple at the moment, using a blend factor that is based on the number of tweening frames and a somewhat simple formula of 

`.35 - .25*math.cos((frame % tweeningFrames) / (tweeningFrames / 2))`

Currently tweeningFrames is hardcoded to 20 frames as to make the jump between images and color spaces a bit less jaring, but Ill break this out to the ui soon.

# Thank you!
Thank you for checking out this modification to deforum, this is a simple extension I have been writing and thought others would have fun with!

If you find issues, or want extensions to this let me know

# using the Deforum Stable Diffusion — official extension script for AUTOMATIC1111's webui
https://github.com/deforum-art/deforum-for-automatic1111-webui

# Having trouble or have suggestions? 
Feel free to contact me on [discord](https://discord.gg/ZUMxF6q3EZ)
