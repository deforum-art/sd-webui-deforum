
# Deforum Stable Diffusion — official extension script for AUTOMATIC1111's webui

For now, video-input, 2D, pseudo-2D and 3D animation modes are available. Interpolation and render image batch temporary excluded for simplicity

<p align="left">
    <a href="https://github.com/deforum-art/deforum-for-automatic1111-webui/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://github.com/deforum-art/deforum-for-automatic1111-webui/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://github.com/deforum-art/deforum-for-automatic1111-webui/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://github.com/deforum-art/deforum-for-automatic1111-webui/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://colab.research.google.com/github/deforum/stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>  
    <a href="https://replicate.com/deforum/deforum_stable_diffusion"><img alt="Replicate" src="https://replicate.com/deforum/deforum_stable_diffusion/badge"></a>
</p>

## Before Starting

Read the README file at the original Deforum repo

https://github.com/deforum/stable-diffusion

## Getting Started

0. ~~Cover yourself in oil~~

1. Install [AUTOMATIC1111's webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/)

2. Download this repository and put the files deforum.py and the 'deforum' folder into the 'scripts' folder inside your WebUI installation directory

3. If you're on Windows and want to launch Deforum in 3D mode, you'll have to download the depths model manually. Download these files https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt and https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt and put them into the 'models/Deforum' folder of your webui installation. (if it doesn't exist, create it)

4.**Important: If you want to use 3D mode, launch the WebUI with the `--disable-safe-unpickle` option or else it won't let you to use the depth models!** [How to add it to the .bat file on Windows.](https://imgur.com/a/TJHglot) Open the webui, switch to 'img2img' tab and select 'Deforum v0.5-webui-beta' in the 'Custom scripts' dropdown menu

5. Enter the animation settings. Refer to [this general guide](https://docs.google.com/document/d/1pEobUknMFMkn8F5TMsv8qRzamXX_75BShMMXV8IFslI/edit) and [this guide to math keyframing functions in Deforum](https://docs.google.com/document/d/1pfW1PwbDIuW0cv-dnuyYj1UzPqe23BlSLTJsqazffXM/edit?usp=sharing). However, **in this version prompt weights less than zero don't just like in original Deforum!** Split the positive and the negative prompt in the json section using --neg argument like this "apple:\`where(cos(t)>=0, cos(t), 0)\`, snow --neg strawberry:\`where(cos(t)<0, -cos(t), 0)\`"

6. To view animation frames as they're being made, without waiting for the completion of an animation, go to the 'Settings' tab and set the value of this toolbar **above zero**, then click 'Apply settings' at the top of the page and return to the 'Deforum' tab. Warning: it may slow down the generation process.

![adsdasunknown](https://user-images.githubusercontent.com/14872007/196064311-1b79866a-e55b-438a-84a7-004ff30829ad.png)


7. Run the script and see if you got it working or even got something. **In 3D mode a large delay is expected at first** as the script loads the depth models. In the end, using the default settings the whole thing should consume only 5.7 GBs of VRAM at 3D mode peaks.

8. If it gives errors on missing modules, such as about missing 'numexpr', go to the original webui directory, open 'requirements_versions.txt' and append the missing packages names at the end of that file. Then restart the webui.

9. After the generation process is completed, click the button with the self-describing name to show the video or gif result right in the GUI!

![Screenshot 2022-10-18 at 01-06-24 Stable Diffusion](https://user-images.githubusercontent.com/14872007/196292734-245eb795-8ffc-4e4a-b8c5-1daa27dacf24.png)

10. Join our Discord where you can post generated stuff, ask questions and ~~infuriate the devs with 'this feature is in auto's build. When will it be in Deforum? Why can't I launch Deforum on my potato computer?'~~(not anymore, ha-ha) https://discord.gg/deforum. There's also the 'Issues' tab in the repo.

11. Profit!

## Showcase

Proof that it works good enough of AUTOMATIC1111's build with MATH keyframing and prompt-weighting enabled in 3D mode

![sw-min](https://user-images.githubusercontent.com/14872007/195954681-6b0f5a8d-e575-4ce3-9c10-e39ffbbca6ac.gif)


'Le Grand Interface' at work:


![Screenshot 2022-10-18 at 01-04-09 Stable Diffusion](https://user-images.githubusercontent.com/14872007/196292481-c77bcf3a-4712-44f5-97b2-d4b2480ca012.png)


Math evaluation:

![math-eval](https://user-images.githubusercontent.com/14872007/195957601-3c3fecab-5ef2-4a2f-9eba-3bb0c70bd4b8.png)



