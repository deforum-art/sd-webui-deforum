
# Deforum Stable Diffusion — (WIP) official extension script for automatic1111's webui

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

0. ~~Cover yourself in oil~~ Wait for completion of this port

1. Install Automatic1111's webui

2. Download this repository and put the files deforum.py and the 'deforum' folder into the 'scripts' folder inside your WebUI installation directory

3. **Important: If you want to use 3D mode, launch the WebUI with the `--disable-safe-unpickle` option or else it won't let you to use the depth models!** .Open the webui, switch to 'img2img' tab and select 'Deforum v0.5-webui-beta' in the 'Custom scripts' dropdown menu

4. Enter the animation settings. Refer to [this general guide](https://docs.google.com/document/d/1pEobUknMFMkn8F5TMsv8qRzamXX_75BShMMXV8IFslI/edit) and [this guide to math keyframing functions in Deforum](https://docs.google.com/document/d/1pfW1PwbDIuW0cv-dnuyYj1UzPqe23BlSLTJsqazffXM/edit?usp=sharing). However, **in this version prompt weights less than zero don't just like in original Deforum!** Split the positive and the negative prompt in the json section using --neg argument like this "apple:\`where(cos(t)>=0, cos(t), 0)\`, snow --neg strawberry:\`where(cos(t)<0, -cos(t), 0)\`"

5. Run the script and see if you got it working or even got something. **In 3D mode a large delay is expected at first** as the script downloads the depth models.

6. If it gives errors on missing modules, run `pip install requirements.txt` with `requirements.txt` being from this repo.

7. Join our Discord where you can post generated stuff, ask questions and ~~infuriate the devs with 'this feature is in auto's build. When will it be in Deforum? Why can't I launch Deforum on my potato computer?'~~(not anymore, ha-ha) https://discord.gg/deforum. There's also the 'Issues' tab in the repo.

8. Profit!
