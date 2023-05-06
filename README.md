
# Deforum Stable Diffusion — official extension for AUTOMATIC1111's webui

<p align="left">
    <a href="https://github.com/deforum-art/sd-webui-deforum/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://github.com/deforum-art/sd-webui-deforum/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://github.com/deforum-art/sd-webui-deforum/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://github.com/deforum-art/sd-webui-deforum/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/deforum-art/deforum-for-automatic1111-webui"></a>
    </a>
</p>

## Need help? See our [FAQ](https://github.com/deforum-art/sd-webui-deforum/wiki/FAQ-&-Troubleshooting)

## Getting Started

1. Install [AUTOMATIC1111's webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/).

2. Now two ways: either clone the repo into the `extensions` directory via git commandline launched within in the `stable-diffusion-webui` folder

```sh
git clone https://github.com/deforum-art/sd-webui-deforum extensions/deforum
```

Or download this repository, locate the `extensions` folder within your WebUI installation, create a folder named `deforum` and put the contents of the downloaded directory inside of it. Then restart WebUI.

3. Open the webui, find the Deforum tab at the top of the page.

4. Enter the animation settings. Refer to [this general guide](https://docs.google.com/document/d/1pEobUknMFMkn8F5TMsv8qRzamXX_75BShMMXV8IFslI/edit) and [this guide to math keyframing functions in Deforum](https://docs.google.com/document/d/1pfW1PwbDIuW0cv-dnuyYj1UzPqe23BlSLTJsqazffXM/edit?usp=sharing). However, **in this version prompt weights less than zero don't just like in original Deforum!** Split the positive and the negative prompt in the json section using --neg argument like this "apple:\`where(cos(t)>=0, cos(t), 0)\`, snow --neg strawberry:\`where(cos(t)<0, -cos(t), 0)\`"

5. To view animation frames as they're being made, without waiting for the completion of an animation, go to the 'Settings' tab and set the value of this toolbar **above zero**. Warning: it may slow down the generation process.

![adsdasunknown](https://user-images.githubusercontent.com/14872007/196064311-1b79866a-e55b-438a-84a7-004ff30829ad.png)


6. Run the script and see if you got it working or even got something. **In 3D mode a large delay is expected at first** as the script loads the depth models. In the end, using the default settings the whole thing should consume 6.4 GBs of VRAM at 3D mode peaks and no more than 3.8 GB VRAM in 3D mode if you launch the webui with the '--lowvram' command line argument.

7. After the generation process is completed, click the button with the self-describing name to show the video or gif result right in the GUI!

8. Join our Discord where you can post generated stuff, ask questions and more: https://discord.gg/deforum. <br>
* There's also the 'Issues' tab in the repo, for well... reporting issues ;) 

9. Profit!

## Known issues

* This port is not fully backward-compatible with the notebook and the local version both due to the changes in how AUTOMATIC1111's webui handles Stable Diffusion models and the changes in this script to get it to work in the new environment. *Expect* that you may not get exactly the same result or that the thing may break down because of the older settings.

## Screenshots

Amazing raw Deforum animation by [Pxl.Pshr](https://www.instagram.com/pxl.pshr):
* Turn Audio ON!

(Audio credits: SKRILLEX, FRED AGAIN & FLOWDAN - RUMBLE (PHACE'S DNB FLIP))

https://user-images.githubusercontent.com/121192995/224450647-39529b28-be04-4871-bb7a-faf7afda2ef2.mp4

Setting file of that video: [here](https://github.com/deforum-art/sd-webui-deforum/files/11353167/PxlPshrWinningAnimationSettings.txt).

<br>

Main extension tab:

![image](https://user-images.githubusercontent.com/121192995/226101131-43bf594a-3152-45dd-a5d1-2538d0bc221d.png)

Keyframes tab:

![image](https://user-images.githubusercontent.com/121192995/226101140-bfe6cce7-9b78-4a1d-be9a-43e1fc78239e.png)
