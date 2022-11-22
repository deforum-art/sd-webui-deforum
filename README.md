
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

0. ~~Cover yourself in oil~~ If you have legacy Deforum installed *as a script* (it is located in stable-diffusion-webui/scripts), remove it (`deforum.py` and the folder `deforum`) and proceed with the instructions below to get more interactive Deforum with an improved UI.

1. Install [AUTOMATIC1111's webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/). Make sure it's updated to the newer version that supports *extensions* (i.e. AUTOMATIC1111 versions released after 2022-10-21).

2. Now two ways: either clone the repo into the `extensions` directory via git commandline launched within in the `stable-diffusion-webui` folder

```sh
git clone https://github.com/deforum-art/deforum-for-automatic1111-webui/ extensions/deforum
```

Or download this repository, locate the `extensions` folder within your WebUI installation, create folder named `deforum` in it and then put the contents of the archive inside. Then restart WebUI. **Warning: the extension folder has to be named 'deforum' or else it will fail to locate the 3D modules as the PATH addition is hardcoded**

3. ~~If you're on Windows and want to launch Deforum in 3D mode, you'll have to download the depths model manually. Download these files https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt and https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt and put them into the 'models/Deforum' folder of your webui installation. (if it doesn't exist, create it)~~ Since 2022-11-20 the depth models are downloaded automatically on any platform supported by the vanilla webui.

4. Open the webui, find the Deforum tab at the top of the page.

5. Enter the animation settings. Refer to [this general guide](https://docs.google.com/document/d/1pEobUknMFMkn8F5TMsv8qRzamXX_75BShMMXV8IFslI/edit) and [this guide to math keyframing functions in Deforum](https://docs.google.com/document/d/1pfW1PwbDIuW0cv-dnuyYj1UzPqe23BlSLTJsqazffXM/edit?usp=sharing). However, **in this version prompt weights less than zero don't just like in original Deforum!** Split the positive and the negative prompt in the json section using --neg argument like this "apple:\`where(cos(t)>=0, cos(t), 0)\`, snow --neg strawberry:\`where(cos(t)<0, -cos(t), 0)\`"

6. To view animation frames as they're being made, without waiting for the completion of an animation, go to the 'Settings' tab and set the value of this toolbar **above zero**. Warning: it may slow down the generation process. If you have 'Do exactly the amount of steps the slider specifies' checkbox selected in the tab, unselect it as it won't allow you to use Deforum schedules and you will get adrupt frame changes without transitions. Then click 'Apply settings' at the top of the page. Now return to the 'Deforum' tab.

![adsdasunknown](https://user-images.githubusercontent.com/14872007/196064311-1b79866a-e55b-438a-84a7-004ff30829ad.png)


7. Run the script and see if you got it working or even got something. **In 3D mode a large delay is expected at first** as the script loads the depth models. In the end, using the default settings the whole thing should consume 6.4 GBs of VRAM at 3D mode peaks and no more than 3.8 GB VRAM in 3D mode if you launch the webui with the '--lowvram' command line argument.

8. If it gives errors on missing modules, such as about missing 'numexpr', go to the original webui directory, open 'requirements_versions.txt' and append the missing packages names at the end of that file. Then restart the webui.

9. After the generation process is completed, click the button with the self-describing name to show the video or gif result right in the GUI!

10. Join our Discord where you can post generated stuff, ask questions and ~~infuriate the devs with 'this feature is in auto's build. When will it be in Deforum? Why can't I launch Deforum on my potato computer?'~~(not anymore, ha-ha) https://discord.gg/deforum. There's also the 'Issues' tab in the repo.

11. In case the overhauled version is too unusable for you or you cannot update AUTOMATIC1111's webui, roll back to this script-only version https://github.com/deforum-art/deforum-for-automatic1111-webui/tree/000deeeef69016612fe3cdec9234f97d87d30748

12. Profit!

## Known issues

* This port is not fully backward-compatible with the notebook and the local version both due to the changes in how AUTOMATIC1111's webui handles Stable Diffusion models and the changes in this script to get it to work in the new environment. *Expect* that you may not get exactly the same result or that the thing may break down because of the older settings.

* Color correction is quite forced atm.

* Browsers often cannot load too big gifs, so try to use `ffmpeg` when possible. Make sure it's installed and linked in your PATH!

* ~~3D mode doesn't work on resolutions when one of the dimensions is greater than 768 or less than 448.~~ has been fixed in [#84](https://github.com/deforum-art/deforum-for-automatic1111-webui/pull/84), so update!

* If you encounter issues with the downloaded depth models failing to load, try launching the WebUI with the `--disable-safe-unpickle` option! [How to add it to the .bat file on Windows.](https://imgur.com/a/TJHglot)

## Screenshots

Proof that it works good enough of AUTOMATIC1111's build with MATH keyframing and prompt-weighting enabled in 3D mode



https://user-images.githubusercontent.com/14872007/197588218-6c42c54f-6874-46df-a650-e41433a09f74.mp4



'Le not so Grand Interface' at work:


![Screenshot 2022-10-24 at 20-04-42 Stable Diffusion](https://user-images.githubusercontent.com/14872007/197587723-290a7ab6-b272-49ca-aeb3-958d5f1f6a37.png)


Math evaluation:

![math-eval](https://user-images.githubusercontent.com/14872007/195957601-3c3fecab-5ef2-4a2f-9eba-3bb0c70bd4b8.png)


## Benchmarks

3D mode without additional WebUI flags

![image](https://user-images.githubusercontent.com/14872007/196294447-7817f138-ec4b-4001-885f-454f8667100d.png)

3D mode when WebUI is launched with '--lowvram'

![image](https://user-images.githubusercontent.com/14872007/196294517-125fbb27-c06d-4c4b-bcbc-7c743103eff6.png)

