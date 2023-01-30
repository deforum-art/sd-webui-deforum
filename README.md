
# Deforum Stable Diffusion — official extension for AUTOMATIC1111's webui

<p align="left">
    <a href="https://github.com/deforum-art/deforum-for-automatic1111-webui/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://github.com/deforum-art/deforum-for-automatic1111-webui/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://github.com/deforum-art/deforum-for-automatic1111-webui/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://github.com/deforum-art/deforum-for-automatic1111-webui/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/deforum-art/deforum-for-automatic1111-webui"></a>
    </a>
</p>

## Before Starting

**Important note about versions updating:** <br>
As auto's webui is getting updated multiple times a day, every day, things tend to break with regards to extensions compatability.
Therefore, it is best recommended to keep two folders:
1. "Stable" folder that you don't regularly update, with versions that you know *work* together (we will provide info on this soon). 
2. "Experimental" folder in which you can add 'git pull' to your webui-user.bat, update deforum every day, etc. Keep it wild - but be prepared for bugs. 


## Getting Started

1. Install [AUTOMATIC1111's webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/). <br>If the repo link doesn't work, please use the alternate official download source: [https://gitgud.io/AUTOMATIC1111/stable-diffusion-webui](https://gitgud.io/AUTOMATIC1111/stable-diffusion-webui). To change your existing webui's installation origin, execute `git remote set-url origin https://gitgud.io/AUTOMATIC1111/stable-diffusion-webui` in the webui starting folder.

2. Now two ways: either clone the repo into the `extensions` directory via git commandline launched within in the `stable-diffusion-webui` folder

```sh
git clone https://github.com/deforum-art/deforum-for-automatic1111-webui extensions/deforum
```

Or download this repository, locate the `extensions` folder within your WebUI installation, create a folder named `deforum` and put the contents of the downloaded directory inside of it. Then restart WebUI. **Warning: the extension folder has to be named 'deforum' or 'deforum-for-automatic1111-webui', otherwise it will fail to locate the 3D modules as the PATH addition is hardcoded**

3. Open the webui, find the Deforum tab at the top of the page.

4. Enter the animation settings. Refer to [this general guide](https://docs.google.com/document/d/1pEobUknMFMkn8F5TMsv8qRzamXX_75BShMMXV8IFslI/edit) and [this guide to math keyframing functions in Deforum](https://docs.google.com/document/d/1pfW1PwbDIuW0cv-dnuyYj1UzPqe23BlSLTJsqazffXM/edit?usp=sharing). However, **in this version prompt weights less than zero don't just like in original Deforum!** Split the positive and the negative prompt in the json section using --neg argument like this "apple:\`where(cos(t)>=0, cos(t), 0)\`, snow --neg strawberry:\`where(cos(t)<0, -cos(t), 0)\`"

5. To view animation frames as they're being made, without waiting for the completion of an animation, go to the 'Settings' tab and set the value of this toolbar **above zero**. Warning: it may slow down the generation process. If you have 'Do exactly the amount of steps the slider specifies' checkbox selected in the tab, unselect it as it won't allow you to use Deforum schedules and you will get adrupt frame changes without transitions. Then click 'Apply settings' at the top of the page. Now return to the 'Deforum' tab.

![adsdasunknown](https://user-images.githubusercontent.com/14872007/196064311-1b79866a-e55b-438a-84a7-004ff30829ad.png)


6. Run the script and see if you got it working or even got something. **In 3D mode a large delay is expected at first** as the script loads the depth models. In the end, using the default settings the whole thing should consume 6.4 GBs of VRAM at 3D mode peaks and no more than 3.8 GB VRAM in 3D mode if you launch the webui with the '--lowvram' command line argument.

7. After the generation process is completed, click the button with the self-describing name to show the video or gif result right in the GUI!

8. Join our Discord where you can post generated stuff, ask questions and more: https://discord.gg/deforum. <br>
* There's also the 'Issues' tab in the repo, for well... reporting issues ;) 

9. Profit!

## Known issues

* This port is not fully backward-compatible with the notebook and the local version both due to the changes in how AUTOMATIC1111's webui handles Stable Diffusion models and the changes in this script to get it to work in the new environment. *Expect* that you may not get exactly the same result or that the thing may break down because of the older settings.

## Screenshots

https://user-images.githubusercontent.com/121192995/215522284-d6fbedd5-09e2-4d2c-bd10-f9bbb4a20f82.mp4

Main extension tab:

![maintab](https://user-images.githubusercontent.com/121192995/215362176-4e5599c1-9cb6-4bf9-964d-0ff882661993.png)

Keyframes tab:

![keyframes](https://user-images.githubusercontent.com/121192995/215362228-c239c43a-d565-4862-b490-d18b19eaaaa5.png)

Math evaluation:

![math-eval](https://user-images.githubusercontent.com/121192995/215362467-481127a4-247a-4b0d-924a-d10719aa4c01.png)


## Benchmarks

3D mode without additional WebUI flags

![image](https://user-images.githubusercontent.com/14872007/196294447-7817f138-ec4b-4001-885f-454f8667100d.png)

3D mode when WebUI is launched with '--lowvram'

![image](https://user-images.githubusercontent.com/14872007/196294517-125fbb27-c06d-4c4b-bcbc-7c743103eff6.png)

