# Practical-RIFE 
**[V4.0 Promotional Video (宣传视频）](https://www.bilibili.com/video/BV1J3411t7qT?p=1&share_medium=iphone&share_plat=ios&share_session_id=7AE3DA72-D05C-43A0-9838-E2A80885BD4E&share_source=QQ&share_tag=s_i&timestamp=1639643780&unique_k=rjqO0EK)**

This project is based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE) and aims to make RIFE more practical for users by adding various features and designing new models. Because improving the PSNR index is not compatible with subjective effects, we hope this part of work and our academic research are independent of each other. To reduce development pressure, this project is for engineers and developers. For common users, we recommend the following softwares:

**[SVFI (中文)](https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation) | [RIFE-App](https://grisk.itch.io/rife-app) | [FlowFrames](https://nmkd.itch.io/flowframes) | [Drop frame fixer and FPS converter](https://github.com/may-son/RIFE-FixDropFrames-and-ConvertFPS)**

Thanks to [SVFI team](https://github.com/Justin62628/Squirrel-RIFE) to support model testing on Animation. For business cooperation, please [contact our PM](mailto:wanghongyuan@megvii.com). 

## Usage
### Model List
The content of these links is under the same MIT license as this project.

We hide some models because they received some serious bug feedback or could be totally replaced by new models. 

v4.6 - 2022.9.26 | [Google Drive](https://drive.google.com/file/d/1EAbsfY7mjnXNa6RAsATj2ImAEqmHTjbE/view?usp=sharing) | [百度网盘，密码:gtkf](https://pan.baidu.com/s/1Oc1enSD7kGnoQda2MdPYsw)

v4.5 - 2022.9.14 | [Google Drive](https://drive.google.com/file/d/17Bl_IhTBexogI9BV817kTjf7eTuJEDc0/view?usp=sharing) | [百度网盘，密码:mvr0](https://pan.baidu.com/s/1gadsLlYMo8uNGA9wjhYy0w) || v4.4 - 2022.8.24 | [Google Drive](https://drive.google.com/file/d/1eI24Kou0FUdlHLkwXfk-_xiZqKaZZFZX/view?usp=sharing) | [百度网盘，密码:2q63](https://pan.baidu.com/s/1LXVWNN5cAq5CZmauobD5yQ)

v4.3 - 2022.8.17 | [Google Drive](https://drive.google.com/file/d/1xrNofTGMHdt9sQv7-EOG0EChl8hZW_cU/view?usp=sharing) | [百度网盘，密码:q83a](https://pan.baidu.com/s/12AUAeZLZf5E1_Zx6WkS3xw?pwd=q83a) || v4.2 - 2022.8.10 | [Google Drive](https://drive.google.com/file/d/1JpDAJPrtRJcrOZMMlvEJJ8MUanAkA-99/view?usp=sharing) | [百度网盘，密码:y3ad](https://pan.baidu.com/s/1Io4Z_QUaBv-O7dYERqQAPw?pwd=y3ad) 

v3.8 - 2021.6.17 | [Google Drive](https://drive.google.com/file/d/1O5KfS3KzZCY3imeCr2LCsntLhutKuAqj/view?usp=sharing) | [百度网盘, 密码:kxr3](https://pan.baidu.com/s/1X-jpWBZWe-IQBoNAsxo2mA) || v3.1 - 2021.5.17 | [Google Drive](https://drive.google.com/file/d/1xn4R3TQyFhtMXN2pa3lRB8cd4E1zckQe/view?usp=sharing) | [百度网盘, 密码:64bz](https://pan.baidu.com/s/1W4p_Ni04HLI_jTy45sVodA) 

### Installation

```
git clone git@github.com:hzwer/Practical-RIFE.git
cd Practical-RIFE
pip3 install -r requirements.txt
```
Download a model from the model list and put *.py and flownet.pkl on train_log/
### Run

**Video Frame Interpolation**

You can use our [demo video](https://drive.google.com/file/d/1i3xlKb7ax7Y70khcTcuePi6E7crO_dFc/view?usp=sharing) or your video. 
```
python3 inference_video.py --multi=2 --video=video.mp4 
```
(generate video_2X_xxfps.mp4)
```
python3 inference_video.py --multi=4 --video=video.mp4
```
(for 4X interpolation)
```
python3 inference_video.py --multi=2 --video=video.mp4 --scale=0.5
```
(If your video has high resolution, such as 4K, we recommend set --scale=0.5 (default 1.0))
```
python3 inference_video.py ---multi=4 --img=input/
```
(to read video from pngs, like input/0.png ... input/612.png, ensure that the png names are numbers)
```
python3 inference_video.py --multi=3 --video=video.mp4 --fps=60
```
(add slomo effect, the audio will be removed)

## Report Bad Cases
Please share your original video clip with us via Github issue and Google Drive. We may add it to our test set so that it is likely to be improved in later versions. It will be beneficial to attach a screenshot of the model's effect on the issue.

## Model training
Since we are in the research stage of engineering tricks, and our work and paper have not been authorized for patents nor published, we are sorry that we cannot provide users with training scripts. If you are interested in academic exploration, please refer to our academic research project [RIFE](https://github.com/hzwer/arXiv2020-RIFE). 

## To-do List
Multi-frame input of the model

Frame interpolation at any time location (Done)

Eliminate artifacts as much as possible

Make the model applicable under any resolution input

Provide models with lower calculation consumption

## Citation

```
@inproceedings{huang2022rife,
  title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Reference

Optical Flow:
[ARFlow](https://github.com/lliuz/ARFlow)  [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet)  [RAFT](https://github.com/princeton-vl/RAFT)  [pytorch-PWCNet](https://github.com/sniklaus/pytorch-pwc)

Video Interpolation: 
[DVF](https://github.com/lxx1991/pytorch-voxel-flow)  [TOflow](https://github.com/Coldog2333/pytoflow)  [SepConv](https://github.com/sniklaus/sepconv-slomo)  [DAIN](https://github.com/baowenbo/DAIN)  [CAIN](https://github.com/myungsub/CAIN)  [MEMC-Net](https://github.com/baowenbo/MEMC-Net)   [SoftSplat](https://github.com/sniklaus/softmax-splatting)  [BMBC](https://github.com/JunHeum/BMBC)  [EDSC](https://github.com/Xianhang/EDSC-pytorch)  [EQVI](https://github.com/lyh-18/EQVI) [RIFE](https://github.com/hzwer/arXiv2020-RIFE)
