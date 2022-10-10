
# Deforum Stable Diffusion

<p align="left">
    <a href="https://github.com/deforum/stable-diffusion/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/deforum/stable-diffusion"></a>
    <a href="https://github.com/deforum/stable-diffusion/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/deforum/stable-diffusion"></a>
    <a href="https://github.com/deforum/stable-diffusion/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/deforum/stable-diffusion"></a>
    <a href="https://github.com/deforum/stable-diffusion/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/deforum/stable-diffusion"></a>
    <a href="https://colab.research.google.com/github/deforum/stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>  
    <a href="https://replicate.com/deforum/deforum_stable_diffusion"><img alt="Replicate" src="https://replicate.com/deforum/deforum_stable_diffusion/badge"></a>
</p>

## Before Starting
install anaconda for managing python environments and packages https://www.anaconda.com/

## Getting Started
clone the github repository:
```
git clone -b local https://github.com/deforum/stable-diffusion.git
cd stable-diffusion

```
create anaconda environment:
```
conda create -n dsd python=3.9 -y
conda activate dsd
conda install pytorch cudatoolkit=11.6 torchvision torchaudio -c pytorch -c conda-forge -y

```
install required packages:
```
python -m pip install -r requirements.txt

```

## Windows Users
the midas and adabins model downloads are broken for windows at the moment. windows users will need to manually download model weights and place in the models folders. note: if you do not specify an existing models folder, the folder will be created automatically when you run either the .py or .ipynb for the first time.

manual download links:
https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt
https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt

## Starting Over
the stable-diffusion folder can be deleted and the dsd conda environment can be removed with the following set of commands:
```
conda deactivate
conda env remove -n dsd

```
with the dsd environment removed you can start over.

## Running Locally
make sure the dsd conda environment is active:
```
conda activate dsd

```
navigate to the stable-diffusion folder and run either the Deforum_Stable_Diffusion.py or the Deforum_Stable_Diffusion.ipynb. running the .py is the quickest and easiest way to check that your installation is working, however, it is not the best environment for tinkering with prompts and settings.
```
python Deforum_Stable_Diffusion.py

```
if you prefer a more colab-like experience you can run the .ipynb in jupyter-lab or jupyter-notebook. activate jupyter-lab or jupyter-notebook from within the stable-diffusion folder with either of the following commands:
```
jupyter-lab
jupyter notebook

```

## Colab Local Runtime
make sure the dsd conda environment is active:
```
conda activate dsd

```
open google colab. file > upload notebook > select .ipynb file in the stable-diffusion folder. enable jupyter extension
```
jupyter serverextension enable --py jupyter_http_over_ws

```
start server
```
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0
  
```
copy paste url token.
