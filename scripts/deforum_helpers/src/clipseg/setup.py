from setuptools import setup

with open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "scipy",
    "matplotlib",
    "torch",
    "torchvision",
    "opencv-python",
    "CLIP @ git+https://github.com/openai/CLIP.git"
]

setup(
    name='clipseg',
    packages=['clipseg'],
    package_dir={'clipseg': 'models'},
    package_data={'clipseg': [
        "../weights/*.pth",
    ]},
    version='0.0.1',
    url='https://github.com/timojl/clipseg',
    python_requires='>=3.9',
    install_requires=requirements,
    description='This repository contains the code used in the paper "Image Segmentation Using Text and Image Prompts".',
    long_description=readme,
    long_description_content_type="text/markdown",
)
