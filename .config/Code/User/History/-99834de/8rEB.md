## Setup
Please start by installing [mamba](https://github.com/conda-forge/miniforge#mambaforge), [Miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) or conda
with Python3.9 or above.

### Conda/Mamba

mamba env create --file environment.yml

In conda env install pip and with pip install:

[Pyrealsense](https://pypi.org/project/pyrealsense/)
- `pip install pyrealsense2==2.50.0.3812`

### Model weights for OVE6D
`mkdir checkpoints; cd checkpoints`

**Pose estimation weights** \
- `wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aXkYOpvka5VAPYUYuHaCMp0nIvzxhW9X' -O OVE6D_pose_model.pth` \
or
- `wget https://drive.proton.me/urls/2GQBGB2DH4#aLLLp43rOm8M -O OVE6D_pose_model.pth`

or from:
OVE6D: [Project page](https://dingdingcai.github.io/ove6d-pose/) 
	- https://drive.google.com/drive/folders/16f2xOjQszVY4aC-oVboAD-Z40Aajoc1s?usp=sharing).

# Acknowledgements
- OVE6D: [Project page](https://dingdingcai.github.io/ove6d-pose/) 
- Chromakey [segmentation](https://en.wikipedia.org/wiki/Chroma_key)
