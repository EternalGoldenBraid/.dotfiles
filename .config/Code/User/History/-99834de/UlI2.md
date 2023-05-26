## Setup
Please start by installing [mamba](https://github.com/conda-forge/miniforge#mambaforge), [Miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) or conda
with Python3.9 or above.

either run `mamba install -f environment.yml` or install dependencies manually:

<details>
Instal the following dependencies (Conda/Mamba or pip):

- [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- numpy, opencv, trimesh, pyrender, scikit-image

At the time of writing, pip only.
- [Pyrealsense](https://pypi.org/project/pyrealsense/)
	- `pip install pyrealsense2==2.50.0.3812`
</details>

### Download Model weights for OVE6D
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
