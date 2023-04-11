# Sat-NeRF

This project implements the Shadow Neural Radiance (S-NeRF) field from [this repository](https://github.com/esa/snerf),
starting from a modified PyTorch [NeRF implementation](https://github.com/yenchenlin/nerf-pytorch). The model
is able to generate novel views from a sparse collection of satellite images of a scene, as well as estimating a
Digital Elevation Model (DEM) of the surface.

<p align="center">
  <img src="https://github.com/fsemerar/satnerf/raw/main/figs/dsm.jpg" width="80%"></img>
</p>

## Dataset

The satellite image dataset that was used can be found at [this link](https://zenodo.org/record/5070039#.YWy-19nMIws)
and it should be placed in a folder called "data" (e.g. data/068 for the images of JAX). The data can be augmented by
modifying the inputs in the data_augmentation.py script and then running the command below. You can pass gauss=True to add gaussian blur in the augmented image along with sigma=0.2 to specify the radius of the gaussian blur.

    python scripts/data_augmentation.py --gauss=True --sigma=0.2

## Installation

It is recommended to create a conda environment using the following command from the root project folder:

    conda env create
    conda activate satnerf

Then follow the instructions recommended on [this website](https://pytorch.org/get-started/locally/) in order to install
the correct version of PyTorch (CPU or GPU enabled).

## How to run

To train NeRF on an example dataset run:

    python run_nerf.py --config configs/068/068_config.txt

## Project Contributors

- Federico Semeraro fsemeraro6 AT gatech.edu
- Yi Zhang yzhang3416 AT gatech.edu
- Wenying Wu wwu393 AT gatech.edu
- Patrick Carroll pcarroll7 AT gatech.edu

## Cite

If you use Sat-NeRF in your research, please use the following BibTeX entries to cite it:

```BibTeX
@misc{semeraro2023nerf,
      title={NeRF applied to satellite imagery for surface reconstruction}, 
      author={Federico Semeraro and Yi Zhang and Wenying Wu and Patrick Carroll},
      year={2023},
      eprint={2304.04133},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
