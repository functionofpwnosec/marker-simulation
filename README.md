# Marker-Embedded Tactile Image Generation via Generative Adversarial Networks
We present a generative adversarial network(GAN)-based method for generating realistic marker-embedded tactile images in Gelsight-like vision-based tactile sensors.

## Prerequisites
The project has been tested on Ubuntu 16.04 & 18.04 with Python 3.7.

To install the dependencies: `pip install -r requirements.txt`

Our project uses a physics simulator to obtain simulated depth images from. We use the MuJoCo simulator and require users to have it installed for using this project. Information on the installation of MuJoCo can be found [here](https://github.com/deepmind/mujoco)

## Data
We have uploaded a 
All images collected in our work can be found under `data` folder.
The `data/raw` folder contains the raw tactile and simulated depth images and the `data/dataset` folder contains the cropped & resized tactile and depth images.

## Generating images with trained generator
<img src = "doc/image_result-1.png" width="500px">

## Evaluation
We have evaluated the quality of 

<img src = "doc/marker_dzdxy.png" width="500px">
