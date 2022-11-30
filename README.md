# Marker-Embedded Tactile Image Generation via Generative Adversarial Networks
We present a generative adversarial network(GAN)-based method for generating realistic marker-embedded tactile images in Gelsight-like vision-based tactile sensors.
The trained generator in the GAN translates simulated depth image sequences to RGB marker-embedded tactile images.

For more information, please check the [paper]().

## Prerequisites
The project has been tested on Ubuntu 16.04 & 18.04 with Python 3.7.

To install the dependencies: `pip install -r requirements.txt`

Our project uses a physics simulator to obtain simulated depth images.
We use the MuJoCo simulator and require users to have it installed for using this project.
Information on the installation of MuJoCo can be found [here](https://github.com/deepmind/mujoco).

## Usage
You can choose to make contact with one of the 16 objects (listed below) using the `--object` (`-obj`) argument.
```
'circleshell', 'cone', 'cross', 'cubehole', 'cuboid', 'cylinder', 'doubleslope', 'hemisphere', 'line', 'pacman', 'S', 'sphere', 'squareshell', 'star', 'tetrahedron', 'torus'
```
You can set the initial pose of the sensor with respect to the center of the target object with `--x_init`, `--y_init`, `--r_init` (`-x`, `-y`, `-r`) arguments.
The units of `-x`, `-y` are millimeter and the unit of `-r` is degree.  

You can control the amount of normal deformation with `--dz`, (`-dz`), and lateral deformations with `--dx`, `--dy` (`-dx`, `-dy`) arguments.
The units of `-dx`, `-dy`, and `-dz` are millimeter.
Note that we have limited the range of the deformations to [0, 1.5] for `-dz`, and [-1, 1] for `-dx`, `-dy`.

Run `python main.py -obj circleshell -dx 0.2 -dy 0.3 -dz 0.7` to visualize the generated tactile image.

## Examples
<img src="doc/fig1.png" height="400px"> <img src="doc/image_result.png" height="400px">

## Custom Model and Training


## Citation
TBA
