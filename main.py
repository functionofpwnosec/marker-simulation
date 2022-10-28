import sys
import os
import argparse
import numpy as np
#import cv2
import torch
import mujoco_py as mjp

obj_list = ['circleshell', 'cone', 'cross', 'cubehole', 'cuboid', 'cylinder', 'doubleslope', 'hemisphere', 'line',
            'pacman', 'S', 'sphere', 'squareshell', 'star', 'tetrahedron', 'torus']

parser = argparse.ArgumentParser()
parser.add_argument('--obj', type=str, default='circleshell', choices=obj_list)
parser.add_argument('--x', type=float, default=0.0)
parser.add_argument('--y', type=float, default=0.0)
parser.add_argument('--r', type=float, default=0.0)
parser.add_argument('--dx', type=float, default=0.0)
parser.add_argument('--dy', type=float, default=0.0)
parser.add_argument('--dz', type=float, default=0.0)
parser.add_argument('--visualize_sim', type=bool, default=False)

args = parser.parse_args()

def depth2img(depth):
    extent = sim.model.stat.extent
    znear = sim.model.vis.map.znear * extent
    zfar = sim.model.vis.map.zfar * extent
    depth = 2. * depth - 1.00
    depth = 2. * znear * zfar / (zfar + znear - depth * (zfar - znear))

    depth[depth > 0.035] = 0.035
    depth[depth > 0.032] = 0.032
    image = 255 * (depth - 0.032) / 0.003

    return image.astype(np.uint8)

def get_depth_img(sim):
    _, depth_frame = sim.render(width=1640, height=1232, camera_name='tact_cam', depth=True)

    return depth2img(depth_frame)


if __name__ == '__main__':
    if args.dz < 0 or args.dz > 1.5:
        raise ValueError('dz should be in range [0, 1.5]')
    if args.dx < -1 or args.dx > 1:
        raise ValueError('dx should be in range [-1, 1]')
    if args.dy < -1 or args.dy > 1:
        raise ValueError('dy should be in range [-1, 1]')

    xml_path = os.path.join('assets', 'sim.xml')
    model = mjp.load_model_from_path(xml_path)
    sim = mjp.MjSim(model)
    sim.forward()

    obj_name = 'obj' + str(obj_list.index(args.obj))
    obj_pos = sim.data.get_body_xpos(obj_name)

    # set to initial position
    qpos = [obj_pos[0] + args.x, obj_pos[1] + args.y, 0, args.r * np.pi / 180]
    sim.data.qpos[:] = qpos
    sim.forward()

    frame = get_depth_img(sim)

    # move dz
    if args.dz != 0:
