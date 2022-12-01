import os
import argparse
import math

import cv2
import numpy as np
import mujoco_py as mjp

import torch
from model.networks import LSTMUnet3dGenerator
import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

obj_list = ['circleshell', 'cone', 'cross', 'cubehole', 'cuboid', 'cylinder', 'doubleslope', 'hemisphere', 'line',
            'pacman', 'S', 'sphere', 'squareshell', 'star', 'tetrahedron', 'torus']

parser = argparse.ArgumentParser()
parser.add_argument('--object', '-obj', type=str, default='circleshell', choices=obj_list)
parser.add_argument('--x_init', '-x', type=float, default=0.)
parser.add_argument('--y_init', '-y', type=float, default=0.)
parser.add_argument('--r_init', '-r', type=float, default=0.)
parser.add_argument('--dx', '-dx', type=float, default=0.)
parser.add_argument('--dy', '-dy', type=float, default=0.)
parser.add_argument('--dz', '-dz', type=float, default=0.)
parser.add_argument('--visualize_motion_vector', '-vis_vec', type=bool, default=True)
parser.add_argument('--vector_magnification', '-mag', type=float, default=5.0)
args = parser.parse_args()


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

    obj_name = 'obj' + str(obj_list.index(args.object))
    obj_pos = sim.data.get_body_xpos(obj_name)

    # set to initial position
    qpos_init = [obj_pos[0] + args.x_init * 0.001, obj_pos[1] + args.y_init * 0.001, 0., args.r_init * np.pi / 180]
    utils.set_sensor_pos(qpos_init, sim)
    depth_seq = utils.get_depth_img(sim)

    # move dz
    qpos = qpos_init[:]
    dz_cnt = math.ceil(args.dz / 0.1)
    for i in range(dz_cnt):
        if i < dz_cnt - 1:
            qpos[2] += 0.0001
        else:
            qpos[2] += (args.dz - 0.1 * i) * 0.001

        #print(qpos[2])
        utils.set_sensor_pos(qpos, sim)
        sim_depth = utils.get_depth_img(sim)
        depth_seq = utils.add_to_sequence(sim_depth, depth_seq)

    # move dx and/or dy
    dxy = math.sqrt(args.dx ** 2 + args.dy ** 2)
    if dxy != 0.:
        dxy_step = [0.0001 * args.dx / dxy, 0.0001 * args.dy / dxy]
    dxy_cnt = math.ceil(dxy / 0.1)
    for i in range(dxy_cnt):
        if i < dxy_cnt - 1:
            qpos[0] += dxy_step[0]
            qpos[1] += dxy_step[1]
        else:
            qpos[0] += 0.001 * args.dx - dxy_step[0] * i
            qpos[1] += 0.001 * args.dy - dxy_step[1] * i

        #print(qpos)
        utils.set_sensor_pos(qpos, sim)
        sim_depth = utils.get_depth_img(sim)
        depth_seq = utils.add_to_sequence(sim_depth, depth_seq)

    # generate tactile images
    G = LSTMUnet3dGenerator().to(device)
    G.load_state_dict(torch.load(os.path.join('model', 'lstm_3d_unet.pt')))
    G.eval()

    G_input = utils.seq_to_input(depth_seq).to(device)
    G_output = G(G_input)
    gen_img = utils.output_to_img(G_output)

    if args.visualize_motion_vector and args.dz != 0.:
        init_gen_img = utils.output_to_img(G_output, n=1)
        gen_img = utils.output_to_img(G_output)
        utils.draw_motion_vectors(init_gen_img, gen_img, mag=args.vector_magnification)

    print('Press any key to terminate...')
    cv2.imshow('generated image', gen_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
