import sys
import os
import argparse
import math

import cv2
import numpy as np
import mujoco_py as mjp

import torch
from model.networks import LSTMUnet3dGenerator
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

obj_list = ['circleshell', 'cone', 'cross', 'cubehole', 'cuboid', 'cylinder', 'doubleslope', 'hemisphere', 'line',
            'pacman', 'S', 'sphere', 'squareshell', 'star', 'tetrahedron', 'torus']

parser = argparse.ArgumentParser()
parser.add_argument('--object', '-obj', type=str, default='circleshell', choices=obj_list)
parser.add_argument('--x_init', '-x', type=float, default=0.0)
parser.add_argument('--y_init', '-y', type=float, default=0.0)
parser.add_argument('--r_init', '-r', type=float, default=0.0)
parser.add_argument('--dx', '-dx', type=float, default=0.652)
parser.add_argument('--dy', '-dy', type=float, default=-0.3)
parser.add_argument('--dz', '-dz', type=float, default=1.5)
parser.add_argument('--visualize_motion_vector', '-vis_vec', type=bool, default=False)
args = parser.parse_args()

def depth2img(depth, sim):
    extent = sim.model.stat.extent
    znear = sim.model.vis.map.znear * extent
    zfar = sim.model.vis.map.zfar * extent
    depth = 2. * depth - 1.00
    depth = 2. * znear * zfar / (zfar + znear - depth * (zfar - znear))

    depth[depth > 0.035] = 0.035
    depth[depth < 0.032] = 0.032
    image = 255 * (depth - 0.032) / 0.003

    return image.astype(np.uint8)

def crop_resize_depth_img(img):
    offset = [-16, -16]
    img_center = [img.shape[0]/2 + offset[0], img.shape[1]/2 + offset[1]]
    dim = [896, 896]
    cropped_img = img[int(img_center[0] - dim[0]/2):int(img_center[0] + dim[0]/2), int(img_center[1] - dim[1]/2):int(img_center[1] + dim[1]/2)]
    resized_img = cv2.resize(cropped_img, (256, 256), interpolation=cv2.INTER_AREA)

    return cv2.merge((resized_img, resized_img, resized_img))

def get_depth_img(sim):
    _, depth_frame = sim.render(width=1640, height=1232, camera_name='tact_cam', depth=True)
    depth_frame = cv2.flip(depth_frame, 0)
    depth_img = depth2img(depth_frame, sim)
    crop_resized_depth_img = crop_resize_depth_img(depth_img)

    cv2.imshow('simulated depth', crop_resized_depth_img)
    cv2.waitKey(1)

    transforms = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_tensor = transforms(crop_resized_depth_img)

    return img_tensor.view(1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])

def add_to_sequence(img, seq):
    return torch.cat((seq, img))

def set_sensor_pos(pos, sim):
    sim.data.qpos[:] = pos
    sim.forward()

def seq_to_input(seq):
    seq = torch.permute(seq, (1, 0, 2, 3))
    seq = seq[None, :, :, :, :]
    return seq.to(device)

def output_to_img(output):
    transform = T.Normalize((-1, -1, -1), (2, 2, 2))
    img = output[0, :, -1, :, :]
    img = transform(img)
    img_np = img.detach().cpu().numpy()
    img_np = np.transpose(img_np * 255, (1, 2, 0))
    img = img_np.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


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
    set_sensor_pos(qpos_init, sim)
    depth_seq = get_depth_img(sim)

    # move dz
    qpos = qpos_init[:]
    dz_cnt = math.ceil(args.dz / 0.1)
    for i in range(dz_cnt):
        if i < dz_cnt - 1:
            qpos[2] += 0.0001
        else:
            qpos[2] += (args.dz - 0.1 * i) * 0.001

        print(qpos[2])
        set_sensor_pos(qpos, sim)
        sim_depth = get_depth_img(sim)
        depth_seq = add_to_sequence(sim_depth, depth_seq)

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

        print(qpos)
        set_sensor_pos(qpos, sim)
        sim_depth = get_depth_img(sim)
        depth_seq = add_to_sequence(sim_depth, depth_seq)

    # generate tactile image
    G = LSTMUnet3dGenerator().to(device)
    G.load_state_dict(torch.load(os.path.join('model', 'lstm_3d_unet.pt')))
    G.eval()

    G_input = seq_to_input(depth_seq)
    G_output = G(G_input)
    gen_img = output_to_img(G_output)

    cv2.imshow('generated image', gen_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
