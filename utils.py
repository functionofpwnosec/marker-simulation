import numpy as np
import cv2
import torch
import torchvision.transforms as T

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

    return seq[None, :, :, :, :]

def output_to_img(output):
    transform = T.Normalize((-1, -1, -1), (2, 2, 2))
    img = output[0, :, -1, :, :]
    img = transform(img)
    img_np = img.detach().cpu().numpy()
    img_np = np.transpose(img_np * 255, (1, 2, 0))
    img = img_np.astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
