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


def output_to_img(output, n=-1):
    transform = T.Normalize((-1, -1, -1), (2, 2, 2))
    img = output[0, :, n, :, :]
    img = transform(img)
    img_np = img.detach().cpu().numpy()
    img_np = np.transpose(img_np * 255, (1, 2, 0))
    img = img_np.astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def marker_mask(img):
    img = img.astype(np.float32)
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    blur2 = cv2.GaussianBlur(img, (3, 3), 0)
    diff = blur - blur2
    diff *= 8.0

    diff[diff < 0.] = 0.
    diff[diff > 255.] = 255.

    mask_b = diff[:, :, 0] > 100
    mask_g = diff[:, :, 1] > 100
    mask_r = diff[:, :, 2] > 100

    mask = ((mask_b * mask_g) + (mask_b * mask_r) + (mask_g * mask_r)) > 0

    mask = mask.astype(np.uint8)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    return mask * 255


def find_markers(mask):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 1
    params.maxThreshold = 12
    params.minDistBetweenBlobs = 16
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = False
    params.minArea = 8
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    #params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask.astype(np.uint8))

    markers = []
    for keypoint in keypoints:
        marker_info = [keypoint.pt[0], keypoint.pt[1]]
        markers.append(marker_info)

    return markers


def sort_markers(markers):
    markers.sort(key=lambda x: x[1])
    sub_marker = []
    sorted_marker = []
    for m in markers:
        sub_marker.append(m)
        if len(sub_marker) == 8:
            sub_marker.sort(key=lambda x: x[0])
            sorted_marker.extend(sub_marker)
            sub_marker = []
    return np.array(sorted_marker, np.float32)


def draw_motion_vectors(init_img, img, mag=5.0):
    init_mask = marker_mask(init_img)
    init_marker = find_markers(init_mask)
    init_marker = sort_markers(init_marker)
    mask = marker_mask(img)
    marker = find_markers(mask)
    marker = sort_markers(marker)
    marker_mag = mag * (marker - init_marker) + marker

    for i in range(len(marker)):
        cv2.arrowedLine(img, tuple(marker[i].astype(int)), tuple(marker_mag[i].astype(int)), (0, 255, 255), 2, tipLength=0.4)

