import math
import mujoco_py as mjp
import numpy as np
from matplotlib import pyplot as plt
#import open3d as o3d

xml_path = "assets/gelsight.xml"
model = mjp.load_model_from_path(xml_path)
sim = mjp.MjSim(model)

def depthimg2meters(depth):
    extent = sim.model.stat.extent
    znear = sim.model.vis.map.znear * extent
    zfar = sim.model.vis.map.zfar * extent
    depth = 2. * depth - 1.00
    image = 2. * znear * zfar / (zfar + znear - depth * (zfar - znear))
    return image

def captureimage():
    rgb, depth = sim.render(width = 1280, height = 720, camera_name = 'depth_camera', depth = True)
    depth = np.flip(depth, axis=0)
    rgb = np.flip(rgb, axis=0)
    real_depth = depthimg2meters(depth)
    return rgb, real_depth
'''
def cameramat():
    width = 1280
    height = 720
    aspect_ratio = width/height
    fovy = math.radians(sim.model.cam_fovy)
    f = height / (2 * math.tan(fovy / 2))
    cx = width / 2
    cy = height / 2
    cam_mat = o3d.camera.PinholeCameraIntrinsic(width, height, f, f, cx, cy)
    return cam_mat

def makepcd(depth, cam_mat):
    depth_img = o3d.geometry.Image(depth)
    pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth_img,cam_mat)
    return pointcloud
'''

while True:
    rgb, depth = captureimage()
    print(depth)
    plt.imshow(depth)
    plt.gray()
    plt.show()
    #cam_mat = cameramat()
    #print(cam_mat)
    #pcd = makepcd(depth,cam_mat)
    #o3d.visualization.draw_geometries([pcd])