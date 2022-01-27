import mujoco_py as mjp
import numpy as np
import cv2
import time

def create_test_depth_maps(width, height, obj_pos_x, obj_pos_y, obj_size, num_of_frames):
    depth_map = np.zeros((height, width), dtype=np.uint8)
    object_depth = np.ones((obj_size, obj_size), dtype=np.uint8)
    for i in range(int(obj_size/2)):
        object_depth[i:obj_size-i, i:obj_size-i] = i+1
    depth_map[obj_pos_y:obj_pos_y+obj_size, obj_pos_x:obj_pos_x+obj_size] = object_depth
    for i in range(num_of_frames):
        ret, depth_mask = cv2.threshold(depth_map, i, 255, cv2.THRESH_BINARY)
        cv2.bitwise_and(depth_mask, depth_mask, mask=depth_mask)

    return depth_maps

if __name__ == '__main__':
    test_depth_maps = create_test_depth_maps(800, 500, 240, 180, 100, 2)
    for i in range(test_depth_maps.shape[2]):
        test_depth_map = test_depth_maps[:, :, i]
        cv2.imshow("test", test_depth_map)
        time.sleep(0.1)
