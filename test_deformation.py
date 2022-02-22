import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.deformation import Deformation

def create_test_depth_maps(width, height, obj_pos_x, obj_pos_y, obj_size, num_of_frames):
    depth_map = np.zeros((height, width), dtype=np.uint8)
    object_depth = np.ones((obj_size, obj_size), dtype=np.uint8)
    for i in range(int(obj_size/2)):
        #depth_map[obj_pos_y-int(obj_size/2)+i:obj_pos_y+int(obj_size/2)-i, obj_pos_x-int(obj_size/2)+i:obj_pos_x+int(obj_size/2)-i] = i*2
        depth_map[obj_pos_y - int(obj_size / 2) + i:obj_pos_y + int(obj_size / 2) - i,
        obj_pos_x - int(obj_size / 2) + i:obj_pos_x + int(obj_size / 2) - i] = 100

    depth_maps = []
    for i in range(num_of_frames):
        ret, depth_mask = cv2.threshold(depth_map, i*2, 255, cv2.THRESH_BINARY)
        temp = cv2.bitwise_and(depth_map, depth_mask)
        depth_maps.append(temp)
    depth_maps.reverse()

    return depth_maps

if __name__ == '__main__':
    test_depth_maps = create_test_depth_maps(800, 500, 240, 180, 100, 100)
    for i in range(len(test_depth_maps)):
        test_depth_map = test_depth_maps[i]
        cv2.imshow("test", test_depth_map)
        cv2.waitKey(50)

    cv2.destroyAllWindows()

    gel_deformation = Deformation("assets/membrane_nodes.txt", "assets/tensor.mat", test_depth_maps[0])
    print(gel_deformation.marker_nodes)

    for i in range(len(test_depth_maps)):
        gel_deformation.depth_map = test_depth_maps[i]
        nodes = gel_deformation.compute_deformation()
        if i % 10 == 0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            '''ax.set_proj_type('persp')
            ax.view_init(-90, 0)'''
            ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], marker='.', color='gray')
            ax.scatter(nodes[gel_deformation.marker_nodes, 0], nodes[gel_deformation.marker_nodes, 1], nodes[gel_deformation.marker_nodes, 2], marker='o', color='k')
            plt.show()
            '''plt.scatter(nodes[:, 1], nodes[:, 2])
            plt.scatter(nodes[gel_deformation.marker_nodes, 1], nodes[gel_deformation.marker_nodes, 2], color='k')
            plt.xticks(np.arange(0, 40))
            plt.yticks(np.arange(0, 5))
            plt.show()'''


