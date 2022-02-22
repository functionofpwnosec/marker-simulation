import numpy as np
import cv2
import copy
from scipy.io import loadmat

class Deformation:
    def __init__(self, nodes_path, tensor_path, init_depth_map):
        nodes = np.loadtxt(nodes_path)
        self.nodes = nodes[:, [0, 2, 1, 3]]
        self.marker_nodes = self.set_marker_nodes()
        self.active_nodes = []
        self.tensor = self.extract_tensor(tensor_path)
        self.pxpermm = 20
        self.depthpermm = -200
        self.nodes_px = np.rint(self.nodes[:, 1:3] * self.pxpermm).astype(int)
        self.depth_map = init_depth_map
        self.depth_binary = np.zeros_like(init_depth_map)
        self.M = None

    def compute_deformation(self):
        contact_exists, active_node_change = self.find_active_nodes(self.depth_map)
        if contact_exists:
            init_constraint = self.set_constraint()
            if active_node_change:
                self.set_M()
            modified_constraint = self.modify_constraint(init_constraint)
            final_node_pos = self.calculate_passive_nodes(modified_constraint)
        else:
            final_node_pos = self.nodes[:, 1:]

        return final_node_pos

    def set_marker_nodes(self):
        marker_nodes = []
        for i in range(self.nodes.shape[0]):
            if self.nodes[i, 1] > 0 and self.nodes[i, 2] > 0 and self.nodes[i, 1] < 25 and self.nodes[i, 2] < 40 and self.nodes[i, 1] % 2 == 0 and self.nodes[i, 2] % 2 == 0:
                marker_nodes.append(i)

        return marker_nodes

    def extract_tensor(self, tensor_path):
        tensor_struct = loadmat(tensor_path)

        return tensor_struct['tensor']

    def find_active_nodes(self, depth_map):
        ret, depth_binary = cv2.threshold(depth_map, 1, 255, cv2.THRESH_BINARY)
        if depth_binary.sum() > 0:
            contact_exists = True
            if (depth_binary != self.depth_binary).sum() > 0:
                active_node_change = True
                self.active_nodes = []
                contours, hierarchy = cv2.findContours(depth_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for i in range(self.nodes.shape[0]):
                    if self.nodes_px[i, 0] < depth_map.shape[0] and self.nodes_px[i, 1] < depth_map.shape[1]:
                        for c in contours:
                            dist = cv2.pointPolygonTest(c, (self.nodes_px[i, 1], self.nodes_px[i, 0]), False)
                            if dist >= 0:
                                self.active_nodes.append(i)
            else:
                active_node_change = False
        else:
            contact_exists = False
            active_node_change = False
            self.depth_binary = depth_binary
            self.active_nodes = []

        return contact_exists, active_node_change

    def set_constraint(self):
        # z deformations only for now
        constraints = []
        for node_num in self.active_nodes:
            constraint = self.depth_map[self.nodes_px[node_num, 0], self.nodes_px[node_num, 1]]/self.depthpermm
            constraints.append(constraint)
        init_constraint_z = np.array(constraints).reshape(len(constraints), 1)
        init_constraint_xy = np.zeros((len(self.active_nodes), 2))
        init_constraint = np.hstack((init_constraint_xy, init_constraint_z))
        init_constraint = init_constraint.flatten()

        return init_constraint

    def set_M(self):
        num_active_nodes = len(self.active_nodes)
        M = np.identity(num_active_nodes*3)
        for i in range(num_active_nodes):
            for j in range(num_active_nodes):
                if i == j:
                    continue
                else:
                    M[i*3:(i+1)*3, j*3:(j+1)*3] = self.tensor[j, i, :, :]
        self.M = M

    def modify_constraint(self, init_constraint):
        modified_constraint = np.linalg.solve(self.M, init_constraint)

        return modified_constraint

    def calculate_passive_nodes(self, modified_constraint):
        final_node_pos = copy.copy(self.nodes[:, 1:])
        for i in range(final_node_pos.shape[0]):
            for j in range(len(self.active_nodes)):
                final_node_pos[i, :] += np.matmul(self.tensor[self.active_nodes[j], i, :, :], modified_constraint[j*3:(j+1)*3])

        return final_node_pos
