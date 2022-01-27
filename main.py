import mujoco_py as mjp
import numpy as np
import matplotlib.pyplot as plt
from utils.deformation import Deformation


if __name__ == '__main__':
    xml_path = "assets/sim.xml"
    model = mjp.load_model_from_path(xml_path)
    sim = mjp.MjSim(model)
    #viewer = mjp.MjViewer(sim)

    gel_def = Deformation("assets/tensor.mat")

    a = sim.render(width=1640, height=1232, camera_name='tactile_cam', depth=True)
    rgb_img = a[0][616-360:616+360, 820-640:820+640, :]
    depth_img = a[1][616-360:616+360, 820-640:820+640]
    plt.imshow(depth_img)
    plt.show()
    print(np.max(depth_img))