import numpy as np
import os
import cv2
import random


def undistort_img(img):
    mtx = np.array([[4339.16818, 0., 802.366038], [0., 4349.33265, 647.352978], [0., 0., 1.]])
    dist = np.array([[3.24183243, -73.5637684, -8.85091754e-03, -2.37823943e-04, 398.493127]])

    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

    return undistorted_img


def resize_img(img):
    resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    return resized_img


def crop_img(img, dim, offset=[0, 0]):
    img_center = [img.shape[0]/2 + offset[0], img.shape[1]/2 + offset[1]]
    cropped_img = img[int(img_center[0] - dim[0]/2):int(img_center[0] + dim[0]/2), int(img_center[1] - dim[1]/2):int(img_center[1] + dim[1]/2)]

    resized_img = resize_img(cropped_img)

    return resized_img


if __name__ == '__main__':
    obj_list = ["S", "pacman", "star", "cone", "circleshell", "hemisphere", "doubleslope", "cubehole"]
    test_list = ["S", "pacman", "star", "cone", "circleshell", "hemisphere", "doubleslope", "cubehole"]
    stop = False
    find_center = False
    init = True

    if find_center:
        ### find center
        a = 16
        b = 16
        while True:
            img = cv2.imread('/home/won/Dropbox/marker_sim_real_images/raw/real/cuboid/34_15_dx_0_dy_0.jpg')
            #img = undistort_img(img)
            img = crop_img(img, [896, 896], [a, b])
            img = cv2.line(img, (0, 127), (255, 127), (0, 0, 0), 1)
            img = cv2.line(img, (127, 0), (127, 255), (0, 0, 0), 1)
            cv2.imshow("Real", img)
            key = cv2.waitKey(0)
            if key == 119:  # w
                a -= 1
            elif key == 115:  # s
                a += 1
            elif key == 97:  # a
                b -= 1
            elif key == 100:  # d
                b += 1
            elif key == 27:  # esc
                break

        print(a, ',', b)

    else:
        for obj_name in obj_list:
            raw_real_dir = "/home/won/Dropbox/marker_sim_real_images/raw/real/" + obj_name
            raw_sim_dir = "/home/won/Dropbox/marker_sim_real_images/raw/sim/" + obj_name

            train_real_dir = "/home/won/Dropbox/marker_sim_real_images/dataset/train/real/"
            train_sim_dir = "/home/won/Dropbox/marker_sim_real_images/dataset/train/sim/"
            test_real_dir = "/home/won/Dropbox/marker_sim_real_images/dataset/test_v2/real/"
            test_sim_dir = "/home/won/Dropbox/marker_sim_real_images/dataset/test_v2/sim/"

            real_dir_list = os.listdir(raw_real_dir)

            test_cnt_list = list(range(63))
            random.shuffle(test_cnt_list)
            test_cnt_list = test_cnt_list[:15]

            for img_file in real_dir_list:
                real_img_dir = os.path.join(raw_real_dir, img_file)
                real_img = cv2.imread(real_img_dir)
                real_img = undistort_img(real_img)
                real_img = crop_img(real_img, [896, 896])

                sim_img_dir = os.path.join(raw_sim_dir, img_file)
                sim_img = cv2.imread(sim_img_dir)
                sim_img = crop_img(sim_img, [896, 896], [-16, -16])

                '''cv2.imshow("Real", real_img)
                cv2.imshow("Sim", sim_img)
                key = cv2.waitKey(10)
                if key == 27:
                    stop = True
                    break'''

                if img_file[1] == '_':
                    cnt = int(img_file[0])
                else:
                    cnt = int(img_file[:2])

                if (obj_name in test_list) or (cnt in test_cnt_list):
                    cv2.imwrite(test_real_dir + '/' + obj_name + '_' + img_file, real_img)
                    cv2.imwrite(test_sim_dir + '/' + obj_name + '_' + img_file, sim_img)
                else:
                    cv2.imwrite(train_real_dir + '/' + obj_name + '_' + img_file, real_img)
                    cv2.imwrite(train_sim_dir + '/' + obj_name + '_' + img_file, sim_img)

            print(obj_name)

            if stop:
                break

    if init:
        init_dir = "/home/won/Dropbox/marker_sim_real_images/raw/real/init/"

        train_real_dir = "/home/won/Dropbox/marker_sim_real_images/dataset/train/real/"
        train_sim_dir = "/home/won/Dropbox/marker_sim_real_images/dataset/train/sim/"
        test_real_dir = "/home/won/Dropbox/marker_sim_real_images/dataset/test_v2/real/"
        test_sim_dir = "/home/won/Dropbox/marker_sim_real_images/dataset/test_v2/sim/"

        for obj_name in obj_list:
            init = cv2.imread(init_dir + 'init_' + obj_name + '.jpg')
            #real_img = undistort_img(init_real)
            real_img = crop_img(init, [896, 896])
            sim_img = np.ones((256, 256, 3), np.uint8) * 254

            cv2.imwrite(test_real_dir + '/' + obj_name + '_init.jpg', real_img)
            cv2.imwrite(test_sim_dir + '/' + obj_name + '_init.jpg', sim_img)
            if obj_name not in test_list:
                cv2.imwrite(train_real_dir + '/' + obj_name + '_init.jpg', real_img)
                cv2.imwrite(train_sim_dir + '/' + obj_name + '_init.jpg', sim_img)

    cv2.destroyAllWindows()
