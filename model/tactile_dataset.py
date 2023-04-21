import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from skimage import io


class TactileDataset(Dataset):
    def __init__(self, is_train=True, only_normal=False):
        super().__init__()
        if is_train:
            phase = 'train'
        else:
            phase = 'test'
        self.dir_real = os.path.join('dataset', phase, 'real')
        self.dir_sim = os.path.join('dataset', phase, 'sim')
        self.img_list = os.listdir(self.dir_real)
        if only_normal:
            normal_img_list = []
            for img in self.img_list:
                if ('dx_0_dy_0' in img) or ('init' in img):
                    normal_img_list.append(img)
            self.img_list = normal_img_list
        self.transforms = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        real_name = os.path.join(self.dir_real, self.img_list[idx])
        sim_name = os.path.join(self.dir_sim, self.img_list[idx])

        real_raw_img = io.imread(real_name)
        real_img = self.transforms(real_raw_img)

        sim_raw_img = io.imread(sim_name)
        sim_img = self.transforms(sim_raw_img)

        return real_img, sim_img


class TactileSequenceDataset_test(Dataset):
    def __init__(self, is_train=False):
        super().__init__()
        if is_train:
            phase = 'train'
        else:
            phase = 'test'
        self.dir_real = os.path.join('dataset', phase, 'real')
        self.dir_sim = os.path.join('dataset', phase, 'sim')
        self.img_list = os.listdir(self.dir_real)
        self.transforms = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]

        real_name = os.path.join(self.dir_real, img_name)
        sim_name = os.path.join(self.dir_sim, img_name)

        real_raw_img = io.imread(real_name)
        real_img = self.transforms(real_raw_img)

        sim_raw_img = io.imread(sim_name)
        sim_img = self.transforms(sim_raw_img)

        real_img_seq = real_img.view(1, 3, 256, 256)
        sim_img_seq = sim_img.view(1, 3, 256, 256)

        if 'init' in img_name:
            pass
        elif 'dx_0_dy_0' in img_name:
            real_img_seq, sim_img_seq = self.create_normal_sequence(img_name, real_img_seq, sim_img_seq)
        else:
            real_img_seq, sim_img_seq = self.create_shear_sequence(img_name, real_img_seq, sim_img_seq)
            real_img_seq, sim_img_seq = self.create_normal_sequence(img_name, real_img_seq, sim_img_seq)

        real_img_seq = real_img_seq.permute((1, 0, 2, 3))
        sim_img_seq = sim_img_seq.permute((1, 0, 2, 3))

        return real_img_seq, sim_img_seq

    def create_normal_sequence(self, img_name, real_img, sim_img):
        img_name_split = img_name[:-4].split('_')
        img_name_split[-1] = '0'
        img_name_split[-3] = '0'

        for i in range(int(img_name_split[-5]) - 1, 0, -1):
            img_name_split[-5] = str(i)
            prev_img_name = '_'.join(img_name_split) + '.jpg'

            prev_real_raw_img = io.imread(os.path.join(self.dir_real, prev_img_name))
            prev_real_img = self.transforms(prev_real_raw_img)

            prev_sim_raw_img = io.imread(os.path.join(self.dir_sim, prev_img_name))
            prev_sim_img = self.transforms(prev_sim_raw_img)

            real_img = torch.cat((prev_real_img.view(1, 3, 256, 256), real_img))
            sim_img = torch.cat((prev_sim_img.view(1, 3, 256, 256), sim_img))

        if len(img_name_split) == 7:
            init_name = img_name_split[0] + '_init.jpg'
        else:
            init_name = img_name_split[0] + '_' + img_name_split[1] + '_init.jpg'

        init_real_raw_img = io.imread(os.path.join(self.dir_real, init_name))
        init_real_img = self.transforms(init_real_raw_img)

        init_sim_raw_img = io.imread(os.path.join(self.dir_sim, init_name))
        init_sim_img = self.transforms(init_sim_raw_img)

        real_img_seq = torch.cat((init_real_img.view(1, 3, 256, 256), real_img))
        sim_img_seq = torch.cat((init_sim_img.view(1, 3, 256, 256), sim_img))

        return real_img_seq, sim_img_seq

    def create_shear_sequence(self, img_name, real_img, sim_img):
        img_name_split = img_name[:-4].split('_')
        if int(img_name_split[-3]) != 0:
            shear_idx = -3  # x shear
        else:
            shear_idx = -1  # y shear

        for i in range(int(img_name_split[shear_idx]) - 1, -1, -1):
            img_name_split[shear_idx] = str(i)
            prev_img_name = '_'.join(img_name_split) + '.jpg'

            prev_real_raw_img = io.imread(os.path.join(self.dir_real, prev_img_name))
            prev_real_img = self.transforms(prev_real_raw_img)

            prev_sim_raw_img = io.imread(os.path.join(self.dir_sim, prev_img_name))
            prev_sim_img = self.transforms(prev_sim_raw_img)

            real_img = torch.cat((prev_real_img.view(1, 3, 256, 256), real_img))
            sim_img = torch.cat((prev_sim_img.view(1, 3, 256, 256), sim_img))

        return real_img, sim_img


class TactileSequenceDataset(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        if is_train:
            phase = 'train'
            self.img_list_file = open("dataset/train.txt", 'r')
        else:
            phase = 'test'
            self.img_list_file = open("dataset/test.txt", 'r')
        self.dir_real = os.path.join('dataset', phase, 'real')
        self.dir_sim = os.path.join('dataset', phase, 'sim')
        self.img_list = self.img_list_file.readlines()
        self.transforms = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx][:-1]

        sequence_list = self.create_sequence_list(img_name)

        real_init_name = os.path.join(self.dir_real, img_name.split('_')[0] + "_init.jpg")
        sim_init_name = os.path.join(self.dir_sim, img_name.split('_')[0] + "_init.jpg")

        real_init_raw_img = io.imread(real_init_name)
        real_init_img = self.transforms(real_init_raw_img)

        sim_init_raw_img = io.imread(sim_init_name)
        sim_init_img = self.transforms(sim_init_raw_img)

        real_img_seq = real_init_img.view(1, 3, 256, 256)
        sim_img_seq = sim_init_img.view(1, 3, 256, 256)

        for img in sequence_list:
            real_name = os.path.join(self.dir_real, img)
            sim_name = os.path.join(self.dir_sim, img)

            real_raw_img = io.imread(real_name)
            real_img = self.transforms(real_raw_img)

            sim_raw_img = io.imread(sim_name)
            sim_img = self.transforms(sim_raw_img)

            real_img_seq = torch.cat((real_img_seq, real_img.view(1, 3, 256, 256)))
            sim_img_seq = torch.cat((sim_img_seq, sim_img.view(1, 3, 256, 256)))

        return real_img_seq.permute((1, 0, 2, 3)), sim_img_seq.permute((1, 0, 2, 3))

    def create_sequence_list(self, img_name):
        img_name_split = img_name.split('_')
        obj_name = img_name_split[0]
        # cnt_num = img_name_split[1]
        depth = int(img_name_split[2])
        dx = int(img_name_split[4])
        dy = int(img_name_split[6])

        sequence_list = []

        new_img_name_split = img_name_split
        new_img_name_split[4] = '0'
        new_img_name_split[6] = '0'
        for i in range(1, depth + 1):
            new_img_name_split[2] = str(i)
            sequence_list.append('_'.join(new_img_name_split) + ".jpg")

        if dx == 0 and dy != 0:
            idx = 6
        elif dy == 0 and dx != 0:
            idx = 4
        else:
            sequence_list.append(img_name + ".jpg")
            return sequence_list

        for j in range(1, dx + dy):
            new_img_name_split[idx] = str(j)
            sequence_list.append('_'.join(new_img_name_split) + ".jpg")

        sequence_list.append(img_name + ".jpg")

        return sequence_list
