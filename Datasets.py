import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import torchvision.transforms as transforms
import math

import os


class TrainingDataset(Dataset):
    def __init__(self, data_folder, person_ids, transform=None, num_sample_persons=4, num_sample_imgs=8, random_mask=False):
        assert os.path.isdir(data_folder)
        self.data_folder = data_folder
        self.person_ids = person_ids
        self.unread_ids = set(self.person_ids)
        assert len(self.person_ids) > 0
        self.person_img_path_dict = {}
        self.random_mask = random_mask
        count = 0
        for pid in self.person_ids:
            pfolder = os.path.join(self.data_folder, pid)
            assert os.path.isdir(pfolder)
            img_paths = [os.path.join(pfolder, x) for x in os.listdir(pfolder)]
            num_img_paths = len(img_paths)
            miss_num = num_sample_imgs - num_img_paths
            miss_img_paths = []
            for miss_i in range(miss_num):
                miss_img_paths.append(random.choice(img_paths))
            img_paths += miss_img_paths
            self.person_img_path_dict[pid] = img_paths
            count += len(self.person_img_path_dict[pid])
        self.num_sample_persons = num_sample_persons
        self.num_sample_imgs = num_sample_imgs
        self.transform = transform
        #self.length = math.ceil(1.*len(person_ids)/num_sample_persons)
        self.length = int(1.*len(person_ids)/num_sample_persons)
        if self.random_mask:
            self.random_mask_obj = RandomErasing(random_fill=True)

    def __getitem__(self, index):
        person_samples = random.sample(self.unread_ids, self.num_sample_persons)
        self.unread_ids = self.unread_ids - set(person_samples)

        if len(self.unread_ids) < self.num_sample_persons:
            self.unread_ids = set(self.person_ids)

        imgs_mini_batch_list = []
        for pid in person_samples:
            img_samples = [Image.open(x).convert('RGB')
                           for x in random.sample(self.person_img_path_dict[pid], self.num_sample_imgs)]
            if self.transform is not None:
                img_samples = [self.transform(x) for x in img_samples]
            else:
                img_samples = [transforms.ToTensor(x) for x in img_samples]
            if self.random_mask:
                for pimg_id in range(len(img_samples)):
                    img_samples[pimg_id] = self.random_mask_obj(img_samples[pimg_id])

            imgs_mini_batch_list += img_samples
        return torch.stack(imgs_mini_batch_list)

    def __len__(self):
        return self.length


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    random_fill: If ture, fill the erased area with random number. If false: fill with image net mean.
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.2, r1=0.3, mean=(0., 0., 0.), random_fill=False):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.random_fill=random_fill

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if not self.random_fill:
                    if img.size()[0] == 3:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    else:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                else:
                    if img.size()[0] == 3:
                        img[0, x1:x1 + h, y1:y1 + w] = torch.randn((h, w))
                        img[1, x1:x1 + h, y1:y1 + w] = torch.randn((h, w))
                        img[2, x1:x1 + h, y1:y1 + w] = torch.randn((h, w))
                    else:
                        img[0, x1:x1 + h, y1:y1 + w] = torch.rand((h, w))
                return img

        return img


class Market1501EvaluateDataset(Dataset):

    def __init__(self, folder, transform, tencrop):
        assert os.path.isdir(folder)
        self.image_paths = []
        self.tencrop =tencrop
        image_names = sorted([i for i in os.listdir(folder) if i[-3:] =='jpg'])
        for i in image_names:
            self.image_paths.append(os.path.join(folder, i))
        self.length = len(self.image_paths)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        if self.tencrop is None:
            return self.transform(img)
        else:
            imgs = self.tencrop(img)
            img_list = [self.transform(img) for img in imgs]
            return torch.stack(img_list)

    def __len__(self):
        return self.length


class MARSEvalDataset(Dataset):
    def __init__(self, folder, image_name_file, transform, tencrop=None):
        self.image_paths = []
        self.tencrop = tencrop
        image_names = open(image_name_file,'r').readlines()
        image_names = [x.strip() for x in image_names]
        for i in image_names:
            img_path = os.path.join(folder, i[:4], i)
            self.image_paths.append(img_path)
        self.length = len(self.image_paths)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        if self.tencrop is None:
            return self.transform(img)
        imgs = self.tencrop(img)
        img_list = [self.transform(img) for img in imgs]
        return torch.stack(img_list)

    def __len__(self):
        return self.length


class CUHK03EvaluateDataset(Dataset):

    def __init__(self, folder, transform, tencrop):
        assert os.path.isdir(folder)
        self.image_paths = []
        self.tencrop =tencrop
        image_names = [i for i in sorted(os.listdir(folder)) if i[-3:] =='png' or i[-3:] == 'jpg']
        image_names = sorted(image_names)
        for i in image_names:
            self.image_paths.append(os.path.join(folder, i))
        self.length = len(self.image_paths)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        if self.tencrop is None:
            return self.transform(img)
        else:
            imgs = self.tencrop(img)
            img_list = [self.transform(img) for img in imgs]
            return torch.stack(img_list)

    def __len__(self):
        return self.length