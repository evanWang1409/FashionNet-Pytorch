from __future__ import print_function, division
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class training_toolset:
    def __init__(self, csv_dir = None, img_dir = None):
        self.csv_dir = csv_dir
        self.img_dir = img_dir

    def initialize_dataset(self):
        if(self.csv_dir):
            csv_dir = self.csv_dir
        else:
            csv_dir = "/Users/evnw/Documents/Github/FashionNet-Pytorch/anno"
        if(self.img_dir):
            img_dir = self.img_dir
        else:
            img_dir = "/Users/evnw/Research/DeepFasion/attri_predict"
        dataset_tensor = initialize(csv_dir, img_dir, transforms.Compose([Rescale(256), CenterCrop(224), ToTensor()]))
        #dataset_arr = initialize(csv_dir, img_dir, transforms.Compose([Rescale(256), CenterCrop(224)]))
        return dataset_tensor#, dataset_arr

    def show_random_sample(self, dataset_arr, num):
        show_sample(dataset_arr, num)

class clothes_dataset(Dataset):

    def __init__(self, csv_dir, img_dir, transform=None):
        print(csv_dir)
        land_csv_file = os.path.join(csv_dir, 'landmarks.csv')
        attr_csv_file = os.path.join(csv_dir, 'attr.csv')
        cat_csv_file = os.path.join(csv_dir, 'cat.csv')

        land_frame = pd.read_csv(land_csv_file, sep = ',', header = 25)
        attr_frame = pd.read_csv(attr_csv_file, sep = ',', header = None)
        cat_frame = pd.read_csv(cat_csv_file, sep = ',', header = None)

        self.landmarks_frame = land_frame.iloc[:, [0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24]]
        self.visibility_frame = land_frame.iloc[:, [1,4,7,10,13,16,19,22]]
        self.attr_frame = attr_frame
        self.cat_frame = cat_frame
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = landmarks.values
        landmarks = landmarks.astype('float').reshape(-1, 2)

        landmarks = pd.DataFrame(landmarks)
        landmarks = landmarks.values

        visibility = self.visibility_frame.iloc[idx, :]
        visibility = visibility.values

        attr = self.attr_frame.iloc[idx, 1:]
        attr = attr.values
        cat_num = self.cat_frame.iloc[idx, 1]
        attr = attr.astype(np.uint8)
        cat = np.zeros(50, dtype = np.uint8)
        cat[cat_num-1] = 1
        sample = {'image': image,
                'landmarks': landmarks,
                'visibility': visibility,
                'attributes' : attr,
                'category' : cat}

        if self.transform:
            sample = self.transform(sample)

        return sample

def initialize(csv_dir, img_dir, transform):
    dataset = clothes_dataset(csv_dir, img_dir, transform)
    return dataset

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, visibility, attr, cat = sample['image'], sample['landmarks'], sample['visibility'], sample['attributes'], sample['category']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        landmarks = (landmarks * [new_w / w, new_h / h]).astype(np.uint8)

        return {'image': img, 'landmarks': landmarks, 'visibility': visibility, 'attributes' : attr, 'category' : cat}


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, visibility, attr, cat = sample['image'], sample['landmarks'], sample['visibility'], sample['attributes'], sample['category']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        min_x = min(landmarks[:, 0])
        min_y = min(landmarks[:, 1])

        max_x = max(landmarks[:, 0])
        max_y = max(landmarks[:, 1])

        if max_x - min_x >= 224 or max_y - min_y >= 224 or min_x == 0 or min_y == 0:
            if min_x == 0:
                left = 0
            else:
                left = np.random.randint(0, min(min_x, 256-224))
            if min_y == 0:
                top = 0
            else:
                top = np.random.randint(0, min(min_y, 256-224))

        else:
            print(min_x, max_x)
            left = np.random.randint(min(max(0, max_x-224), min_x), max(max(0, max_x-224), min_x))
            top = np.random.randint(min(max(0, max_y-224), min_y), max(max(0, max_y-224), min_y))

        image = image[top: top + new_h,
                        left: left + new_w]
        landmarks = landmarks - [left, top]

        for i in range(len(visibility)):
            if landmarks[i][0] > 224 or landmarks[i][1] > 224:
                #print('out')
                visibility[i] = 0

        return {'image': image, 'landmarks': landmarks, 'visibility': visibility, 'attributes' : attr, 'category' : cat}


class CenterCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, visibility, attr, cat = sample['image'], sample['landmarks'], sample['visibility'], sample['attributes'], sample['category']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        left = int(256/2-224/2)
        top = int(256/2-224/2)

        image = image[top: top + new_h,
                        left: left + new_w]
        landmarks = landmarks - [left, top]

        for i in range(len(visibility)):
            if landmarks[i][0] > 224 or landmarks[i][1] > 224:
                #print('out')
                visibility[i] = 0

        return {'image': image, 'landmarks': landmarks, 'visibility': visibility, 'attributes' : attr, 'category' : cat}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks, visibility, attr, cat = sample['image'], sample['landmarks'], sample['visibility'], sample['attributes'], sample['category']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)
        landmarks = landmarks/100
        landmarks_tensor = torch.from_numpy(landmarks)
        visibility_tensor = torch.from_numpy(visibility)
        attr_tensor = torch.from_numpy(attr)
        cat_tensor = torch.from_numpy(cat)
        return {'image': image_tensor,
                'landmarks': landmarks_tensor,
                'visibility': visibility_tensor,
                'attributes': attr_tensor,
                'category': cat_tensor}


def show_sample(dataset, num):
    fig = plt.figure()

    index = []

    for i in range(num):
        index.append(np.random.randint(0, len(dataset)))

    for i in range(num):

        sample = dataset[index[i]]
        print(index[i])
        print(sample)

        #print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1, num, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        try:
            show_landmarks(**sample)
        except:
            print(sample)
            
        show_landmarks(sample['image'], sample['landmarks'], sample['visibility'])

    plt.show()

def show_landmarks(image, landmarks, visibility):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='g')
    for i in range(len(visibility)):
        if visibility[i] == 0:
            continue
        plt.scatter(landmarks[i, 0], landmarks[i, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == '__main__':
    csv_dir = "/Users/evnw/Documents/Github/FashionNet-Pytorch/anno"
    img_dir = "/Users/evnw/Research/DeepFasion/attri_predict"
    dataset = initialize(csv_dir, img_dir, transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
    dataset_arr = initialize(csv_dir, img_dir, transforms.Compose([Rescale(256), RandomCrop(224)]))
    show_sample(dataset_arr, 1)




