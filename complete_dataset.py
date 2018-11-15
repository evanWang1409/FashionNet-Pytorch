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
    def __init__(self, csv_file = None, img_dir = None):
        self.csv_file = csv_file
        self.img_dir = img_dir

    def initialize_dataset(self):
        if(self.csv_file):
            csv_file = self.csv_file
        else:
            csv_file = "/Users/evnw/Research/DeepFasion/attri_predict/landmarks_csv/landmarks.csv"
        if(self.img_dir):
            img_dir = self.img_dir
        else:
            img_dir = "/Users/evnw/Research/DeepFasion/attri_predict"
        dataset_tensor = initialize(csv_file, img_dir, transforms.Compose([Rescale(256), CenterCrop(224), ToTensor()]))
        dataset_arr = initialize(csv_file, img_dir, transforms.Compose([Rescale(256), CenterCrop(224)]))
        return dataset_tensor, dataset_arr

    def show_random_sample(self, dataset_arr, num):
        show_sample(dataset_arr, num)

class cloth_landmarks_dataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print(csv_file)
        frame = pd.read_csv(csv_file, sep = ',', header = 25)
        self.landmarks_frame = frame.iloc[:, [0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24]]
        self.visibility_frame = frame.iloc[:, [1,4,7,10,13,16,19,22]]
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
        '''
        landmarks_no_0 = []
        for landmark in landmarks:
            if landmark[0] != 0 and landmark[1] != 0:
                landmarks_no_0.append([landmark[0], landmark[1]])

        landmarks = pd.DataFrame(landmarks_no_0)
        '''
        landmarks = pd.DataFrame(landmarks)
        landmarks = landmarks.values

        visibility = self.visibility_frame.iloc[idx, :]
        visibility = visibility.values
        sample = {'image': image, 'landmarks': landmarks, 'visibility': visibility}

        if self.transform:
            sample = self.transform(sample)

        return sample

def initialize(csv_file, img_dir, transform):
    dataset = cloth_landmarks_dataset(csv_file, img_dir, transform)
    return dataset

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, visibility = sample['image'], sample['landmarks'], sample['visibility']

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

        return {'image': img, 'landmarks': landmarks, 'visibility': visibility}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, visibility = sample['image'], sample['landmarks'], sample['visibility']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        min_x = min(landmarks[:, 0])
        min_y = min(landmarks[:, 1])

        max_x = max(landmarks[:, 0])
        max_y = max(landmarks[:, 1])
        '''
        if (max_x - min_x) > 204 or (max_y - min_y) > 204:
            image = image[max(min_y - 10, 0): min(max_y + 10, h),
                        max(min_x - 10, 0): min(max_x + 10, w)]
            image = transform.resize(image, (new_h, new_w))
            landmarks = landmarks * [new_w / w, new_h / h]
            return {'image': image, 'landmarks': landmarks, 'visibility': visibility}

        new_start_x_max = max(0, min_x - 10)
        new_start_y_max = max(0, min_y - 10)

        new_start_x_min = max(max_x- new_w + 10, 0)
        new_start_y_min = max(max_y- new_h + 10, 0)

        #top = np.random.randint(0, h - new_h)
        #left = np.random.randint(0, w - new_w)

        if new_start_y_min >= new_start_y_max or new_start_x_min >= new_start_x_max:
            top = new_start_y_min
            left = new_start_x_min
        else:
            top = np.random.randint(new_start_y_min, new_start_y_max)
            left = np.random.randint(new_start_x_min, new_start_x_max)

        image = image[top: top + new_h,
                      left: left + new_w]
        
        print('top', top, top+new_h)
        print('left', left, left + new_w)
        '''

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

        return {'image': image, 'landmarks': landmarks, 'visibility': visibility}


class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, visibility = sample['image'], sample['landmarks'], sample['visibility']

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

        return {'image': image, 'landmarks': landmarks, 'visibility': visibility}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks, visibility = sample['image'], sample['landmarks'], sample['visibility']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)
        landmarks = landmarks/100
        landmarks_tensor = torch.from_numpy(landmarks)
        visibility_tensor = torch.from_numpy(visibility)
        return {'image': image_tensor,
                'landmarks': landmarks_tensor,
                'visibility': visibility_tensor}


def show_sample(dataset, num):
    fig = plt.figure()

    index = []

    for i in range(num):
        index.append(np.random.randint(0, len(dataset)))

    for i in range(num):

        sample = dataset[index[i]]
        print(index[i])

        #print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1, num, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        try:
            show_landmarks(**sample)
        except:
            print(sample)

    plt.show()

def show_landmarks(image, landmarks,visibility):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='g')
    for i in range(len(visibility)):
        if visibility[i] == 0:
            continue
        plt.scatter(landmarks[i, 0], landmarks[i, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == '__main__':
    csv_file = "/Users/evnw/Research/DeepFasion/attri_predict/landmarks_csv/landmarks.csv"
    img_dir = "/Users/evnw/Research/DeepFasion/attri_predict"
    dataset = initialize(csv_file, img_dir, transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
    dataset_arr = initialize(csv_file, img_dir, transforms.Compose([Rescale(256), RandomCrop(224)]))
    show_sample(dataset_arr, 4)




