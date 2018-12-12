import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torch.utils.model_zoo as model_zoo

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from matplotlib import pyplot as plt
import numpy as np

import cv2, time

from PIL import Image

men_classes = ['jackets-coats', 'jeans', 'pants', 'shirts', 'shorts', 'suits-blazers', 'sweaters']
women_classes = ['dresses', 'jackets-coats', 'jeans', 'jumpsuits', 'pants', 'shorts', 'skirts', 'sweaters', 'tops']

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        #self.show_feature_map(x)
        x = self.layer2(x)
        #self.show_feature_map(x)
        x = self.layer3(x)
        #self.show_feature_map(x)
        x = self.layer4(x)
        #print(x.size())   # 2048*8*8
        #self.show_feature_map(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def show_feature_map(self, feature):
        current_feature = feature[0]
        current_feature_map = feature[0, 0, :, :]
        for j in range(1, feature[0].size(0)):
            current_feature_map = torch.add(current_feature_map, feature[0,j,:,:])
        current_feature_map_np = current_feature_map.detach().numpy()
        plt.imshow(current_feature_map_np)
        plt.show()

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("=> using pre-trained model '{}'".format('resnet_50'))
        pretrained_state = model_zoo.load_url(model_urls['resnet50'])
        model_state = model.state_dict()
        pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        print("=> using pre-trained model '{}'".format('resnet_101'))
        pretrained_state = model_zoo.load_url(model_urls['resnet101'])
        model_state = model.state_dict()
        pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    return model

def men_pred(clothes_path):
    tm = time.time()
    data_transforms = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(227),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    test_img_path = clothes_path
    im = Image.open(test_img_path)
    im_input = data_transforms(im).float()
    rand_tensor = torch.rand(3, 227, 227)
    #imshow(im_input); plt.show()
    inputs = torch.stack([im_input, rand_tensor.float(), rand_tensor.float(), rand_tensor.float()])

    #net = resnet50(num_classes = 7)
    net = resnet101(num_classes = len(men_classes))
    net.load_state_dict(torch.load('101_train/fas_resnet101_men_100000.pt',  map_location='cpu'))

    outputs = net(inputs)

    _, predicted = torch.max(outputs.data, 1)

    #print(men_classes[predicted[0]])
    #print(time.time()-tm)
    return men_classes[predicted[0]]

def women_pred(clothes_path):
    tm = time.time()
    data_transforms = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(227),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    test_img_path = clothes_path
    im = Image.open(test_img_path)
    im_input = data_transforms(im).float()
    #imshow(im_input); plt.show()
    rand_tensor = torch.rand(3, 227, 227)
    inputs = torch.stack([im_input, rand_tensor.float(), rand_tensor.float(), rand_tensor.float()])

    #net = resnet50(num_classes = 7)
    net = resnet101(num_classes = len(women_classes))
    net.load_state_dict(torch.load('101_train/fas_resnet101_women_100000.pt',  map_location='cpu'))

    outputs = net(inputs)

    _, predicted = torch.max(outputs.data, 1)

    #print(women_classes[predicted[0]])
    #print(time.time()-tm)
    return women_classes[predicted[0]]


if __name__ == '__main__':
    clothes_path = '/Users/evnw/documents/github/fashionnet-pytorch/resnet/test.jpg'