##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, torch
import os.path as osp
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image

from .DownsampledImageNet import ImageNet16
from .SearchDatasetWrap import SearchDataset
from config_utils import load_config

Dataset2Class = {'cifar10': 10,
                 'cifar100': 100,
                 'fake': 10,
                 'imagenet-1k-s': 1000,
                 'imagenette2': 10,
                 'imagenet-1k': 1000,
                 'ImageNet16': 1000,
                 'ImageNet16-150': 150,
                 'ImageNet16-120': 120,
                 'ImageNet16-200': 200}


class CUTOUT(object):

    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return ('{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_datasets(name, root):
    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == 'cifar10':
        lists = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == 'cifar10':
        train_data = dset.CIFAR10(root, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(root, train=False, transform=test_transform, download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]

    return train_data, test_data, xshape, class_num


def get_nas_search_loaders(train_data, valid_data, dataset, config_root, batch_size, workers):
    if isinstance(batch_size, (list, tuple)):
        batch, test_batch = batch_size
    else:
        batch, test_batch = batch_size, batch_size
    if dataset == 'cifar10':
        # split_Fpath = 'configs/nas-benchmark/cifar-split.txt'
        cifar_split = load_config('{:}/cifar-split.txt'.format(config_root), None, None)
        train_split, valid_split = cifar_split.train, cifar_split.valid  # search over the proposed training and validation set
        # logger.log('Load split file from {:}'.format(split_Fpath))      # they are two disjoint groups in the original CIFAR-10 training set
        # To split data
        xvalid_data = deepcopy(train_data)
        if hasattr(xvalid_data, 'transforms'):  # to avoid a print issue
            xvalid_data.transforms = valid_data.transform
        xvalid_data.transform = deepcopy(valid_data.transform)
        search_data = SearchDataset(dataset, train_data, train_split, valid_split)
        # data loader
        search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True, num_workers=workers,
                                                    pin_memory=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
                                                   num_workers=workers, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(xvalid_data, batch_size=test_batch,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
                                                   num_workers=workers, pin_memory=True)
    elif dataset == 'cifar100':
        cifar100_test_split = load_config('{:}/cifar100-test-split.txt'.format(config_root), None, None)
        search_train_data = train_data
        search_valid_data = deepcopy(valid_data);
        search_valid_data.transform = train_data.transform
        search_data = SearchDataset(dataset, [search_train_data, search_valid_data],
                                    list(range(len(search_train_data))), cifar100_test_split.xvalid)
        search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True, num_workers=workers,
                                                    pin_memory=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=workers,
                                                   pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=test_batch,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       cifar100_test_split.xvalid), num_workers=workers,
                                                   pin_memory=True)
    elif dataset == 'ImageNet16-120':
        imagenet_test_split = load_config('{:}/imagenet-16-120-test-split.txt'.format(config_root), None, None)
        search_train_data = train_data
        search_valid_data = deepcopy(valid_data);
        search_valid_data.transform = train_data.transform
        search_data = SearchDataset(dataset, [search_train_data, search_valid_data],
                                    list(range(len(search_train_data))), imagenet_test_split.xvalid)
        search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True, num_workers=workers,
                                                    pin_memory=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=workers,
                                                   pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=test_batch,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       imagenet_test_split.xvalid), num_workers=workers,
                                                   pin_memory=True)
    else:
        raise ValueError('invalid dataset : {:}'.format(dataset))
    return search_loader, train_loader, valid_loader

# if __name__ == '__main__':
#  train_data, test_data, xshape, class_num = dataset = get_datasets('cifar10', '/data02/dongxuanyi/.torch/cifar.python/', -1)
#  import pdb; pdb.set_trace()
