from datasets import get_datasets
from config_utils import load_config
import torch
import torchvision


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.001):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RepeatSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, samp, repeat):
        self.samp = samp
        self.repeat = repeat

    def __iter__(self):
        for i in self.samp:
            for j in range(self.repeat):
                yield i

    def __len__(self):
        return self.repeat * len(self.samp)


def get_data(dataset, data_loc, batch_size):
    train_data, valid_data, xshape, class_num = get_datasets(dataset, data_loc)
    train_data.transform.transforms = train_data.transform.transforms[:]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True)
    return train_loader
