
from sklearn import datasets
from sklearn import preprocessing

import numpy as np
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Sample_Dataloader(Dataset):

    """
    Dataloader class used to load in data in an image folder.
    Made it so that it performs a fixed set of transformations to a pair of images in different folders
    """

    def __init__(self, dataset):
        '''

        :param dataset: A list containing x and y variables
        '''

        self.x = dataset[0]
        self.y = dataset[1]

    def __len__(self):

        return len(self.x)

    def __getitem__(self, index):

        x_sample = self.x[index]
        y_sample = self.y[index]

        return torch.tensor(x_sample, dtype=torch.float32), torch.tensor(y_sample, dtype=torch.long)

def generator(input_size = 2, num_samples= 100, num_classes=2):
    '''

    :param input_size:
    :param num_samples:
    :return: a tuple that contains x, y and coef.
    '''

    dataset = datasets.make_classification(n_samples=num_samples, n_features=input_size, n_classes=num_classes)

    return dataset