import os
import string
import random
from datetime import datetime

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np

def get_datetime_key():
    """ Get a string key based on current datetime. """
    return 'D' + datetime.now().strftime("%Y_%m_%dT%H_%M_%S_") + get_random_string(4)


def get_random_string(length):
    letters = string.ascii_uppercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def create_if_noexists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def stratify_dataset(dataset, test_size, random_seed):

    user_array = np.array([traj["user_id"].iloc[0] for traj in dataset])

    indices = np.arange(len(dataset))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, stratify=user_array, random_state=random_seed)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset