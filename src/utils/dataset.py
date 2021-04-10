import cv2
import numpy as np
from pathlib import Path
from copy import deepcopy
from collections import defaultdict


def get_dataset():
    dataset_path = Path().cwd().joinpath('data/dataset')
    files = dataset_path.glob('*')
    data = []
    labels = []
    for file in files:
        label = int(file.name.split('.')[0].split('_')[-1])  # for name '398_40.png' label is '40'
        image = cv2.imread(str(file))
        data.append(image)
        labels.append(label)
    return np.array(data), np.array(labels)


def train_test_split(data, labels, train_size_per_class, random_state=0):
    assert 1 <= train_size_per_class <= 10
    assert len(data) == len(labels)
    np.random.seed(random_state)

    prep_data = defaultdict(list)
    for image, label in zip(data, labels):
        prep_data[label].append(image)

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for label, values in prep_data.items():
        images = deepcopy(values)
        np.random.shuffle(images)
        train, test = images[:train_size_per_class], images[train_size_per_class:]
        train_x += train
        train_y += [label]*len(train)
        test_x += test
        test_y += [label]*len(test)
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


if __name__ == '__main__':
    pass
