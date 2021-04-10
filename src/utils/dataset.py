import cv2
import numpy as np
from pathlib import Path
from copy import deepcopy


def get_train_test(train_size=9):
    assert 1 <= train_size <= 10

    dataset_path = Path().cwd().parent.parent.joinpath('data/dataset')
    files = dataset_path.glob('*')

    data = {}
    for file in files:
        label = file.name.split('.')[0].split('_')[-1]  # for name '398_40.png' label is '40'
        if label not in data:
            data[label] = []
        image = cv2.imread(str(file))
        data[label].append(image)

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for label, values in data.items():
        images = deepcopy(values)
        np.random.shuffle(images)
        train, test = images[:train_size], images[train_size:]
        train_x += train
        train_y += [label]*len(train)
        test_x += test
        test_y += [label]*len(test)
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    pass
