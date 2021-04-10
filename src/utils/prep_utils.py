from sklearn.datasets import fetch_olivetti_faces
import cv2
from pathlib import Path
import shutil


def prepare_from_sklearn(dataset_path):
    DATASET_DIR = Path(dataset_path)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    data = fetch_olivetti_faces()
    images = data['images']

    for i, image in enumerate(images):
        name = '{}_{}.png'.format(i // 10, i % 10)
        cv2.imwrite(str(DATASET_DIR.joinpath(name)), image * 256)


def all_jpg2png(dataset_path):
    dataset_path = Path(dataset_path)
    jpgs = dataset_path.glob('*.jpg')

    for path in jpgs:
        img = cv2.imread(str(path))
        cv2.imwrite(str(dataset_path.joinpath('{}.png'.format(path.stem))), img)


def copy_files(from_path, to_path, pattern='*'):
    from_path = Path(from_path)
    if not from_path.exists():
        raise FileExistsError

    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)

    files = from_path.glob(pattern)
    for file in files:
        path = Path(file)
        shutil.copy(path, to_path.joinpath(path.name))


if __name__ == '__main__':
    pass
