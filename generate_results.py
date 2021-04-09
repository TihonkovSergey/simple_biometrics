from pathlib import Path
import cv2
from utils.face_detection import *


def generate(result_path, func=template_matching, **kwargs):
    dataset_dir = Path().cwd().joinpath('dataset')
    if not dataset_dir.exists():
        raise FileNotFoundError('Dataset does not exist')

    images = dataset_dir.glob('*.png')

    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    for path in images:
        image = cv2.imread(str(path))
        image = func(image, **kwargs)
        cv2.imwrite(str(result_path.joinpath(path.name)), image)


def generate_face_detection_results():
    results_dir = Path().cwd().joinpath('results')

    results_dict = {
        'template_matching_face': {'func': template_matching, 'detect': 'face'},
        'template_matching_eyes': {'func': template_matching, 'detect': 'eyes'},
        'template_matching_eyes&nose': {'func': template_matching, 'detect': 'eyes&nose'},
        'template_matching_nose&mouse': {'func': template_matching, 'detect': 'nose&mouse'},

        'viola_jones_face': {'func': viola_jones, 'detect': 'face'},
        'viola_jones_eyes': {'func': viola_jones, 'detect': 'eyes'},
        'viola_jones_face&eyes': {'func': viola_jones, 'detect': 'face&eyes'},

        'symmetry_lines': {'func': symmetry_lines},
    }
    for name, kwg in results_dict.items():
        generate(results_dir.joinpath(name), **kwg)


if __name__ == '__main__':
    pass
