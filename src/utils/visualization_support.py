from src.utils.dataset import train_test_split, get_dataset
from src.utils.face_recognition_system import FaceClassifierSystem
from src.utils.face_recognition import best_system_params_for_size
from collections import defaultdict


class VisualizeConnector(object):
    def __init__(self, size):
        data, labels = get_dataset()
        train_X, train_y, test_X, test_y = train_test_split(data, labels,
                                                            train_size_per_class=size,
                                                            random_state=0)
        self.test_images = test_X
        self.test_labels = test_y
        self.system = FaceClassifierSystem(best_system_params_for_size[size])
        self.system.fit(train_X, train_y)

        self.test_results = None
        self.precision_lines = None
        self.head_index = 0

        self._prepare_test()

    def _prepare_test(self):
        self.test_results = []
        self.precision_lines = defaultdict(list)

        total = 0
        correct = defaultdict(int)
        for test_image, true_label in zip(self.test_images, self.test_labels):
            system_report = self.system.predict([test_image], full_report=True)
            result = {}
            total += 1
            for i in range(len(system_report)):
                res = system_report[i][0]

                method = res['method']
                image = res['image']
                is_correct = res['label'] == true_label
                result[method] = {
                    'image': image,
                    'is_correct': is_correct, }
                correct[method] += is_correct
                self.precision_lines[method].append(correct[method]/total)
            self.test_results.append(result)

    def _get_result_by_index(self, idx):
        lines = {}
        for key in self.precision_lines:
            lines[key] = self.precision_lines[key][:idx+1]
        return self.test_results[idx], lines

    def get_next(self):
        if self.head_index == len(self.test_images):
            return None, None
        self.head_index += 1
        return self._get_result_by_index(self.head_index - 1)

    def get_prev(self):
        if self.head_index == 0:
            return None, None
        self.head_index -= 1
        return self._get_result_by_index(self.head_index - 1)
