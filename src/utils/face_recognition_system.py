from src.utils.face_recognition import FaceClassifier, best_system_params_for_size, evaluate_model
import numpy as np
from collections import Counter
from src.utils.dataset import get_dataset


class FaceClassifierSystem(object):
    def __init__(self, method_parameters):
        self.classifiers = {}
        if method_parameters == 'best':
            method_parameters = best_system_params_for_size[7]
        for method, param in method_parameters.items():
            self.classifiers[method] = FaceClassifier(method, param)

    def fit(self, data, labels):
        for clf in self.classifiers.values():
            clf.fit(data, labels)

    def predict(self, data, full_report=False):
        if full_report:
            system_result = []
            results = []
            for clf in self.classifiers.values():
                results.append(clf.predict(data, nearest_image=True))

            for j in range(len(results[0])):
                votes = []
                for i in range(len(results)):
                    v = results[i][j]
                    votes.append(v['label'])
                major_vote = Counter(votes).most_common()[0][0]
                system_result.append({'method': 'system',
                                      'label': major_vote,
                                      'image': data[j], })
            results.append(system_result)
            return results
        else:
            results = []
            for clf in self.classifiers.values():
                results.append(clf.predict(data))

            pred = []
            results = np.array(results)
            for i in range(results.shape[1]):
                votes = results[:, i]
                major_vote = Counter(votes).most_common()[0][0]
                pred.append(major_vote)
            return np.array(pred)


if __name__ == "__main__":
    for size in range(1, 10):
        params = best_system_params_for_size[size]
        clf = FaceClassifierSystem(params)
        data, labels = get_dataset()

        scores = evaluate_model(model=clf, train_size_per_class=size, data=data, labels=labels)
        print("Size: {}, score: {}".format(size, scores.mean()))
        # Size: 1, score: 0.7046070460704607
        # Size: 2, score: 0.8567073170731707
        # Size: 3, score: 0.9198606271777003
        # Size: 4, score: 0.943089430894309
        # Size: 5, score: 0.9853658536585366
        # Size: 6, score: 0.9939024390243902
        # Size: 7, score: 1.0
        # Size: 8, score: 1.0
        # Size: 9, score: 1.0
