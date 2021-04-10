import numpy as np
from scipy.fftpack import dct
from src.utils.dataset import get_dataset, train_test_split


class FaceClassifier(object):
    def __init__(self, method, *method_args, **method_kwargs):
        self.train = None
        self.labels = None
        self.method_args = method_args
        self.method_kwargs = method_kwargs

        methods = {'histogram': self._histogram,
                   'dft': self._dft,
                   'dct': self._dct,
                   'gradient': self._gradient,
                   'scale': self._scale, }
        assert method in methods
        self.method = methods[method]

    def fit(self, data, labels):
        assert len(data) == len(labels)
        self.labels = labels
        self.train = []
        for image, label in zip(data, labels):
            feature = self.method(image, *self.method_args, **self.method_kwargs)
            self.train.append(feature)

    def predict(self, data):
        pred = []
        for image in data:
            pred.append(self._predict_one(image))
        return np.array(pred)

    def _predict_one(self, image):
        assert self.train is not None and self.labels is not None

        feature = self.method(image, *self.method_args, **self.method_kwargs)

        best_label = self.labels[0]
        min_dist = 1e15
        for x, label in zip(self.train, self.labels):
            dist = self._distance(feature, x)
            if dist < min_dist:
                min_dist = dist
                best_label = label
        return best_label

    @staticmethod
    def _distance(x1, x2):
        return np.linalg.norm(np.array(x1) - np.array(x2))

    @staticmethod
    def _histogram(image, n_bins=30, return_bins=False):
        hist, bins = np.histogram(image, bins=np.linspace(0, 255, n_bins))
        if return_bins:
            return hist, bins
        return hist

    @staticmethod
    def _dft(image, side=13):
        fourier = np.fft.fft2(image)
        fourier = fourier[:side, :side]
        return np.abs(fourier)

    @staticmethod
    def _dct(image, side=13):
        cos = dct(image, axis=1)
        cos = dct(cos, axis=0)
        cos = cos[:side, :side]
        return cos

    @staticmethod
    def _gradient(image, smooth=16):
        h = image.shape[0]
        x = image.copy().astype(np.int32)
        grads = []
        for curr in range(smooth, h - smooth):
            top = x[curr - smooth:curr, :]
            bottom = np.flip(x[curr:curr + smooth, :], axis=1)
            grads.append(FaceClassifier._distance(top, bottom))
        return np.array(grads)

    @staticmethod
    def _scale(image, scale=2):
        h, w = image.shape[0], image.shape[1]
        m, n = h // scale, w // scale
        x = image.copy().astype(np.int32)
        scaled_image = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                scaled_image[i, j] = np.sum(x[i * scale:min((i + 1) * scale, h), j * scale:min((j + 1) * scale, w)])
        return np.asarray(scaled_image).reshape(-1)


def search_best_param(method, start, stop, step, repeats=1):
    data, labels = get_dataset()
    results = []

    for size in range(1, 10):
        for param in range(start, stop+step, step):
            clf = FaceClassifier(method, param)
            scores = evaluate_model(data, labels, clf, size, repeats=repeats)

            results.append({'param': param, 'size': size, 'mean': np.mean(scores), 'std': np.std(scores)})

    best_params, best_score = {}, 0
    best_size = 10
    for res in results:
        if res['mean'] > best_score or (res['mean'] == best_score and res['size'] < best_size):
            best_score = res['mean']
            best_params = {'param': res['param'], 'size': res['size']}

    results = {
        'cv_results': results,
        'best_params': best_params,
        'best_score': best_score,
    }
    return results


def evaluate_model(data, labels, model, train_size_per_class, repeats=1):
    scores = []
    for seed in range(repeats):
        train_X, train_y, test_X, test_y = train_test_split(data, labels,
                                                            train_size_per_class=train_size_per_class,
                                                            random_state=seed)
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        correct = (pred == test_y).sum()
        scores.append(correct / len(pred))
    return np.array(scores)


def get_size_depend(method, param):
    result = [[], []]
    clf = FaceClassifier(method, param)
    data, labels = get_dataset()
    for size in range(1, 10):
        scores = evaluate_model(data, labels, clf, size, repeats=1)
        result[0].append(size)
        result[1].append(np.mean(scores))
    return np.array(result)


def get_param_depend(method, size, params):
    result = [[], []]
    data, labels = get_dataset()
    for param in params:
        clf = FaceClassifier(method, param)
        scores = evaluate_model(data, labels, clf, size, repeats=1)
        result[0].append(param)
        result[1].append(np.mean(scores))
    return np.array(result)


if __name__ == '__main__':
    pass
