import numpy as np
from scipy.fftpack import dct
from src.utils.dataset import get_dataset, train_test_split


class FaceClassifier(object):
    def __init__(self, method, **method_kwargs):
        self.train = None
        self.labels = None
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
            feature = self.method(image, **self.method_kwargs)
            self.train.append(feature)

    def predict(self, data):
        pred = []
        for image in data:
            pred.append(self._predict_one(image))
        return pred

    def _predict_one(self, image):
        assert self.train and self.labels

        feature = self.method(image, **self.method_kwargs)

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
    def _gradient(img, smooth=16):
        h = img.shape[0]
        x = img.copy().astype(np.int32)
        grads = []
        for curr in range(smooth, h - smooth):
            top = x[curr - smooth:curr, :]
            bottom = np.flip(x[curr:curr + smooth, :], axis=1)
            grads.append(FaceClassifier._distance(top, bottom))
        return np.array(grads)

    @staticmethod
    def _scale(img, scale=2):
        h, w = img.shape[0], img.shape[1]
        m, n = h // scale, w // scale
        x = img.copy().astype(np.int32)
        img_sc = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                img_sc[i, j] = np.sum(x[i * scale:min((i + 1) * scale, h), j * scale:min((j + 1) * scale, w)])
        return np.asarray(img_sc).reshape(-1)


def grid_search(method, params):
    pass


if __name__ == '__main__':
    data, labels = get_dataset()
    train_X, train_y, test_X, test_y = train_test_split(data, labels, train_size_per_class=6)

    clf = FaceClassifier(method='scale', scale=8)
    clf.fit(train_X, train_y)

    pred = clf.predict(test_X)

    print(test_y)
    print(pred)
    correct = 0
    for test, pr in zip(test_y, pred):
        correct += (test == pr)
    print('Accuracy on test: {}'.format(100 * correct / len(test_y)))
