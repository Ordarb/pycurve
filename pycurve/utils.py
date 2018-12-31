import pickle
import numpy as np
from scipy.interpolate import splint


def save_pickle(path, obj):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


class Transformation(object):

    @staticmethod
    def zero2par(self, zero, ttm):
        # todo: formula hast to be checked. not working at the moment
        discountrate = (1. / (1 + zero)) ** ttm
        par = (1 - discountrate) / np.cumsum(discountrate)
        return par

    @staticmethod
    def fwd2zero(ttm, splineObj):
        zero = [splint(0, t, splineObj) / t for t in ttm]
        return zero

    @staticmethod
    def zero2discount(zeros, ttm):
        return np.exp(-(zeros * ttm))
