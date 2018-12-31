import numpy as np
import pandas as pd
from scipy.interpolate import splint
import matplotlib.pyplot as plt
import seaborn as sns

from .parametric import Models


def nelson_siegel(ttm, params):

    z, fwd = Models().NelsonSiegel(params, ttm)
    df = pd.concat([z, fwd], 1)
    df.index = ttm
    df.columns = ['zero', 'fwd']
    return df


def svensson(ttm, params):
    z, fwd = Models().Svensson(params, ttm)
    df = pd.concat([z, fwd], 1)
    df.index = ttm
    df.columns = ['zero', 'fwd']
    return df


def svensson_adjusted(ttm, params):
    z, fwd = Models().SvenssonAdj(params, ttm)
    df = pd.concat([z, fwd], 1)
    df.index = ttm
    df.columns = ['zero', 'fwd']
    return df


def boerk_christensen(ttm, params):
    z, fwd = Models().BoerkChristensen(params, ttm)
    df = pd.concat([z, fwd], 1)
    df.index = ttm
    df.columns = ['zero', 'fwd']
    return df


class Charting(object):

    def _init__(self, obj):

        if not isinstance(obj, list):
            self.crv = [obj]
        else:
            self.crv = obj

    def draw(self, lz=(0, 20), y=None):

        clr = sns.color_palette("hls", len(self.crv))

        maxlz = lz[1]
        if y is None:
            ymin = 0.0
            ymax = 0.01
        else:
            ymin = y[0]
            ymax = y[1]

        plt.close('all')
        for i, o in enumerate(self.crv):

            if o.instruments.zero.max() < 0.0015:
                zero = o.instruments.zero * 100
                fwd = o.instruments.fwd * 100
            elif o.instruments.zero.max() > 0.15:
                zero = o.instruments.zero / 100
                fwd = o.instruments.fwd / 100
            else:
                zero = o.instruments.zero
                fwd = o.instruments.fwd

            maxlz = max(o.instruments.ttm.max(), maxlz)
            ymin = min(o.instruments.ytm.min(), zero.min(), fwd.min(), ymin)
            ymax = max(o.instruments.ytm.max(), zero.max(), fwd.max(), ymax)

            plt.scatter(o.instruments.ttm, o.instruments.ytm)
            plt.plot(o.instruments.ttm, zero, label='%s zero' % o.algorithm)
            plt.plot(o.instruments.ttm, fwd, label='%s fwd' % o.algorithm, linestyle='--')

        plt.legend(loc='best')
        plt.xlim((0, maxlz + 1))
        plt.ylim((ymin - 0.005, ymax + 0.005))
        plt.show()


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
