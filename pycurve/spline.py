import sys
import numpy as np
from scipy.optimize import fmin
from scipy.interpolate import spalde, splint, splev, splrep
from scipy import integrate
from .bond import Bond
from .instruments import meta_data
import time
from .curve import Transformation


class SmoothingSpline(object):

    def __init__(self, instruments, pay_freq=1):
        self.instruments = None
        self.instruments = meta_data(instruments, payment_frequency=pay_freq).sort_values(by='ttm')

    def fit(self, algorithm='VRP', target='price', w_method='duration', w_short=False,
                        knots=[2, 10, 20]):
        '''
        Spline-based yield curve technic using a piecewise cubic polynominal with the segments joined
        at so-called knot points. Allows for a much higher degree of flexibility than the parametric
        approaches. To achieve a sufficient degree of smoothing, the objective function is extended
        by a penalty function (e.g. VRP) on the forward rate to control the trade-off between
        goodness-of-fit (flexibility) and the smoothness of the curve.
        ------------------
        INPUT   algorithm:      'VRP' (default) or 'None' -> selection of penalty term
                target:         'price' (default) -> 'ytm' not implemented as no fumula found
                w_method:       'duration' or 'uniform' - importance of individual bonds in optimization
                w_short:        Bolean - special weight to very short rate - False = Default
                knots:          [2,10,20] (Default) - separation points of piecewise polinominals
        ------------------
        RETURNS  params
        '''
        start = time.time()
        # 1) Initiate parameter for the smoothing spline
        self.algorithm = algorithm
        self._target = target
        if self._target == 'ytm':
            sys.exit('ytm cannot be chosen as input parameter as exact penalty formula not known.')
        self._w_method = w_method
        self._w_short = w_short
        if self.instruments.ttm.min() > knots[0]: knots[0] = np.ceil(self.instruments.ttm.min())
        if self.instruments.ttm.max() < knots[-1]: knots[-1] = np.floor(self.instruments.ttm.max())
        self.knots = np.array(knots)  # as used from the BoE
        self.order = 3
        y_init = np.array(self.instruments.ytm * 1.5)  # ytm as best guess for fwd rates
        ttm = np.array(self.instruments.ttm)
        obj = splrep(ttm, y_init, k=3, t=self.knots, task=-1)
        knots = obj[0]
        coeff = obj[1]
        order = obj[2]

        weights = self._weighting(w_method=self._w_method, w_short=self._w_short)

        # optimization
        start = time.time()
        coeff = fmin(self._optimizeSpline, coeff, args=(knots, order, weights,), maxiter=2000)
        self.out = coeff
        self.parameter = (knots, coeff, order)

        # todo: make them correct
        self.instruments['zero'] = Transformation().fwd2zero(self.instruments.ttm, self.parameter)
        # self.instruments['zero'] = self._fwd2zero(self.instruments.ttm,self.parameter)
        self.instruments['fwd'] = splev(self.instruments.ttm, self.parameter)
        self._runningTime = time.time() - start
        return self.parameter

    def _optimizeSpline(self, coeff, knots, order, weights):
        '''
        Smoothed Spline optimization method. Traditionally there is a penalty function used in
        the interpolation for better smoothness. However, there are several possible ways to
        create such a penalty function.
        ------------------
        INPUT   algorithm:      'VRP' (default), 'FNZ', or None
                target:         'price' (default) or 'ytm'
                splineObj:      (knots,coeff,order)
                weights:        weighting of individual price errors
        ------------------
        RETURNS  params
        '''
        t_min = self.instruments.ttm.min()
        pxObs = np.array(self.instruments.px_dirty)
        ytmObs = np.array(self.instruments.ytm)
        pxFit = list()
        ytmFit = list()
        penalty = list()

        for i in range(len(self.instruments)):

            ttm = self.instruments.ttm.iloc[i]
            cf = np.array(self.instruments.cf.iloc[i])
            t = np.array(self.instruments.timing.iloc[i])
            acc = np.array(self.instruments.accrued.iloc[i])
            freq = np.array(self.instruments.cpnFreq.iloc[i])
            coupon = self.instruments.coupon.iloc[i]

            fwd = splev(t, (knots, coeff, order))
            zero = Transformation().fwd2zero(t, (knots, coeff, order))

            discount = Transformation().zero2discount(zero, t)

            # ---------------------------------------------------------
            # 1) Find fitted values of target
            if self._target == 'price':
                pxFit.append(sum(discount * cf))
            elif self._target == 'ytm':
                pxFit = sum(discount * cf)
                ytmFit.append(Bond().ytm_short(pxFit, freq, cf, t, acc, px='dirty'))

                # ------------------------------------------------------------
            # 2) Calculation of penalty term as in Sleath and Anderson (2001)
            if self.algorithm == 'VRP':
                L, S, mu = 9.2, -1, 1
                f_penalty = lambda x: np.exp((L - (L - S) * np.exp(-x / mu))) * spalde(x, (knots, coeff, order))[2] ** 2
                pen = integrate.quad(f_penalty, t_min, ttm)[0]
            elif self.algorithm == None:
                pen = 0.
            else:
                # todo: other penalty-terms like FNZ
                sys.exit('Only VRP or no penalty term are currently implemented. Choose one of them')
            penalty.append(pen)

        # --------------------------------------------------------------------
        # 3) Objective function and error measure calculation
        if self._target == 'price':
            distance = weights * (pxFit - pxObs)
            self.instruments['pxFit'] = pxFit
            self.instruments['pxObs'] = pxObs
            self._RMSE = np.sqrt(np.mean((pxFit - pxObs) ** 2))
            self._MAE = np.mean(np.abs(pxFit - pxObs))

        elif self._target == 'ytm':
            distance = weights * (ytmFit - ytmObs) * 100
            self.instruments['ytmFit'] = ytmFit
            self.instruments['ytmObs'] = ytmObs
            self._RMSE = np.sqrt(np.mean((ytmFit - ytmObs) ** 2))
            self._MAE = np.mean(np.abs(ytmFit - ytmObs))
        # --------------------------------------------------------------------
        error = sum(distance ** 2) + sum(penalty)  # final error measure for optimization
        return error

    def _weighting(self, w_method='duration', w_short=True):
        '''
        Weighting of the errors of individual bonds is crucial to solve several problems.
        Changing weights in the optimization can also address a potential heteroscedasticity
        problem. Implmented are uniform and inverse duration weighting, but theoretically,
        it can also be weighted by bid-ask spread, amount outstanding, closeness to par, etc.
        ------------------
        INPUT    bonds:         df of all bond characteristics
                 method:        'uniform' or (inverse) 'duration'
                 weight_short:  boolean, give the very first short rate more weight
        ------------------
        RETURNS  weights
        '''
        if w_method == 'uniform':
            weights = np.ones(len(self.instruments))
        elif w_method == 'duration':
            weights = np.array(1. / self.instruments.duration)
        if w_short == True:         weights[0] = 100.
        return weights
