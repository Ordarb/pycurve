import numpy as np
from scipy import optimize


class Bond(object):

    def __init__(self):
        self.modDuration = None
        self.coupons = None
        self.coupon_timing = None
        self.accrued = None
        self.ytm = None

    def ttm(self, colStart, colEnd):
        return (colEnd - colStart) / np.timedelta64(1, 'D') / 365

    def duration(self, px, cf, TimeFactor, ytm):
        """
        Calculates the modified duration of a bond given ...
        :param px:
        :param cf:
        :param TimeFactor:
        :param ytm:
        :return:
        """
        discount = 1 / ((1 + ytm) ** TimeFactor)
        D = sum(cf * TimeFactor * discount) / px
        md = D / (1 + ytm)
        self.modDuration = md
        return md

    def structure(self, cpn, ttm, par=100, freq=1):
        '''
        ======================================================================
        returns the following bond details:
            o cashflow vector
            o ttm in years of the cashflows as vector
            o accrued interest as float
        ======================================================================
        '''
        freq = float(freq)
        if int(ttm) == ttm:
            count = int(ttm * freq)
            accrued_time = 1 / freq
        else:
            count = int(int(ttm) * freq + 1)
            accrued_time = ttm - int(ttm)
        t = np.array(range(0, count)) / freq + accrued_time  # time in years
        c = np.ones(count) * cpn / 100 * par / freq
        c[-1] += par
        accrued = (1 / freq - t[0]) * cpn
        self.coupons = c
        self.coupon_timing = t
        self.accrued = accrued
        return c, t, accrued

    def bond_ytm(self, price, cpn, ttm, par=100, freq=1, px='clean'):
        '''
        ======================================================================
        cpn: coupon, in percentage points, i.e 5
        all inputs are supposed to be np.arrays
        ----------
        return ytm in decimals, i.e. 0.0125
        ======================================================================
        '''
        c, t, accrued = self.structure(cpn, ttm, par, freq)
        ytm = self.ytm_short(price, freq, c, t, accrued, px=px)
        self.ytm = ytm
        return ytm, c, t, accrued

    def ytm_short(self, price, freq, c, t, accrued, px='clean'):
        '''
        ======================================================================
        short function, if structure (t, c, ...) is already available
        ======================================================================
        '''
        if px == 'dirty':
            px_dirty = price
        elif px == 'clean':
            px_dirty = price + accrued

        ytm_func = lambda y: sum(c / (1 + y / freq) ** (freq * t)) - px_dirty
        initial = min(c[0] / 100, 0.02)

        ytm = optimize.newton(ytm_func, initial)
        self.ytm = ytm
        return ytm
