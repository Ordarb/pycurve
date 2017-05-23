
import sys
import pandas as pd
import numpy as np
from bond import Bond
from optimize import Optimization
import matplotlib.pyplot as plt


class Curve(Optimization):

    def __init__(self, instruments, payFreq = 1):
        '''
        Provide bond and other information in a DataFrame format. Necessary informations have
        to come in the form of 'date', 'maturity', 'coupon', and (clean) 'price' columns.
        ------------------
        INPUT    instrument:    df of fixed-income instrument infromation 
        ------------------
        '''
        self.instruments = instruments
        self._payFreq     = payFreq
        instruments = self._BondInfo(instruments)



    def _BondInfo(self, instruments):
        '''
        Fills all the necessary information of the bonds to make the model work.
        '''

        # 1) todo: check if necessary bond infos -> columns are available
        try: test = instruments[['date','maturity','coupon']]
        except ValueError: print('Input data do not have date - maturity - or - coupon infos.')

        # 2) todo: check wheter px_dirty or px_clean is given
        try:
            test = instruments['px_dirty']
            px_type, px_field = 'dirty', 'px_dirty'
        except:
            try:
                test = instruments['px_clean']
                px_type, px_field = 'clean', 'px_clean'
            except ValueError:
                sys.exit('Price info must be either px_dirty or px_clean.')

        # 3) complete all the necessary bond information
        ytm, modDuration = list(),list()
        px_dirty, px_clean = list(), list()
        cf, timing = list(), list()
        accrued = list()
        freq = list()

        for b in instruments.index:

            px              = instruments[px_field].loc[b]
            cpn             = instruments.coupon.loc[b] 
            ttm             = Bond().ttm(instruments.date,instruments.maturity)
            y, c, t, acc    = Bond().bond_ytm(px, cpn, ttm.loc[b], par=100, freq=self._payFreq, px=px_type)

            if px_type == 'dirty': px_dirty.append(px)
            else: px_dirty.append(px+acc)
            if px_type == 'clean': px_clean.append(px)
            else: px_clean.append(px-acc)

            cf.append(c)
            timing.append(t)
            accrued.append(acc)
            ytm.append(y)
            modDuration.append(Bond().duration(px_clean[-1], c, t, y))

        instruments['ytm']      = ytm 
        instruments['duration'] = modDuration
        instruments['ttm']      = ttm 
        instruments['px_dirty'] = px_dirty
        instruments['px_clean'] = px_clean
        instruments['cpnFreq']  = 1.  
        instruments['cf']       = cf
        instruments['timing']   = timing
        instruments['accrued'] = accrued
        return instruments
