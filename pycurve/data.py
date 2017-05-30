__author__ = 'Sandro Braun'

import sys
import numpy as np
import pandas as pd

try:
    from bonds import get_data
except:
    print('Bonds-module not available.')

try:
    import bb
    from bb import BSRCHInterface, BLPInterface
except:
    print('Bloomberg-module not available.')




class Data(object):

    def db_data(self,ticker,fx,datum='newest'):

        df = get_data(ticker,fx,'all')

        if not datum == 'newest':
            dt = datum
        else:
            dt = df.index.max()

        df = df[df.index == dt]
            
        df = df[['Datum','Maturity_Date','Coupon','Price']]
        df.columns = ['date','maturity','coupon','px_clean']
        df = df.dropna()
        df = df.sort('maturity')
        df = df.reset_index(drop=True)
        return df


    def bb_srch(self, ticker):
        isin = BSRCHInterface().bsrch(ticker)
        for i in isin:
            df = BLPInterface().referenceRequest(isin,['today_dt','maturity','coupon','px_last'])
            df.columns = ['date','maturity','coupon','px_clean']
            return df
