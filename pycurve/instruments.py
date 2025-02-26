import sys
from .bond import Bond


def meta_data(instruments, payment_frequency=1):
    """
    Enriches the data input with all the necessary Bond data for Curve interpolation.
    :param instruments: dataframe with coupon, maturity, date, and price
    :param payment_frequency: number of coupon payments per year
    :return: df
    """

    # 1) todo: check if necessary bond infos -> columns are available
    try:
        test = instruments[['date', 'maturity', 'coupon']]
    except ValueError:
        print('Input data do not have date - maturity - or - coupon infos.')

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
    ytm, modDuration = list(), list()
    px_dirty, px_clean = list(), list()
    cf, timing = list(), list()
    accrued = list()
    freq = list()

    for b in instruments.index:

        px = instruments[px_field].loc[b]
        cpn = instruments.coupon.loc[b]
        ttm = Bond().ttm(instruments.date, instruments.maturity)
        y, c, t, acc = Bond().bond_ytm(px, cpn, ttm.loc[b], par=100, freq=payment_frequency, px=px_type)

        if px_type == 'dirty':
            px_dirty.append(px)
        else:
            px_dirty.append(px + acc)
        if px_type == 'clean':
            px_clean.append(px)
        else:
            px_clean.append(px - acc)

        cf.append(c)
        timing.append(t)
        accrued.append(acc)
        ytm.append(y)
        modDuration.append(Bond().duration(px_clean[-1], c, t, y))

    data = instruments.copy()
    data['ytm'] = ytm
    data['duration'] = modDuration
    data['ttm'] = ttm
    data['px_dirty'] = px_dirty
    data['px_clean'] = px_clean
    data['cpnFreq'] = 1.
    data['cf'] = cf
    data['timing'] = timing
    data['accrued'] = accrued
    return data