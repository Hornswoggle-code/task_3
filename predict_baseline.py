import datetime
import os

import pandas as pd

import analyze_lib as al
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def predict_baseline(product_key, transactions):
    if not os.path.exists(f'baseline_{product_key}'):
        os.mkdir(f'baseline_{product_key}')

    product_transactions = al.sum_values_groupby_c(
        transactions[(transactions['ProductKey'] == product_key) & (transactions['DistributionChannel'] == 'Physical')],
        'TransactionDate',
        ['UnitVolume', 'ActualSales', 'SalesDiscount', 'RetailFullPrice', 'WeekendFlag']).reset_index()

    product_transactions['OnPromotion'] = product_transactions['SalesDiscount'] / product_transactions[
        'RetailFullPrice'] < -0.05

    product_transactions.loc[~product_transactions['OnPromotion'], 'ActualSales'] = \
        product_transactions.loc[~product_transactions['OnPromotion'], 'RetailFullPrice']

    product_transactions.loc[~product_transactions['OnPromotion'], 'SalesDiscount'] = 0

    product_transactions.sort_values(by='TransactionDate', inplace=True)

    def func(t, a, b, c, d, e, f, g, x, y, k1, k2, k3, k4,
             l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16,
             m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16,
             n1, n2, n3, n4, n5, n6, n7, n8):
        return a \
            + (x * t ** 3 + b * t ** 2 + c * t + d) * \
            (k1 * np.sin(2 * np.pi * t / 1096) + k2 * np.sin(4 * np.pi * t / 1096)
             + k3 * np.cos(2 * np.pi * t / 1096) + k4 * np.cos(4 * np.pi * t / 1096)
             + l1 * np.sin(2 * np.pi * t / 365) + l2 * np.sin(4 * np.pi * t / 365)
             + l3 * np.sin(6 * np.pi * t / 365) + l4 * np.sin(8 * np.pi * t / 365)
             + l5 * np.sin(10 * np.pi * t / 365) + l6 * np.sin(12 * np.pi * t / 365)
             + l7 * np.sin(14 * np.pi * t / 365) + l8 * np.sin(16 * np.pi * t / 365)
             + l9 * np.sin(18 * np.pi * t / 365) + l10 * np.sin(20 * np.pi * t / 365)
             + l11 * np.sin(22 * np.pi * t / 365) + l12 * np.sin(24 * np.pi * t / 365)
             + l13 * np.sin(26 * np.pi * t / 365) + l14 * np.sin(28 * np.pi * t / 365)
             + l15 * np.sin(30 * np.pi * t / 365) + l16 * np.sin(32 * np.pi * t / 365)
             + m1 * np.cos(2 * np.pi * t / 365) + m2 * np.cos(4 * np.pi * t / 365)
             + m3 * np.cos(6 * np.pi * t / 365) + m4 * np.cos(8 * np.pi * t / 365)
             + m5 * np.cos(10 * np.pi * t / 365) + m6 * np.cos(12 * np.pi * t / 365)
             + m7 * np.cos(14 * np.pi * t / 365) + m8 * np.cos(16 * np.pi * t / 365)
             + m9 * np.cos(18 * np.pi * t / 365) + m10 * np.cos(20 * np.pi * t / 365)
             + m11 * np.cos(22 * np.pi * t / 365) + m12 * np.cos(24 * np.pi * t / 365)
             + m13 * np.cos(26 * np.pi * t / 365) + m14 * np.cos(28 * np.pi * t / 365)
             + m15 * np.cos(30 * np.pi * t / 365) + m16 * np.cos(32 * np.pi * t / 365)) \
            + (y * t ** 3 + e * t ** 2 + f * t + g) * (n1 * np.sin(2 * np.pi * t / 7) + n2 * np.sin(4 * np.pi * t / 7)
                                          + n3 * np.sin(6 * np.pi * t / 7) + n4 * np.sin(8 * np.pi * t / 7)
                                          + n5 * np.cos(2 * np.pi * t / 7) + n6 * np.cos(4 * np.pi * t / 7)
                                          + n7 * np.cos(6 * np.pi * t / 7) + n8 * np.cos(8 * np.pi * t / 7))

    unpromoted_transactions = product_transactions[~product_transactions['OnPromotion']]
    promoted_transactions = product_transactions[product_transactions['OnPromotion']]

    xaxis = list(map(lambda x: (x - pd.Timestamp(2020, 1, 1)).days, unpromoted_transactions['TransactionDate'].values))
    popt, _ = curve_fit(func, xaxis, unpromoted_transactions['UnitVolume'].values)
    xaxis_extended = list(range(product_transactions.shape[0]))
    yaxis = list(map(lambda x: func(x, popt[0], popt[1], popt[2], popt[3], popt[4],
                                    popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11], popt[12],
                                    popt[13], popt[14], popt[15], popt[16], popt[17], popt[18], popt[19], popt[20],
                                    popt[21], popt[22], popt[23], popt[24], popt[25], popt[26], popt[27], popt[28],
                                    popt[29], popt[30], popt[31], popt[32], popt[33], popt[34], popt[35], popt[36],
                                    popt[37], popt[38], popt[39], popt[40], popt[41], popt[42], popt[43], popt[44],
                                    popt[45], popt[46], popt[47], popt[48], popt[49], popt[50], popt[51], popt[52]),
                     xaxis_extended))

    plt.figure(figsize=(12, 4))
    plt.plot(product_transactions['TransactionDate'].values, product_transactions['UnitVolume'].values)
    plt.plot(product_transactions['TransactionDate'].values, yaxis)
    plt.legend(['UnitVolume', 'Predicted baseline UnitVolume'])
    plt.title(f'Baseline vs. actual UnitVolume for ProductKey {product_key}')
    promotion_dates = promoted_transactions['TransactionDate'].values
    i = 0
    while i < len(promotion_dates):
        start = promotion_dates[i]
        while i < len(promotion_dates) - 1 and (promotion_dates[i+1] - promotion_dates[i]) / np.timedelta64(1, 'D') == 1.0:
            i += 1
        end = promotion_dates[i]
        i += 1
        plt.axvspan(start, end, color='green', alpha=0.2)

    plt.savefig(f'baseline_{product_key}/baseline_unit_volume_{product_key}.png')
    plt.cla()

    """
    al.sum_values_groupby(product_transactions, ['TransactionDate', 'OnPromotion'], 'UnitVolume').unstack() \
        .plot(title=f'Baseline volume for product key {product_key}')
    """


transactions = al.read_files()

predict_baseline(49340, transactions)
predict_baseline(49341, transactions)
predict_baseline(49333, transactions)
predict_baseline(49329, transactions)
