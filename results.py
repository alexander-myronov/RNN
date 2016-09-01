from functools import partial
import sys

__author__ = 'amyronov'

import pandas as pd
import numpy as np

def transform(str_array, array_mapper):
    if isinstance(str_array, float):
        if np.isnan(str_array):
            return []
        return [str_array]
    if isinstance(str_array, str) and str_array[0] != '[':
        return str_array
    try:
        arr= np.fromstring(str_array.replace('[', '').replace(']',''), sep=' ')
        return array_mapper(arr)

    except:
        return str_array

def mean_std_tuple(arr):
    return arr.mean(), arr.std()

def str_mean_plus_minus_std(arr):
    return '%.3lf +- %.3lf' % (arr.mean(), arr.std())

if __name__ == '__main__':

    results_file = sys.argv[1]

    df = pd.read_csv(results_file)




    df_res = df.applymap(partial(transform, array_mapper=str_mean_plus_minus_std))

    df_res.to_csv(r'data/result_processed.csv')
