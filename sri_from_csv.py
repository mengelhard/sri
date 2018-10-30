#!/usr/bin/env python

"""sri_from_csv.py: Calculate the Sleep Regularity Index from sleep/wake data

usage: sri_from_csv.py [-h] [-e [EPOCHS]] [-c [COLUMN]] fn
example: sri_from_csv.py -e 2880 test.csv

Calculate the Sleep Regularity Index from sleep/wake data

positional arguments:
  fn                    csv file where sleep data is located (in column 'sleepcol')

optional arguments:
  -h, --help            show this help message and exit
  -e [EPOCHS], --epochs [EPOCHS]
                        number of epochs per day (default: 1440)
  -c [COLUMN], --column [COLUMN]
                        column name in line 1 of 'fn' (csv file) to be used as sleep values (default: sleep)"""

from __future__ import print_function

import argparse
import numpy as np
import pandas as pd

def sri(sleep,epochs_per_day=1440):
    '''Returns the approximate sleep regularity index, a measure of sleep consistency, for this activity'''
    sleep_mat = np.reshape(sleep,(epochs_per_day,-1),order='F')
    assert len(sleep_mat[0]) == len(sleep)/epochs_per_day
    sri = [match24(x) for x in sleep_mat if len(x)>1]
    return 200*np.mean(sri)-100

def match24(x):
    assert np.ndim(x)==1
    m = np.diff(x)
    m = (m+1)%2
    return sum(m)/len(m)

def main():

    parser = argparse.ArgumentParser(description='Calculate the Sleep Regularity Index from sleep/wake data')
    parser.add_argument('fn', type=str,
                        help='csv file where sleep data is located (in column \'sleepcol\')')
    parser.add_argument('-e', '--epochs', type=int, nargs='?', default=1440,
                        help='number of epochs per day (default: 1440)')
    parser.add_argument('-c', '--column', type=str, nargs='?', default='sleep',
                        help='column name in line 1 of \'fn\' (csv file) to be used as sleep values (default: sleep)')
    args = parser.parse_args()
    
    sleep = pd.read_csv(args.fn)[args.column].values
    assert len(sleep)%args.epochs == 0
    print('Processed %i days of data' % (len(sleep)//args.epochs))
    print('Calculated SRI (with %i epochs per day) is %.2f' % (args.epochs,sri(sleep,epochs_per_day=args.epochs)))
    
if __name__ == "__main__":
    main()
    
__author__      = "Matthew Engelhard"
__copyright__   = "Copyright 2018, Matthew Engelhard"
__license__ = "MIT"
__email__ = "m.engelhard@duke.edu"