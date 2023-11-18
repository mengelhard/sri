#!/usr/bin/env python

"""sri_from_csv.py: Calculate the Sleep Regularity Index (SRI) and Sleep Midpoint from sleep/wake data

usage: sri_from_csv.py [-h] [-e [EPOCHS_PER_DAY]] [-c [SLEEP_COLUMN]] [-p | --plot | --no-plot] filename

Calculate the Sleep Regularity Index (SRI) and Sleep Midpoint from sleep/wake data

positional arguments:
  filename              csv file where sleep data is located (in column 'sleep_column')

optional arguments:
  -h, --help            show this help message and exit
  -e [EPOCHS_PER_DAY], --epochs_per_day [EPOCHS_PER_DAY]
                        number of epochs per day
  -c [SLEEP_COLUMN], --sleep_column [SLEEP_COLUMN]
                        column name in line 1 of 'filename' to be used as sleep values (0 = wake, 1 = sleep)
  -p, --plot, --no-plot
                        plot sleep values (default: False)"""

from __future__ import print_function

import argparse
import warnings
import numpy as np
import pandas as pd


def calculate_sri(arr, epochs_per_day=1440):
    return 200 * np.nanmean(arr[:-epochs_per_day] == arr[epochs_per_day:]) - 100


def calculate_midpoint(arr, epochs_per_day=1440, start_epoch=0):
    '''Circular mean:
    Note that sleep==1 -> sleep, sleep==0 -> wake'''

    sleep_mat = np.reshape(arr, (-1, epochs_per_day))

    cosines = np.cos(np.arange(epochs_per_day) * 2 * np.pi / epochs_per_day)[None, :]
    sines = np.sin(np.arange(epochs_per_day) * 2 * np.pi / epochs_per_day)[None, :]

    tm = (
        epochs_per_day *
        np.arctan2(
            np.nansum(sines * sleep_mat),
            np.nansum(cosines * sleep_mat)
        )
        // (2 * np.pi)
    )

    return (tm + start_epoch) % epochs_per_day


def remove_trailing_epochs(arr, epochs_per_day=1440):

    extra_epochs = len(arr) % epochs_per_day

    if extra_epochs > 0:
        warnings.warn('Removing %i trailing epochs' % extra_epochs)
        return arr[:-extra_epochs]
    else:
        return arr.copy()


def plot_sleep(arr, epochs_per_day=2880, cmap='Greys', **kwargs):

    import matplotlib.pyplot as plt

    plt.matshow(
        arr.reshape((-1, epochs_per_day)),
        cmap=cmap,
        aspect=400,
        **kwargs
    )

    ticks = np.arange(13) * (epochs_per_day // 12)

    plt.xticks(ticks=ticks, labels=ticks, rotation='45')
    plt.xlabel('Epoch')
    plt.ylabel('Day')

    plt.show()


def main():

    parser = argparse.ArgumentParser(
        description='Calculate the Sleep Regularity Index (SRI) and Sleep Midpoint from sleep/wake data')
    parser.add_argument(
        'filename', type=str,
        help='csv file where sleep data is located (in column \'sleep_column\')')
    parser.add_argument(
        '-e', '--epochs_per_day', type=int, nargs='?', default=1440,
        help='number of epochs per day')
    parser.add_argument(
        '-c', '--sleep_column', type=str, nargs='?', default='sleep',
        help='column name in line 1 of \'filename\' to be used as sleep values (0 = wake, 1 = sleep)')
    parser.add_argument(
        '-p', '--plot', action=argparse.BooleanOptionalAction, default=False,
        help='plot sleep values')
    args = parser.parse_args()
    
    print()
    print('Calculating SRI values from the %s column of %s based on %i epochs per day' % (
        args.sleep_column, args.filename, args.epochs_per_day
    ))
    
    df = pd.read_csv(args.filename)

    sleep = df[args.sleep_column].values
    
    n_epochs = len(sleep)
    n_days = len(sleep) // args.epochs_per_day
    sleep = remove_trailing_epochs(sleep[:args.epochs_per_day * n_days])

    sri = calculate_sri(sleep, epochs_per_day=args.epochs_per_day)
    midpoint = calculate_midpoint(sleep, epochs_per_day=args.epochs_per_day)
    
    print()
    print('Found %i epochs and %i complete days' % (n_epochs, n_days))
    print('Found %i missing sleep values, which will be ignored' % np.sum(np.isnan(sleep)))
    print()
    print('The calculated SRI is %.1f' % sri)
    print('The calculated sleep midpoint is epoch %i' % midpoint)

    if args.plot:
        plot_sleep(sleep)

    
if __name__ == "__main__":
    main()
