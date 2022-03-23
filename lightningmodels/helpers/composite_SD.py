#!/usr/bin/python3
'''
composite_sd.py defines three functions:

  composite_SD is a function to calculate the overall standard deviation of
    a collection of sample groups, given only the group standard deviations,
    means, and sample counts.

  sample_SD is a simple function to calculate the standard deviation of a
    collection of samples.

  avg (to calculate the simple average) is also defined.

These functions are compatible with Python 2.6, Python 2.7, and Python 3.*.

Copyright 2010, 2013, by David A. Burton
Cary, NC  USA
+1 919-481-0149
Email: http://www.burtonsys.com/email/
Permission is hereby granted to use this function for any purpose,
with or without modification, provided only that this copyright &
permission notice is retained.

Note: if what you really want is a "running" calculation of variance and/or
standard deviation, see this article about the Welford Method:
http://www.johndcook.com/standard_deviation.html

'''

from __future__ import print_function, division  # requires python 2.6 or later (2.7 or later preferred)
import math

__all__ = ['iterable', 'avg', 'sample_SD', 'composite_SD']


def iterable(obj):
    '''True iff obj is iterable: a list, tuple, or string.'''
    return hasattr(obj, '__contains__')


def avg(samples):
    if len(samples) >= 1:
        return sum(samples) / len(samples)
    return float('nan')


def sample_SD(samples):
    '''input is an array of samples; result is the standard deviation'''
    mean = avg(samples)
    sum_of_squared_deviations = 0
    sd = 0
    if len(samples) >= 2:
        for datum in samples:
            sum_of_squared_deviations += ((datum - mean) * (datum - mean))
        sd = math.sqrt(sum_of_squared_deviations / (len(samples) - 1))
    return sd


def composite_SD(means, SDs, ncounts):
    '''Calculate combined standard deviation via ANOVA (ANalysis Of VAriance)
       See:  http://www.burtonsys.com/climate/composite_standard_deviations.html
       Inputs are:
         means, the array of group means
         SDs, the array of group standard deviations
         ncounts, number of samples in each group (can be scalar
                  if all groups have same number of samples)
       Result is the overall standard deviation.
    '''
    G = len(means)  # number of groups
    if G != len(SDs):
        raise Exception('inconsistent list lengths')
    if not iterable(ncounts):
        ncounts = [ncounts] * G  # convert scalar ncounts to array
    elif G != len(ncounts):
        raise Exception('wrong ncounts list length')

    # calculate total number of samples, N, and grand mean, GM
    N = sum(ncounts)  # total number of samples
    if N <= 1:
        raise Exception("Warning: only " + str(N) +
                        " samples, SD is incalculable")
    GM = 0.0
    for i in range(G):
        GM += means[i] * ncounts[i]
    GM /= N  # grand mean

    # calculate Error Sum of Squares
    ESS = 0.0
    for i in range(G):
        ESS += ((SDs[i])**2) * (ncounts[i] - 1)

    # calculate Total Group Sum of Squares
    TGSS = 0.0
    for i in range(G):
        TGSS += ((means[i] - GM)**2) * ncounts[i]

    # calculate standard deviation as square root of grand variance
    result = math.sqrt((ESS + TGSS) / (N - 1))
    return result
