from scipy.signal import savgol_filter
# from peakutils import baseline
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import integrate
import helpers as hlp


# the 'deriv' parameter in savgol_filter() is used in conjuction with the
# `delta` parameter, which is the x-spacing of the data. so, if the data are
# not evenly spaced, the computed derivatives are not correct. the workaround
# is to either not use the differentiation capability of savgol_filter() and
# use the differentiate() function of this module or interpolate the data with
# evenly spaced x values and use the spacing between them for the 'delta'
# parameter.
def smooth(data, window_length, polyorder, deriv=0, mode='interp'):
    """Apply a Savitzky-Golay filter to smooth an array.

     This is a wrapper around scipy.signal.savgol_filter. Original function can
     be found here:
     https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.savgol_filter.html

    Arguments:
        data {numpy.ndarray} -- Data to be smoothed.

        window_length {int} -- Length of the smoothing window.

        polyorder {int} -- Order of the polynomial that is used for smoothing.
        It must be smaller than the window length.

    Keyword Arguments:
        deriv {int} -- Order of derivative to compute. (default: {0})

        mode {str} -- Must be ‘mirror’, ‘constant’, ‘nearest’, ‘wrap’ or
        ‘interp’. Determines the type of extension that is used for the padded
        signal to which the filter is applied. (default: {'interp'})
    """
    return savgol_filter(data, window_length=window_length,
                         polyorder=polyorder, deriv=deriv, mode=mode)


def differentiate(x, y, order=1):
    """Numerically calculate the derivative of an array.

    Arguments:
        x {array} -- x-axis array.

        y {array} -- The array whose derivative is to be calculated.

    Keyword Arguments:
        order {int} -- Order of derivative to compute. (default: {1})
    """
    for i in range(order):
        y = np.gradient(y, x, edge_order=2)
    return y


# def poly(data, deg=2, max_it=100, tol=0.001):
#     """Baseline estimation using an n-th order polynomial.

#     This is a wrapper around peakutils.baseline.baseline. Original function can
#     be found here:
#     https://peakutils.readthedocs.io/en/latest/reference.html#module-peakutils.baseline

#     Arguments:
#         data {numpy.ndarray} --  Data for which the baseline is to be estimated
#         using n-th order polynomial fitting.

#     Keyword Arguments:
#         deg {int} -- The degree of the polynomial. (default: {2})

#         max_iter {int} -- Maximum number of iterations for the polynomial
#         fitting to converge. (default: {100})

#         tol {float} -- Tolerance to use when comparing the difference between
#         the current fit coefficients and the ones from the last iteration. The
#         iteration procedure will stop when the difference between them is lower
#         than tol. (default: {0.001})

#     Returns:
#         numpy.ndarray -- Polynomial baseline estimation.
#     """
#     return baseline(data, deg=deg, max_it=max_it, tol=tol)


def snip(data, iterations, increasing=False):
    """SNIP implementation for 1-D data based on the M. Morháč algorithm [1].

    [1] Morháč M, Kliman J, Matoušek V, Veselský M, Turzo I. Background
    elimination methods for multidimensional coincidence γ-ray spectra. Nuclear
    Instruments and Methods in Physics Research Section A: Accelerators,
    Spectrometers, Detectors and Associated Equipment. 1997 Dec 11;401(1):113-
    32.

    Arguments:
        data {numpy.ndarray or pd.core.series.Series} -- Data for which the
        background is to be estimated using the SNIP algorithm.

        iterations {int} -- Number of iterations for the SNIP algorithm.

    Keyword Arguments:
        increasing {bool} -- Implementation of the SNIP algorithm using
        increasing or decreasing iteration window. (default: {False})

    Returns:
        numpy.ndarray -- SNIP-calculated background.
    """

    # check value of iterations
    if isinstance(iterations, int) is False or iterations < 0:
        raise ValueError(
            'The number of iterations must be a positive integer (int).')

    N = len(data)
    w = np.empty(N)  # working vector

    v = data.copy()  # use copy of data so the original remain intact
    # if data is a pandas series convert them to numpy array
    if isinstance(data, pd.core.series.Series):
        v = v.values

    # snip for increasing iteration window
    def snip_increasing(data, iterations):

        p = 1
        while p <= iterations:

            i = p
            while i < N - p:
                w[i] = min(v[i], (v[i - p] + v[i + p]) / 2)
                i += 1

            j = p
            while j < N - p:
                v[j] = w[j]
                j += 1

            p += 1

        return v

    # snip for decreasing iteration window
    def snip_decreasing(data, iterations):

        p = iterations
        while p > 0:

            i = p
            while i < N - p:
                w[i] = min(v[i], (v[i - p] + v[i + p]) / 2)
                i += 1

            j = p
            while j < N - p:
                v[j] = w[j]
                j += 1

            p -= 1

        return v

    if increasing:
        return snip_increasing(data, iterations)
    else:
        return snip_decreasing(data, iterations)


def get_index(x, value, closest=True):
    """Get the index of an array that corresponds to a given value.
    If closest is true, get the index of the value closest to the
    value entered.
    """
    if closest:
        index = np.abs(np.array(x) - value).argsort()[0]
    else:
        index = list(x).index(value)

    return index


def interpolate(x1, y1, x2, kind='cubic'):
    """Interpolate an array x1, y1 with an array x2.
    Return a tuple of the x1_new, y1 arrays.
    """
    # start_value = max(x1[0], x2[0])
    # stop_value = min(x1[-1], x2[-1])

    # x1_start_index = get_index(x1, start_value, closest=True)
    # x1_start_value = x1[x1_start_index]
    # x1_stop_index = get_index(x1, stop_value, closest=True)
    # x1_stop_value = x1[x1_stop_index]

    # x2_start_index = get_index(x2, start_value, closest=True)
    # x2_start_value = x2[x2_start_index]
    # x2_stop_index = get_index(x2, stop_value, closest=True)
    # x2_stop_value = x2[x2_stop_index]

    # # interpolation range needs to be smaller than x1 range
    # if x1_start_value > x2_start_value:
    #     x2_start_index = x2_start_index + 1
    #     x2_start_value = x2[x2_start_index]

    # if x1_stop_value < x2_stop_value:
    #     x2_stop_index = x2_stop_index - 1
    #     x2_stop_value = x2[x2_stop_index]

    f = interp1d(
        x1,
        y1,
        kind=kind
    )

    # x1_new = x2[x2_start_index:x2_stop_index]
    x1_new = interpolation_intersection(x1, x2)

    return f(x1_new)


def interpolation_intersection(x1, x2):
    """Intersect two arrays, x1 and x2, and return the x2 intersection for
    interpolation, i.e. x2 upper and lower values must lie within x1.
    """
    start_value = max(x1[0], x2[0])
    stop_value = min(x1[-1], x2[-1])

    x1_start_index = get_index(x1, start_value, closest=True)
    x1_start_value = x1[x1_start_index]
    x1_stop_index = get_index(x1, stop_value, closest=True)
    x1_stop_value = x1[x1_stop_index]

    x2_start_index = get_index(x2, start_value, closest=True)
    x2_start_value = x2[x2_start_index]
    x2_stop_index = get_index(x2, stop_value, closest=True)
    x2_stop_value = x2[x2_stop_index]

    # interpolation range needs to be smaller than x1 range
    if x1_start_value > x2_start_value:
        x2_start_index = x2_start_index + 1
        x2_start_value = x2[x2_start_index]

    if x1_stop_value < x2_stop_value:
        x2_stop_index = x2_stop_index - 1
        x2_stop_value = x2[x2_stop_index]

    return x2[x2_start_index:x2_stop_index + 1]


def norm_peak(y, x, peak, closest=True):
    """Normalize a y-array to the value of a peak given its x-array value.
    If 'peak' is an integer, the y-array is normalized to the
    value of y that corresponds to this x-array value.
    If 'peak' is a list or tuple that contains two values, the y-array is
    normalized to the maximum value of y between these x-array values.
    """
    # check if the x-array is sorted
    if not hlp.is_sorted(x, sort_order='both'):
        raise ValueError("Array 'x' is not sorted.")
    # check if y and x are of same length
    if len(x) != len(y):
        raise ValueError("Arrays 'x' and 'y' have different lengths.")

    if isinstance(peak, (list, tuple)) and len(peak) != 2:
        raise ValueError(
            "'peak' can either be an int/float or a 2-elements list/tuple.")
    elif isinstance(peak, (list, tuple)) and len(peak) == 2:
        start_index = get_index(x, peak[0], closest=closest)
        stop_index = get_index(x, peak[1], closest=closest)
        # swap indices if start_index > stop_index
        if start_index > stop_index:
            start_index, stop_index = stop_index, start_index
        value = max(y[start_index:stop_index + 1])
    elif isinstance(peak, (int, float)):
        peak_index = get_index(x, peak, closest=closest)
        value = y[peak_index]

    return y / value


def norm_area(y, x, x_range, closest=True):
    """Normalize an array y to the value of the integral between the specified
    range of the x array.
    """
    # check the sort order of x and make sure that the calculated integral
    # will have the correct sign for x either ascending and descending
    # (result of integrate.simps() has the opposite sign)
    if hlp.is_sorted(x, sort_order='ascending'):
        sort = 1
    elif hlp.is_sorted(x, sort_order='descending'):
        sort = -1
    else:
        raise ValueError("Array 'x' is not sorted.")

    # check if y and x are of same length
    if len(x) != len(y):
        raise ValueError("Arrays 'x' and 'y' have different lengths.")

    if isinstance(x_range, (list, tuple)) and len(x_range) != 2:
        raise ValueError(
            "'x_range' can either be an integer or a 2-elements list/tuple.")
    elif isinstance(x_range, (list, tuple)) and len(x_range) == 2:
        start_index = get_index(x, x_range[0], closest=closest)
        stop_index = get_index(x, x_range[1], closest=closest)

    # swap indices if start_index > stop_index
    if start_index > stop_index:
        start_index, stop_index = stop_index, start_index

    area = integrate.simps(y[start_index:stop_index + 1],
                           x[start_index:stop_index + 1]) * sort

    return y / area
