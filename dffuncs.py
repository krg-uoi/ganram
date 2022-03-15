# import numpy as np
import pandas as pd
import numpy as np
from functools import reduce
import preprocessing as pre
# from pandas.util._validators import validate_bool_kwarg


def get_row(df, iloc, drop=False, reset_index=True):
    """Get a row from a DataFrame based on its integer-location index.
    If drop is True, remove the row from the DataFrame.
    If reset_index is True, reset the index of the remaining DataFrame.
    """

    if isinstance(iloc, int):
        row = df.iloc[iloc]
    else:
        raise TypeError('Parameter "iloc" must be an int.')

    if drop:
        df.drop(df.index[iloc], inplace=True)

    if reset_index:
        df.reset_index(inplace=True, drop=True)

    return row


def insert_row(df, value, index, inplace=False):
    """Insert a row at the specified integer-location-based index of a
    DataFrame.
    """
    start_top = 0
    stop_top = index
    start_bottom = index + 1
    stop_bottom = df.shape[0] + 1

    top_index = [i for i in range(start_top, stop_top)]
    bottom_index = [i for i in range(start_bottom, stop_bottom)]
    new_index = top_index + bottom_index

    df.index = new_index
    df.loc[index] = value

    if inplace:
        df.sort_index(inplace=True)
    else:
        df = df.sort_index()

    return df


def merge(
        df_list,
        interp=True,
        how='inner',
        on_col=None,
        on_index=True,
        # left_on=None,
        # right_on=None,
        # left_index=False,
        # right_index=False,
        col_names=None,
        # **kwargs
        ):
    """Merge a list of dataframes in a single dataframe.
    """
    # need for multiple suffixes needs to be further evaluated
    # https://stackoverflow.com/questions/44327999/python-pandas-merge-multiple-dataframes/44338256
    # if suffixes:
    #     for i, df in enumerate(df_list):
    #         df_new_cols = [f'{col}_{suffixes[i]}'
    #                        for col in df.loc[:, df.columns != on].columns]
    #         d = dict(zip(df.loc[:, df.columns != on].columns, df_new_cols))
    #         df_list[i] = df.rename(columns=d)
    if interp:
        df_list = [interpolate(df, x2=df_list[0].index) for df in df_list]

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=on_col,
                                                    how=how,
                                                    left_index=on_index,
                                                    right_index=on_index,
                                                    # suffixes=suffixes,
                                                    # **kwargs
                                                    ),
                       df_list)

    if col_names is not None:
        df_merged.columns = col_names

    return df_merged


################################################################
# needs to be checked, because it was deleted probably by mistake!!!!!!
def get_index(df, value, column=None, closest=True):
    """Get the index of a cell value.
    If closest is true, get the index of the value closest to the
    value entered.
    """
    if column is None:
        index = np.argsort(np.abs(df.index - value))[0]
    elif isinstance(column, str):
        if closest:
            index = df.iloc[(df[column]-value).abs().argsort()[:1]].index[0]
        else:
            try:
                index = df[df[column] == value].index[0]
            except IndexError:
                raise ValueError('No index for the specified value.')
    return index
######################################################################


# Get the interpolation intersection ox x1 and x2.)


def reverse(df, reset_index=False):
    """Reverse the dataframe.
    """
    df_reversed = df.iloc[::-1]

    if reset_index:
        df_reversed = df_reversed.reset_index(drop=True)

    return df_reversed


def interpolate(df, x2, column=None, kind='cubic', reset_index=False):
    """Interpolate all columns of a dataframe with a given array x2.
    """
    # column: name of initial column for y=f(x1)
    # x2: final column
    if column is None:
        x1 = df.index
    elif isinstance(column, str):
        x1 = df[column].values

    df_interp = df.apply(
        lambda x: pre.interpolate(
            x1=x1,
            y1=x,
            x2=x2,
            kind=kind
        )
    )

    # replace df[column], which was also interpolated in the previous step,
    # with the intersection of the interpolation arrays
    if column is None:
        df_interp.index = pre.interpolation_intersection(df.index, x2)
    elif isinstance(column, str):
        df_interp[column] = pre.interpolation_intersection(df[column].values,
                                                           x2)

    return df_interp


def snip(df, iterations, increasing=False, exclude=None):
    """Apply the SNIP algorithm for background removal on the dataframe.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    df_snipped = df.apply(
        lambda x: pre.snip(x, iterations, increasing=increasing)
        if x.name not in exclude
        else x
    )

    return df_snipped


def smooth(df, window_length, polyorder, deriv=0, mode='interp',
           exclude=None):
    """Apply the SNIP algorithm for background removal on the dataframe.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    df_smoothed = df.apply(
        lambda x: pre.smooth(
            x,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            mode=mode
        )
        if x.name not in exclude
        else x
    )

    return df_smoothed


def crop(df, start_value, stop_value=None, column=None, closest=True,
         reset_index=False):
    """Crop the dataframe.
    """
    if column is None:
        if closest:
            method = 'nearest'
        else:
            method = None

        start_index = df.index.get_loc(start_value, method=method)

        if stop_value is None:
            stop_index = None
        else:
            stop_index = df.index.get_loc(stop_value, method=method) + 1
    else:
        start_index = pre.get_index(df[column].values, start_value,
                                    closest=closest)

        if stop_value is None:
            stop_index = None
        else:
            stop_index = pre.get_index(df[column].values, stop_value,
                                       closest=closest) + 1

    df_cropped = df.iloc[start_index:stop_index]

    if reset_index:
        df_cropped.reset_index(drop=True, inplace=True)

    return df_cropped


def norm_peak(df, peak, x=None, closest=True):
    """Normalize the dataframe to a specified peak.
    x is the name or location index of a dataframe column.
    """
    if x is None:
        x = df.index

        df_normed = df.apply(
            lambda y: pre.norm_peak(
                y,
                x=x,
                peak=peak,
                closest=closest
            ),
            raw=True
        )

        return df_normed

    elif isinstance(x, str):
        x_col = x
        x = df[x]
    elif isinstance(x, int):
        x_col = df.columns[x]
        x = df.iloc[:, x]

    df_normed = df.apply(
        lambda y: pre.norm_peak(
            y,
            x=x,
            peak=peak,
            closest=closest
        )
        if y.name != x_col
        else y,
        raw=True
    )

    return df_normed


def norm_area(df, x_range, x=None, closest=True):
    """Normalize the dataframe to the selected area.
    x is the name or location index of a dataframe column.
    """
    if x is None:
        x = df.index

        df_normed = df.apply(
            lambda y: pre.norm_area(
                y,
                x=x,
                x_range=x_range,
                closest=closest
            ),
            raw=True
        )

        return df_normed

    if isinstance(x, str):
        x_col = x
        x = df[x]
    elif isinstance(x, int):
        x_col = df.columns[x]
        x = df.iloc[:, x]

    df_normed = df.apply(
        lambda y: pre.norm_area(
            y,
            x=x,
            x_range=x_range,
            closest=closest
        )
        if y.name != x_col
        else y,
        raw=True
    )

    return df_normed


def differentiate(df, x=None, order=1):
    """Apply n-th order differentiation on the dataframe.
    """
    if x is None:
        x = df.index

        df_diffed = df.apply(
            lambda y: pre.differentiate(
                y,
                x=x,
                order=order
            ),
            raw=True
        )

        return df_diffed

    if isinstance(x, str):
        x_col = x
        x = df[x]
    elif isinstance(x, int):
        x_col = df.columns[x]
        x = df.iloc[:, x]

    df_diffed = df.apply(
        lambda y: pre.differentiate(
            y,
            x=x,
            order=order
        )
        if y.name != x_col
        else y,
        raw=True
    )

    return df_diffed


def subtract(df, y, exclude=None):
    """Subtract array-like y from dataframe columns.
    """
    if exclude is None:
        exclude = []
    df_subtracted = df.apply(
        lambda x: x - y
        if x.name not in exclude
        else x
    )

    return df_subtracted


def transpose(df, col_as_header=None, header_as_col=False):
    """Transpose the dataframe,
    e.g. from

    | Index | Raman Shift | Sample_1 | Sample_2 | Sample_3 |
    |:-----:|:-----------:|:--------:|:--------:|:--------:|
    |   1   |    500.5    |     1    |     4    |     7    |
    |   2   |    501.0    |     2    |     5    |     8    |
    |   3   |    501.5    |     3    |     6    |     9    |

    to

    | Raman Shift | 500.5 | 501.0 | 501.5 |
    |:-----------:|:-----:|:-----:|:-----:|
    |   Sample_1  |   1   |   2   |   3   |
    |   Sample_2  |   4   |   5   |   6   |
    |   Sample_3  |   7   |   8   |   9   |

    'col_as_header' determines if we want to use a column of the initial
    dataframe as the header of the transposed dataframe. It can be a string
    containing the name of the column to be used, an integer that defines the
    position of the column to be used or 'None', if we just want to transpose
    the dataframe.

    'header_as_col' determines if we want to use the header of the initial
    dataframe as a column of the transposed dataframe. It can be a boolean or
    a string that defines the name of the column.
    """
    if isinstance(col_as_header, str):
        transposed_df = df.set_index(col_as_header).T
    elif isinstance(col_as_header, int):
        transposed_df = df.set_index(df.columns[col_as_header]).T
    elif col_as_header is None:
        transposed_df = df.T

    if isinstance(header_as_col, str):
        transposed_df.index = transposed_df.index.set_names(header_as_col)
        transposed_df.reset_index(inplace=True)
    elif header_as_col:
        transposed_df = transposed_df.reset_index()

    return transposed_df

# def preprocess(
#         df,
#         interpolate=None,
#         smooth=None,
#         snip=None,
#         norm_peak=None,
#         norm_area=None,
#     )
