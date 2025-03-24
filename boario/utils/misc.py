# BoARIO : The Adaptative Regional Input Output model in python.
# Copyright (C) 2022  Samuel Juhel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import copy
import json
import tempfile
import textwrap
from collections.abc import Iterable

import numpy as np
import pymrio


def _fast_sum(array: np.ndarray, axis: int) -> np.ndarray:
    """
    Perform a fast summation over the specified axis of the input array using einsum.

    Parameters:
        array (np.ndarray): Input array.
        axis (int): Axis along which to perform the summation.

    Returns:
        np.ndarray: Summed array along the specified axis.

    Raises:
        ValueError: If the specified axis is out of bounds for the input array.

    Example:
        array = np.array([[1, 2, 3], [4, 5, 6]])
        result = _fast_sum(array, 0)  # Sum along the first axis
        print(result)  # Output: [5 7 9]
    """
    if axis == 0:
        if array.ndim == 1:
            return np.einsum("i->", array)
        if array.ndim == 2:
            return np.einsum("ij->j", array)
        if array.ndim == 3:
            return np.einsum("ijk->jk", array)
        else:
            raise NotImplementedError("Too many dimensions.")
    elif axis == 1:
        if array.ndim == 2:
            return np.einsum("ij->i", array)
        if array.ndim == 3:
            return np.einsum("ijk->ik", array)
        else:
            raise NotImplementedError("Too many dimensions.")
    elif axis == 2:
        return np.einsum("ijk->ij", array)
    else:
        raise ValueError("Axis out of bounds for input array.")


def _divide_arrays_ignore(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.divide(a, b)
        np.nan_to_num(ret, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return ret


class TempMemmap(np.memmap):
    def __new__(
        subtype,
        filename,
        dtype=np.float64,
        mode="r+",
        offset=0,
        shape=None,
        order="C",
        save=False,
    ):
        if save:
            self = np.memmap.__new__(
                subtype,
                filename=filename,
                dtype=dtype,
                mode=mode,
                offset=offset,
                shape=shape,
                order=order,
            )
            return self
        else:
            ntf = tempfile.NamedTemporaryFile()
            self = np.memmap.__new__(
                subtype,
                ntf,
                dtype=dtype,
                mode=mode,
                offset=offset,
                shape=shape,
                order=order,
            )
            self._tmpfile = ntf
            return self

    def __del__(self):
        if hasattr(self, "_tmpfile") and self._tmpfile is not None:
            self._tmpfile.close()
            del self._tmpfile


class CustomNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(CustomNumpyEncoder, self).default(obj)


def flatten(lst):
    for el in lst:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def lexico_reindex(mriot: pymrio.IOSystem) -> pymrio.IOSystem:
    """Reindex IOSystem lexicographicaly

    Sort indexes and columns of the dataframe of a ``pymrio.IOSystem`` by
    lexical order.

    Parameters
    ----------
    mrio : pymrio.IOSystem
        The IOSystem to sort

    Returns
    -------
    pymrio.IOSystem
        The sorted IOSystem


    """
    mriot = copy.deepcopy(mriot)
    if getattr(mriot, "Z", None) is None:
        raise ValueError("Given mriot has no Z attribute set")
    mriot.Z = mriot.Z.reindex(sorted(mriot.Z.index), axis=0)
    mriot.Z = mriot.Z.reindex(sorted(mriot.Z.columns), axis=1)
    if getattr(mriot, "Y", None) is None:
        raise ValueError("Given mriot has no Y attribute set")
    mriot.Y = mriot.Y.reindex(sorted(mriot.Y.index), axis=0)
    mriot.Y = mriot.Y.reindex(sorted(mriot.Y.columns), axis=1)
    if getattr(mriot, "x", None) is None:
        raise ValueError("Given mriot has no x attribute set")
    mriot.x = mriot.x.reindex(
        sorted(mriot.x.index), axis=0
    )  # ignore type (wrong type hinting in pymrio)
    if getattr(mriot, "A", None) is None:
        raise ValueError("Given mriot has no A attribute set")
    mriot.A = mriot.A.reindex(sorted(mriot.A.index), axis=0)
    mriot.A = mriot.A.reindex(sorted(mriot.A.columns), axis=1)

    return mriot


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def print_summary(my_list):
    if my_list:
        current_element = None
        current_count = 0
        summary = []
        for element in my_list:
            if element != current_element:
                if current_element is not None:
                    if current_count == 1:
                        summary.append(str(current_element))
                    else:
                        summary.append(f"{current_element} (x {current_count})")
                current_element = element
                current_count = 1
            else:
                current_count += 1
        if current_element is not None:
            if current_count == 1:
                summary.append(str(current_element))
            else:
                summary.append(f"{current_element} (x {current_count})")
        total_length = len(my_list)
        total_sum = sum(my_list)
        summary_string = (
            "[" + ", ".join(summary) + f"] (len: {total_length}, sum: {total_sum})"
        )
        return textwrap.wrap(summary_string, width=80)
    else:
        return ""
