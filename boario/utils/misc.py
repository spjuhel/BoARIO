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

import json
import tempfile
import textwrap
from collections.abc import Iterable

import numpy
import pymrio


class TempMemmap(numpy.memmap):
    def __new__(
        subtype,
        filename,
        dtype=numpy.float64,
        mode="r+",
        offset=0,
        shape=None,
        order="C",
        save=False,
    ):
        if save:
            self = numpy.memmap.__new__(
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
            self = numpy.memmap.__new__(
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
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(CustomNumpyEncoder, self).default(obj)


def flatten(lst):
    for el in lst:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def lexico_reindex(mrio: pymrio.IOSystem) -> pymrio.IOSystem:
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
    if mrio.Z is None:
        raise ValueError("Given mrio has no Z attribute set")
    mrio.Z = mrio.Z.reindex(sorted(mrio.Z.index), axis=0)
    mrio.Z = mrio.Z.reindex(sorted(mrio.Z.columns), axis=1)
    if mrio.Y is None:
        raise ValueError("Given mrio has no Y attribute set")
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.index), axis=0)
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.columns), axis=1)
    if mrio.x is None:
        raise ValueError("Given mrio has no x attribute set")
    mrio.x = mrio.x.reindex(sorted(mrio.x.index), axis=0)
    if mrio.A is None:
        raise ValueError("Given mrio has no A attribute set")
    mrio.A = mrio.A.reindex(sorted(mrio.A.index), axis=0)
    mrio.A = mrio.A.reindex(sorted(mrio.A.columns), axis=1)

    return mrio


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
