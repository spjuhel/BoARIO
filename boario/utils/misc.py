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

from boario.event import Event
from collections.abc import Iterable
import pymrio

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def lexico_reindex(mrio: pymrio.IOSystem) -> pymrio.IOSystem:
    """Reindex IOSystem lexicographicaly

    Sort indexes and columns of the dataframe of a ``pymrio`` `IOSystem <https://pymrio.readthedocs.io/en/latest/intro.html>` by
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
        raise ValueError("Given mrio has no Z attribute set")
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.index), axis=0)
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.columns), axis=1)
    if mrio.x is None:
        raise ValueError("Given mrio has no Z attribute set")
    mrio.x = mrio.x.reindex(sorted(mrio.x.index), axis=0)
    if mrio.A is None:
        raise ValueError("Given mrio has no Z attribute set")
    mrio.A = mrio.A.reindex(sorted(mrio.A.index), axis=0)
    mrio.A = mrio.A.reindex(sorted(mrio.A.columns), axis=1)

    return mrio
