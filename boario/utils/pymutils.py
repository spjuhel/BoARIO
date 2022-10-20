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

import pymrio as pym
import numpy as np
import pandas as pd

VA_EXTENSIONS_NAMES = ['satellite', 'factor_input']

def lexico_reindex(mrio: pym.IOSystem) -> pym.IOSystem:
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

    mrio.Z = mrio.Z.reindex(sorted(mrio.Z.index), axis=0)
    mrio.Z = mrio.Z.reindex(sorted(mrio.Z.columns), axis=1)
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.index), axis=0)
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.columns), axis=1)
    mrio.x = mrio.x.reindex(sorted(mrio.x.index), axis=0) #type: ignore
    mrio.A = mrio.A.reindex(sorted(mrio.A.index), axis=0)
    mrio.A = mrio.A.reindex(sorted(mrio.A.columns), axis=1)

    return mrio

def get_wages_from_factor_inputs(mrio: pym.IOSystem) -> pd.DataFrame :
    mrio = lexico_reindex(mrio)
    if "exio3" in str(mrio.name):
        wages_idx = ["Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled",
                     "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled",
                     "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled"]
        if not hasattr(mrio,"satellite"):
            raise ValueError("MRIO appears as EXIOBASE3, but 'satellite' isn't a member...")
        else:
            if not hasattr(mrio.satellite,"F"):
                raise ValueError("satellite has no \"F\" member...")
            else:
                return mrio.satellite.F.loc[wages_idx].sum().unstack()
    else:
        raise NotImplementedError("Other mrio than EXIOBASE3 not implemented yet. Create an issue !")
