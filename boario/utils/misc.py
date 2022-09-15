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

from json import JSONEncoder
class EventEncoder(JSONEncoder):
    def default(self, event:Event):
        dic = dict(event.__dict__)
        del dic["final_demand_rebuild"]
        del dic["final_demand_rebuild_share"]
        del dic["industry_rebuild"]
        del dic["industry_rebuild_share"]
        del dic["production_share_allocated"]
        del dic["rebuildable"]


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
