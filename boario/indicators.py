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
from __future__ import annotations

from typing import Dict, Optional, Union

from pymrio.core.mriosystem import collections
from boario.extended_models import ARIOPsiModel
from boario.simulation import Simulation
import numpyencoder
import json
import pathlib
import numpy as np
import pandas as pd
import itertools
from boario.utils import misc
import dask.dataframe as da
from boario import logger
from boario.event import *

__all__ = ["Indicators"]


def create_df(data, regions, sectors):
    df = pd.DataFrame(
        data,
        columns=pd.MultiIndex.from_product(
            [regions, sectors], names=["region", "sector"]
        ),
    )
    df = df.interpolate()
    df = df.rename_axis("step")
    return df


def df_from_memmap(memmap: np.memmap, indexes: dict) -> pd.DataFrame:
    a = pd.DataFrame(
        memmap,
        columns=pd.MultiIndex.from_product([indexes["regions"], indexes["sectors"]]),
    )
    a["step"] = a.index
    return a


def stock_df_from_memmap(
    memmap: np.memmap, indexes: dict, timesteps: int, timesteps_simulated: int
) -> pd.DataFrame:
    a = pd.DataFrame(
        memmap.reshape(timesteps * indexes["n_sectors"], -1),
        index=pd.MultiIndex.from_product(
            [timesteps, indexes["sectors"]], names=["step", "stock of"]
        ),
        columns=pd.MultiIndex.from_product([indexes["regions"], indexes["sectors"]]),
    )
    a = a.loc[pd.IndexSlice[:timesteps_simulated, :]]
    return a


class Indicators(object):
    record_files_list = [
        "final_demand_record",
        "io_demand_record",
        "final_demand_unmet_record",
        "iotable_X_max_record",
        "iotable_XVA_record",
        "limiting_stocks_record",
        "overprodvector_record",
        "rebuild_demand_record",
        "rebuild_prod_record",
        "iotable_kapital_destroyed_record",
    ]

    params_list = ["simulated_params", "simulated_events"]

    def __init__(
        self,
        *,
        has_crashed,
        params,
        regions,
        sectors,
        kapital,
        prod,
        prodmax,
        overprod,
        final_demand,
        io_demand,
        r_demand,
        r_prod,
        fd_unmet,
        limiting_stocks,
        n_temporal_units_to_sim: int,
        n_temporal_units_by_step: int,
        events: list[Event],
        results_storage: Optional[str | pathlib.Path] = None,
        psi_param: Optional[float] = None,
        inventory_restoration_tau: Optional[int | list[int]] = None,
        stocks=None,
        stocks_treatment=False,
        include_crash: bool = False,
    ) -> None:
        logger.info("Instanciating indicators")
        super().__init__()
        self.data_dict = data_dict
        if not include_crash:
            if has_crashed:
                raise RuntimeError(
                    "Simulation crashed and include_crash is False, I won't compute indicators"
                )
        steps = [i for i in range(n_temporal_units_to_sim)]

        self.params = params
        self.n_rows = n_temporal_units_to_sim
        self.n_temporal_units_by_step = n_temporal_units_by_step

        self.kapital_df = create_df(kapital, regions, sectors)
        self.prod_df = create_df(prod, regions, sectors)
        self.prodmax_df = create_df(prodmax, regions, sectors)
        self.overprod_df = create_df(overprod, regions, sectors)
        self.final_demand_df = create_df(final_demand, regions, sectors)
        self.io_demand_df = create_df(io_demand, regions, sectors)
        self.r_demand_df = create_df(r_demand, regions, sectors)
        self.r_prod_df = create_df(r_prod, regions, sectors)
        self.fd_unmet_df = create_df(fd_unmet, regions, sectors)

        if stocks_treatment:
            stocks_df = pd.DataFrame(
                stocks.reshape(n_temporal_units_to_sim * n_sectors, -1),
                index=pd.MultiIndex.from_product(
                    [steps, sectors], names=["step", "stock of"]
                ),
                columns=pd.MultiIndex.from_product(
                    [regions, sectors],
                    names=["region", "sector"],
                ),
            )
            stocks_df = stocks_df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
            stocks_df = stocks_df.astype(np.float32)
            stocks_df = (
                stocks_df.groupby("stock of")
                .pct_change()
                .fillna(0)
                .add(1)
                .groupby("stock of")
                .cumprod()
                .sub(1)
            )  # type: ignore
            stocks_df = stocks_df.melt(ignore_index=False).rename(
                columns={
                    "variable_0": "region",
                    "variable_1": "sector",
                    "variable_2": "stock of",
                }
            )
            stocks_df = stocks_df.reset_index()
            stocks_df["step"] = stocks_df["step"].astype("uint16")
            stocks_df["stock of"] = stocks_df["stock of"].astype("category")
            stocks_df["region"] = stocks_df["region"].astype("category")
            stocks_df["sector"] = stocks_df["sector"].astype("category")

        else:
            stocks_df = None

        self.df_stocks = stocks_df
        del stocks_df
        # self.df_stocks = self.df_stocks.interpolate()

        self.df_loss = (
            self.fd_unmet_df.melt(ignore_index=False)
            .rename(
                columns={
                    "variable_0": "region",
                    "variable_1": "fd_cat",
                    "value": "fdloss",
                }
            )
            .reset_index()
        )

        self.df_limiting = pd.DataFrame(
            limiting_stocks.reshape(n_temporal_units_to_sim * n_sectors, -1),
            index=pd.MultiIndex.from_product(
                [steps, sectors], names=["step", "stock of"]
            ),
            columns=pd.MultiIndex.from_product(
                [regions, sectors], names=["region", "sector"]
            ),
        )
        self.aff_regions = []
        for e in events:
            self.aff_regions.append(e.aff_regions)

        self.aff_regions = list(misc.flatten(self.aff_regions))

        self.aff_sectors = []
        for e in events:
            self.aff_sectors.append(e.aff_sectors)
        self.aff_sectors = list(misc.flatten(self.aff_sectors))

        self.rebuilding_sectors = []

        # This is working but wrong ! I need to fix it
        for e in events:
            if isinstance(e, EventKapitalRebuild):
                self.rebuilding_sectors.append(list(e.rebuilding_sectors))
                self.rebuilding_sectors = list(misc.flatten(self.rebuilding_sectors))
            else:
                self.rebuilding_sectors = []

        # This is also wrong but will settle for now
        gdp_dmg_share = -1.0
        self.indicators = {
            "region": self.aff_regions,
            "gdp_dmg_share": gdp_dmg_share,
            "tot_fd_unmet": "unset",
            "aff_fd_unmet": "unset",
            "rebuild_durations": "unset",
            "shortage_b": False,
            "shortage_date_start": "unset",
            "shortage_date_end": "unset",
            "shortage_date_max": "unset",
            "shortage_ind_max": "unset",
            "shortage_ind_mean": "unset",
            "10_first_shortages_(step,region,sector,stock_of)": "unset",
            "prod_gain_tot": "unset",
            "prod_lost_tot": "unset",
            "prod_gain_unaff": "unset",
            "prod_lost_unaff": "unset",
            "psi": psi_param,
            "inv_tau": inventory_restoration_tau,
            "n_temporal_units_to_sim": n_temporal_units_to_sim,
            "has_crashed": has_crashed,
        }
        if results_storage is not None:
            self.storage_path = (pathlib.Path(results_storage)).resolve() / "indicators"
            self.parquets_path = (pathlib.Path(results_storage)).resolve() / "parquets"
            self.storage = self.storage_path / "indicators.json"
            self.parquets_path.mkdir(parents=True, exist_ok=True)
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.save_dfs()

    @classmethod
    def from_simulation(
        cls,
        sim: Simulation,
        stocks_treatment: bool = False,
        include_crash: bool = False,
        results_storage: Optional[str | pathlib.Path] = None,
    ) -> Indicators:
        if isinstance(sim.model, ARIOPsiModel):
            psi = sim.model.psi
            inventory_restoration_tau = list(sim.model.restoration_tau)
        else:
            psi = None
            inventory_restoration_tau = None

        prod = getattr(sim, "production_evolution")
        kapital = getattr(sim, "regional_sectoral_kapital_destroyed_evolution")
        prodmax = getattr(sim, "production_cap_evolution")
        overprod = getattr(sim, "overproduction_evolution")
        final_demand = getattr(sim, "final_demand_evolution")
        io_demand = getattr(sim, "io_demand_evolution")
        r_demand = getattr(sim, "rebuild_demand_evolution")
        r_prod = getattr(sim, "rebuild_production_evolution")
        fd_unmet = getattr(sim, "final_demand_unmet_evolution")
        if stocks_treatment:
            stocks = getattr(sim, "inputs_evolution")
        else:
            stocks = None

        limiting_stocks = getattr(sim, "limiting_inputs_evolution")
        return cls(
            has_crashed=sim.has_crashed,
            params=sim.params_dict,
            regions=sim.model.regions,
            sectors=sim.model.sectors,
            kapital=kapital,
            prod=prod,
            prodmax=prodmax,
            overprod=overprod,
            final_demand=final_demand,
            io_demand=io_demand,
            r_demand=r_demand,
            r_prod=r_prod,
            fd_unmet=fd_unmet,
            limiting_stocks=limiting_stocks,
            n_temporal_units_to_sim=sim.n_temporal_units_to_sim,
            n_temporal_units_by_step=sim.model.n_temporal_units_by_step,
            events=sim.all_events,
            results_storage=results_storage,
            psi_param=psi,
            inventory_restoration_tau=inventory_restoration_tau,
            stocks=stocks,
            stocks_treatment=stocks_treatment,
            include_crash=include_crash,
        )

    @classmethod
    def from_storage_path(
        cls, storage_path: str, params=None, include_crash: bool = False
    ) -> Indicators:
        raise NotImplementedError()
        return cls(
            cls.dict_from_storage_path(storage_path, params=params), include_crash
        )

    # noinspection PyTypeChecker
    @classmethod
    def from_folder(
        cls,
        folder: Union[str, pathlib.Path],
        indexes_file: Union[str, pathlib.Path],
        include_crash: bool = False,
    ) -> Indicators:
        raise NotImplementedError()
        data_dict = {}
        if not isinstance(indexes_file, pathlib.Path):
            indexes_file = pathlib.Path(indexes_file)
            if not indexes_file.exists():
                raise FileNotFoundError(str("File does not exist:" + str(indexes_file)))
        if not isinstance(folder, pathlib.Path):
            folder = pathlib.Path(folder)
            if not folder.exists():
                raise FileNotFoundError(str("Directory does not exist:" + str(folder)))
        with indexes_file.open() as f:
            indexes = json.load(f)

        params_folder = folder / "jsons"
        records_folder = folder / "records"
        params_file = {f.stem: f for f in params_folder.glob("*.json")}
        absentee = [f for f in cls.params_list if f not in params_file.keys()]
        if absentee:
            raise FileNotFoundError(
                "Some of the required parameters files not found (looked for {} in {}".format(
                    cls.params_list, folder
                )
            )

        record_files = [f for f in records_folder.glob("*record") if f.is_file()]
        absentee = [
            f
            for f in cls.record_files_list
            if f not in [fn.name for fn in record_files]
        ]
        if absentee:
            raise FileNotFoundError(
                "Some of the required records are not there : {}".format(absentee)
            )

        with params_file["simulated_params"].open("r") as f:
            params = json.load(f)

        with params_file["simulated_events"].open("r") as f:
            events = json.load(f)

        if "has_crashed" in params:
            data_dict["has_crashed"] = params["has_crashed"]
        else:
            data_dict["has_crashed"] = False
        data_dict["results_storage"] = folder.absolute()
        records_path = records_folder.absolute()
        data_dict["n_temporal_units_to_sim"] = params["n_temporal_units_to_sim"]
        t = data_dict["n_temporal_units_to_sim"]
        data_dict["params"] = params
        data_dict["n_temporal_units_simulated"] = params["n_temporal_units_simulated"]
        data_dict["n_temporal_units_by_step"] = params["n_temporal_units_by_step"]
        data_dict["regions"] = indexes["regions"]
        data_dict["n_regions"] = indexes["n_regions"]
        data_dict["sectors"] = indexes["sectors"]
        data_dict["n_sectors"] = indexes["n_sectors"]
        data_dict["events"] = events
        data_dict["prod"] = np.memmap(
            records_path / "iotable_XVA_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["kapital"] = np.memmap(
            records_path / "iotable_kapital_destroyed_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["prodmax"] = np.memmap(
            records_path / "iotable_X_max_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["overprod"] = np.memmap(
            records_path / "overprodvector_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["final_demand"] = np.memmap(
            records_path / "final_demand_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["io_demand"] = np.memmap(
            records_path / "io_demand_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["r_demand"] = np.memmap(
            records_path / "rebuild_demand_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["r_prod"] = np.memmap(
            records_path / "rebuild_prod_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["fd_unmet"] = np.memmap(
            records_path / "final_demand_unmet_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_regions"] * indexes["n_sectors"]),
        )
        data_dict["limiting_stocks"] = np.memmap(
            records_path / "limiting_stocks_record",
            mode="r+",
            dtype="byte",
            shape=(t * indexes["n_sectors"], indexes["n_industries"]),
        )
        if params["register_stocks"]:
            if not (records_path / "stocks_record").exists():
                raise FileNotFoundError(
                    "Stocks record file was not found {}".format(
                        records_path / "stocks_record"
                    )
                )
            data_dict["stocks"] = np.memmap(
                records_path / "stocks_record",
                mode="r+",
                dtype="float64",
                shape=(t * indexes["n_sectors"], indexes["n_industries"]),
            )
        return cls(data_dict, include_crash)

    # noinspection PyTypeChecker
    @classmethod
    def dict_from_storage_path(
        cls, storage_path: Union[str, pathlib.Path], params=None
    ) -> dict:
        data_dict = {}
        if not isinstance(storage_path, pathlib.Path):
            storage_path = pathlib.Path(storage_path)
            assert storage_path.exists(), str(
                "Directory does not exist:" + str(storage_path)
            )
        if params is not None:
            simulation_params = params
        else:
            with (storage_path / "simulated_params.json").open() as f:
                simulation_params = json.load(f)
        if (
            storage_path
            / simulation_params["results_storage"]
            / "jsons"
            / "simulated_params.json"
        ).exists():
            with (
                storage_path
                / simulation_params["results_storage"]
                / "jsons"
                / "simulated_params.json"
            ).open() as f:
                simulation_params = json.load(f)
        with (
            storage_path
            / simulation_params["results_storage"]
            / "jsons"
            / "indexes.json"
        ).open() as f:
            indexes = json.load(f)
        with (
            storage_path
            / simulation_params["results_storage"]
            / "jsons"
            / "simulated_events.json"
        ).open() as f:
            events = json.load(f)
        t = simulation_params["n_temporal_units_to_sim"]
        if indexes["fd_cat"] is None:
            indexes["fd_cat"] = np.array(["Final demand"])
        results_path = storage_path / pathlib.Path(simulation_params["results_storage"])
        if "has_crashed" in simulation_params:
            data_dict["has_crashed"] = simulation_params["has_crashed"]
        data_dict["params"] = simulation_params
        data_dict["results_storage"] = results_path
        data_dict["n_temporal_units_by_step"] = simulation_params[
            "n_temporal_units_by_step"
        ]
        data_dict["n_temporal_units_to_sim"] = t
        data_dict["n_temporal_units_simulated"] = simulation_params[
            "n_temporal_units_simulated"
        ]
        data_dict["regions"] = indexes["regions"]
        data_dict["n_regions"] = indexes["n_regions"]
        data_dict["sectors"] = indexes["sectors"]
        data_dict["n_sectors"] = indexes["n_sectors"]
        data_dict["events"] = events
        data_dict["prod"] = np.memmap(
            results_path / "iotable_XVA_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["kapital"] = np.memmap(
            results_path / "iotable_kapital_destroyed_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["prodmax"] = np.memmap(
            results_path / "iotable_X_max_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["overprod"] = np.memmap(
            results_path / "overprodvector_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["final_demand"] = np.memmap(
            results_path / "final_demand_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["io_demand"] = np.memmap(
            results_path / "io_demand_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["r_demand"] = np.memmap(
            results_path / "rebuild_demand_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["r_prod"] = np.memmap(
            results_path / "rebuild_prod_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        data_dict["fd_unmet"] = np.memmap(
            results_path / "final_demand_unmet_record",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_regions"] * indexes["n_sectors"]),
        )
        if simulation_params["register_stocks"]:
            data_dict["stocks"] = np.memmap(
                results_path / "stocks_record",
                mode="r+",
                dtype="float64",
                shape=(t * indexes["n_sectors"], indexes["n_industries"]),
            )
        data_dict["limiting_stocks"] = np.memmap(
            results_path / "limiting_stocks_record",
            mode="r+",
            dtype="byte",
            shape=(t * indexes["n_sectors"], indexes["n_industries"]),
        )
        return data_dict

    def calc_top_failing_sect(self):
        pass

    def calc_tot_fd_unmet(self):
        self.indicators["tot_fd_unmet"] = self.df_loss["fdloss"].sum()

    def calc_aff_fd_unmet(self):
        # TODO: check this
        self.indicators["aff_fd_unmet"] = self.df_loss[
            self.df_loss.region.isin(self.aff_regions)
        ]["fdloss"].sum()

    def calc_rebuild_durations(self):
        rebuilding = (
            self.r_demand_df.reset_index()
            .melt(
                id_vars="step",
                var_name=["region", "sector"],
                value_name="rebuild_demand",
            )
            .groupby("step")
            .sum(numeric_only=True)
            .ne(0)
            .rebuild_demand.to_numpy()
        )
        self.indicators["rebuild_durations"] = [
            sum(1 for _ in group) for key, group in itertools.groupby(rebuilding) if key
        ]

    def calc_recovery_duration(self):
        pass

    def calc_general_shortage(self):
        # TODO: replace hard values by soft.
        n_regions = self.df_limiting.columns.levels[0].size
        n_sectors = self.df_limiting.columns.levels[1].size
        # have only steps in index (next step not possible with multiindex)
        a = self.df_limiting.unstack()
        # select only simulated steps (we can't store nan in bool or byte dtype array)
        a = a.iloc[lambda x: x.index % self.n_temporal_units_by_step == 0]
        # We put -1 initially, so this should check we have correctly selected the simulated rows
        assert (a >= 0).all().all()
        # put input stocks as columns and the rest as index
        a = a.stack().T.stack(level=0)
        # sum for all input and divide by n_sector to get "input shortage fraction"
        # by industry
        b = a.sum(axis=1).groupby(["step", "region", "sector"]).sum() / n_sectors
        # by sector
        c = b.groupby(["step", "region"]).sum() / n_sectors
        # by region
        c = c.groupby("step").sum() / n_regions
        if not c.ne(0).any():
            self.indicators["shortage_b"] = False
        else:
            self.indicators["shortage_b"] = True
            shortage_date_start = c.ne(0.0).idxmax()
            self.indicators["shortage_date_start"] = shortage_date_start
            shortage_date_end = c.loc[shortage_date_start:].eq(0).idxmax()
            self.indicators["shortage_date_end"] = shortage_date_end
            self.indicators["shortage_date_max"] = c.idxmax()
            self.indicators["shortage_ind_max"] = c.max()
            self.indicators["shortage_ind_mean"] = c.loc[
                shortage_date_start:shortage_date_end
            ].mean()

    def calc_first_shortages(self):
        a = self.df_limiting.stack([0, 1])  # type: ignore
        a = a.swaplevel(1, 2).swaplevel(2, 3)
        b = a[a > 0]
        b = b[:10]
        res = list(b.index)  # type:ignore
        self.indicators["10_first_shortages_(step,region,sector,stock_of)"] = res

    def calc_tot_prod_change(self):
        df2 = self.prod_df.copy()
        # df2.columns=df2.columns.droplevel(0)
        prod_chg = df2 - df2.iloc[0, :]
        # Round to € to avoid floating error differences
        prod_chg = prod_chg.round(6)
        # Aggregate rebuilding and non-rebuilding sectors
        prod_chg_agg_1 = pd.concat(
            [
                prod_chg.loc[:, pd.IndexSlice[:, self.rebuilding_sectors]]
                .groupby("region", axis=1)
                .sum(),
                prod_chg.loc[
                    prod_chg.index.difference(
                        prod_chg.loc[
                            :, pd.IndexSlice[:, self.rebuilding_sectors]
                        ].columns
                    )
                ]
                .groupby("region", axis=1)
                .sum(),
            ],
            keys=["rebuilding", "non-rebuilding"],
            names=["sectors affected"],
            axis=1,
        )

        n_semesters = self.n_rows // (self.params["year_to_temporal_unit_factor"] // 2)
        row_to_semester = self.params["year_to_temporal_unit_factor"] // 2
        modulo = self.params["year_to_temporal_unit_factor"] % 2
        logger.info(
            "There are {} semesters [{} rows, each representing a {}th of a year]".format(
                n_semesters, self.n_rows, self.params["year_to_temporal_unit_factor"]
            )
        )
        prod_chg_sect_sem_l = []
        for sem in range(0, n_semesters):
            prod_chg_sect_sem_l.append(
                prod_chg_agg_1.iloc[
                    sem * row_to_semester : ((sem + 1) * row_to_semester)
                    + (sem % 2) * modulo
                ]
                .sum()
                .T
            )

        prod_chg_region = pd.concat(
            prod_chg_sect_sem_l,
            keys=["semester {}".format(v + 1) for v in range(n_semesters)],
            names=["semester"],
        )
        aff_regions = "~".join(self.aff_regions)
        prod_chg_region = pd.DataFrame({aff_regions: prod_chg_region}).T
        prod_chg_region.to_json(
            self.storage_path / "prod_chg.json", indent=4, orient="split"
        )

        prod_chg_sect = prod_chg.sum()
        tmp = prod_chg_sect.sort_values(ascending=False, key=abs).head(5).to_dict()
        self.indicators["top_5_sector_chg"] = {str(k): v for k, v in tmp.items()}

        self.indicators["prod_gain_tot"] = prod_chg.mul(prod_chg.gt(0)).sum().sum()
        self.indicators["prod_lost_tot"] = prod_chg.mul(~prod_chg.gt(0)).sum().sum() * (
            -1
        )
        prod_chg = prod_chg.drop(self.aff_regions, axis=1)
        self.indicators["prod_gain_unaff"] = prod_chg.mul(prod_chg.gt(0)).sum().sum()
        self.indicators["prod_lost_unaff"] = prod_chg.mul(
            ~prod_chg.gt(0)
        ).sum().sum() * (-1)

    def update_indicators(self):
        logger.info("(Re)computing all indicators")
        logger.info("Tot fd unmet")
        self.calc_tot_fd_unmet()
        logger.info("aff fd unmet")
        self.calc_aff_fd_unmet()
        logger.info("rebuild durations")
        self.calc_rebuild_durations()
        logger.info("recovery duration")
        self.calc_recovery_duration()
        logger.info("general shortage")
        self.calc_general_shortage()
        logger.info("tot prod change")
        self.calc_tot_prod_change()
        logger.info("fd loss region")
        self.calc_fd_loss_region()
        logger.info("first shortages")
        self.calc_first_shortages()

    def write_indicators(self):
        logger.info("Writing indicators to json")
        # self.update_indicators()
        with self.storage.open("w") as f:
            json.dump(self.indicators, f, cls=numpyencoder.NumpyEncoder)

    def calc_fd_loss_region(self):
        fd_loss = self.fd_unmet_df.copy().round(6)
        fd_loss_agg_1 = pd.concat(
            [
                fd_loss.loc[:, pd.IndexSlice[:, self.rebuilding_sectors]]
                .groupby("region", axis=1)
                .sum(),
                fd_loss.loc[
                    fd_loss.index.difference(
                        fd_loss.loc[
                            :, pd.IndexSlice[:, self.rebuilding_sectors]
                        ].columns
                    )
                ]
                .groupby("region", axis=1)
                .sum(),
            ],
            keys=["rebuilding", "non-rebuilding"],
            names=["sectors affected"],
            axis=1,
        )

        n_semesters = self.n_rows // (self.params["year_to_temporal_unit_factor"] // 2)
        row_to_semester = self.params["year_to_temporal_unit_factor"] // 2
        modulo = self.params["year_to_temporal_unit_factor"] % 2
        logger.info(
            "There are {} semesters [{} rows, each representing a {}th of a year]".format(
                n_semesters, self.n_rows, self.params["year_to_temporal_unit_factor"]
            )
        )
        fd_loss_sect_sem_l = []
        for sem in range(0, n_semesters):
            fd_loss_sect_sem_l.append(
                fd_loss_agg_1.iloc[
                    sem * row_to_semester : ((sem + 1) * row_to_semester)
                    + (sem % 2) * modulo
                ]
                .sum()
                .T
            )

        fd_loss_region = pd.concat(
            fd_loss_sect_sem_l,
            keys=["semester {}".format(v + 1) for v in range(n_semesters)],
            names=["semester"],
        )
        aff_regions = "~".join(self.aff_regions)
        fd_loss_region = pd.DataFrame({aff_regions: fd_loss_region}).T
        fd_loss_region.to_json(
            self.storage_path / "fd_loss.json", indent=4, orient="split"
        )

        fd_loss_sect = fd_loss.sum()
        tmp = fd_loss_sect.sort_values(ascending=False, key=abs).head(5).to_dict()
        self.indicators["top_5_sector_fdloss"] = {str(k): v for k, v in tmp.items()}

    def save_dfs(self):
        logger.info("Saving computed dataframe to results folder")
        self.prod_df.to_parquet(self.parquets_path / "prod_df.parquet")
        self.kapital_df.to_parquet(self.parquets_path / "kapital_df.parquet")
        self.prodmax_df.to_parquet(self.parquets_path / "prodmax_df.parquet")
        self.overprod_df.to_parquet(self.parquets_path / "overprod_df.parquet")
        self.fd_unmet_df.to_parquet(self.parquets_path / "fd_unmet_df.parquet")
        self.final_demand_df.to_parquet(self.parquets_path / "final_demand_df.parquet")
        self.io_demand_df.to_parquet(self.parquets_path / "io_demand_df.parquet")
        self.r_demand_df.to_parquet(self.parquets_path / "r_demand_df.parquet")
        self.df_loss.to_parquet(self.parquets_path / "treated_df_loss.parquet")
        if self.df_stocks is not None:
            ddf = da.from_pandas(self.df_stocks, chunksize=10000000)
            ddf.to_parquet(
                self.parquets_path / "treated_df_stocks.parquet", engine="pyarrow"
            )
        if self.df_limiting is not None:
            # ddf_l = da.from_pandas(self.df_limiting, chunksize=10000000)
            # ddf_l = ddf_l.melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'variable_2':'stock of'})
            # ddf_l = ddf_l.reset_index()
            # ddf_l['step'] = ddf_l['step'].astype("uint16")
            # ddf_l['stock of'] = ddf_l['stock of'].astype("category")
            # ddf_l['region'] = ddf_l['region'].astype("category")
            # ddf_l['sector'] = ddf_l['sector'].astype("category")
            self.df_limiting.to_parquet(
                self.parquets_path / "treated_df_limiting.parquet", engine="pyarrow"
            )
        # self.df_limiting.to_feather(self.parquets_path/"treated_df_limiting.feather")
