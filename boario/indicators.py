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

from typing import Optional, Union

from boario.extended_models import ARIOPsiModel
from boario.simulation import Simulation
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
        "final_demand",
        "intermediate_demand",
        "final_demand_unmet",
        "production_capacity",
        "production_realised",
        "limiting_inputs",
        "overproduction",
        "rebuild_demand",
        "rebuild_prod",
        "productive_capital_to_recover",
    ]

    params_list = ["simulated_params", "simulated_events"]

    def __init__(
        self,
        *,
        has_crashed,
        params,
        regions,
        sectors,
        productive_capital,
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
        logger.info("Instantiating indicators")
        if not include_crash:
            if has_crashed:
                raise RuntimeError(
                    "Simulation crashed and include_crash is False, I won't compute indicators"
                )
        steps = [i for i in range(n_temporal_units_to_sim)]

        self.params = params
        self.n_rows = n_temporal_units_to_sim
        self.n_temporal_units_by_step = n_temporal_units_by_step

        self.productive_capital_to_recover_df = create_df(
            productive_capital, regions, sectors
        )
        self.production_realised_df = create_df(prod, regions, sectors)
        self.production_capacity_df = create_df(prodmax, regions, sectors)
        self.overproduction_df = create_df(overprod, regions, sectors)
        self.final_demand_df = create_df(final_demand, regions, sectors)
        self.intermediate_demand_df = create_df(io_demand, regions, sectors)
        self.rebuild_demand_df = create_df(r_demand, regions, sectors)
        self.rebuild_prod_df = create_df(r_prod, regions, sectors)
        self.final_demand_unmet_df = create_df(fd_unmet, regions, sectors)

        if stocks_treatment:
            stocks_df = pd.DataFrame(
                stocks.reshape(n_temporal_units_to_sim * len(sectors), -1),
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

        self.inputs_stocks_df = stocks_df
        del stocks_df
        # self.df_stocks = self.df_stocks.interpolate()

        self.treated_loss_df = (
            self.final_demand_unmet_df.melt(ignore_index=False)
            .rename(
                columns={
                    "variable_0": "region",
                    "variable_1": "fd_cat",
                    "value": "fdloss",
                }
            )
            .reset_index()
        )

        self.limiting_inputs_df = limiting_stocks
        # pd.DataFrame(
        #     limiting_stocks.reshape(n_temporal_units_to_sim * len(sectors), -1),
        #     index=pd.MultiIndex.from_product(
        #         [steps, sectors], names=["step", "stock of"]
        #     ),
        #     columns=pd.MultiIndex.from_product(
        #         [regions, sectors], names=["region", "sector"]
        #     ),
        # )
        self.aff_regions = []
        self.aff_sectors = []
        for e in events:
            if isinstance(e, dict):
                self.aff_sectors.append(e["aff_sectors"])
                self.aff_regions.append(e["aff_regions"])
            elif isinstance(e, Event):
                self.aff_sectors.append(e.aff_sectors)
                self.aff_regions.append(e.aff_regions)
            else:
                raise ValueError(f"Unrecognised event of type {type(e)}")

        self.aff_regions = list(misc.flatten(self.aff_regions))
        self.aff_sectors = list(misc.flatten(self.aff_sectors))

        self.rebuilding_sectors = []

        # This is working but wrong ! I need to fix it
        for e in events:
            if isinstance(e, dict):
                self.rebuilding_sectors.append(e.get("rebuilding_sectors"))
            if isinstance(e, EventKapitalRebuild):
                self.rebuilding_sectors.append(list(e.rebuilding_sectors))
            else:
                self.rebuilding_sectors = []

        self.rebuilding_sectors = list(misc.flatten(self.rebuilding_sectors))
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
        results_storage: Optional[str | pathlib.Path] = None,
        stocks_treatment: bool = False,
        include_crash: bool = False,
    ) -> Indicators:
        if isinstance(sim.model, ARIOPsiModel):
            psi = sim.model.psi
            inventory_restoration_tau = list(sim.model.restoration_tau)
        else:
            psi = None
            inventory_restoration_tau = None

        prod = getattr(sim, "production_realised")
        productive_capital = getattr(sim, "productive_capital_to_recover")
        prodmax = getattr(sim, "production_capacity")
        overprod = getattr(sim, "overproduction")
        final_demand = getattr(sim, "final_demand")
        io_demand = getattr(sim, "intermediate_demand")
        r_demand = getattr(sim, "rebuild_demand")
        r_prod = getattr(sim, "rebuild_prod")
        fd_unmet = getattr(sim, "final_demand_unmet")
        if stocks_treatment:
            stocks = getattr(sim, "inputs_stocks")
        else:
            stocks = None

        limiting_stocks = getattr(sim, "limiting_inputs")
        return cls(
            has_crashed=sim.has_crashed,
            params=sim.params_dict,
            regions=sim.model.regions,
            sectors=sim.model.sectors,
            productive_capital=productive_capital,
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
    def from_folder(
        cls,
        folder: Union[str, pathlib.Path],
        include_crash: bool = False,
    ) -> Indicators:
        if not isinstance(folder, pathlib.Path):
            folder = pathlib.Path(folder)
            if not folder.exists():
                raise FileNotFoundError(str("Directory does not exist:" + str(folder)))

        with (folder / "jsons" / "indexes.json").open() as f:
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

        record_files = [f for f in records_folder.glob("*") if f.is_file()]
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
            has_crashed = params["has_crashed"]
        else:
            has_crashed = False

        results_storage = folder.absolute()
        records_path = records_folder.absolute()
        n_temporal_units_to_sim = params["n_temporal_units_to_sim"]
        t = n_temporal_units_to_sim
        n_temporal_units_by_step = params["n_temporal_units_by_step"]
        regions = indexes["regions"]
        sectors = indexes["sectors"]
        psi = params.get("psi_param")
        inventory_restoration_tau = params.get("inventory_restoration_tau")
        prod = np.memmap(
            records_path / "production_realised",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        productive_capital = np.memmap(
            records_path / "productive_capital_to_recover",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        prodmax = np.memmap(
            records_path / "production_capacity",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        overprod = np.memmap(
            records_path / "overproduction",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        final_demand = np.memmap(
            records_path / "final_demand",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        io_demand = np.memmap(
            records_path / "intermediate_demand",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        r_demand = np.memmap(
            records_path / "rebuild_demand",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        r_prod = np.memmap(
            records_path / "rebuild_prod",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_industries"]),
        )
        fd_unmet = np.memmap(
            records_path / "final_demand_unmet",
            mode="r+",
            dtype="float64",
            shape=(t, indexes["n_regions"] * indexes["n_sectors"]),
        )
        limiting_stocks = np.memmap(
            records_path / "limiting_inputs",
            mode="r+",
            dtype="byte",
            shape=(t * indexes["n_sectors"], indexes["n_industries"]),
        )
        if params.get("register_stocks", False):
            if not (records_path / "inputs_stocks").exists():
                raise FileNotFoundError(
                    "Stocks record file was not found {}".format(
                        records_path / "stocks_record"
                    )
                )
            stocks = np.memmap(
                records_path / "stocks_record",
                mode="r+",
                dtype="float64",
                shape=(t * indexes["n_sectors"], indexes["n_industries"]),
            )
            stocks_treatment = True
        else:
            stocks = None
            stocks_treatment = False

        return cls(
            has_crashed=has_crashed,
            params=params,
            regions=regions,
            sectors=sectors,
            productive_capital=productive_capital,
            prod=prod,
            prodmax=prodmax,
            overprod=overprod,
            final_demand=final_demand,
            io_demand=io_demand,
            r_demand=r_demand,
            r_prod=r_prod,
            fd_unmet=fd_unmet,
            limiting_stocks=limiting_stocks,
            n_temporal_units_to_sim=n_temporal_units_to_sim,
            n_temporal_units_by_step=n_temporal_units_by_step,
            events=events,
            results_storage=results_storage,
            psi_param=psi,
            inventory_restoration_tau=inventory_restoration_tau,
            stocks=stocks,
            stocks_treatment=stocks_treatment,
            include_crash=include_crash,
        )

    def calc_top_failing_sect(self):
        pass

    def calc_tot_fd_unmet(self):
        self.indicators["tot_fd_unmet"] = self.treated_loss_df["fdloss"].sum()

    def calc_aff_fd_unmet(self):
        # TODO: check this
        self.indicators["aff_fd_unmet"] = self.treated_loss_df[
            self.treated_loss_df.region.isin(self.aff_regions)
        ]["fdloss"].sum()

    def calc_rebuild_durations(self):
        rebuilding = (
            self.rebuild_demand_df.reset_index()
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
        n_regions = self.limiting_inputs_df.columns.levels[0].size
        n_sectors = self.limiting_inputs_df.columns.levels[1].size
        # have only steps in index (next step not possible with multiindex)
        a = self.limiting_inputs_df.unstack()
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
        a = self.limiting_inputs_df.stack([0, 1])  # type: ignore
        a = a.swaplevel(1, 2).swaplevel(2, 3)
        b = a[a > 0]
        b = b[:10]
        res = list(b.index)  # type:ignore
        self.indicators["10_first_shortages_(step,region,sector,stock_of)"] = res

    def calc_tot_prod_change(self):
        df2 = self.production_realised_df.copy()
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
        self.prod_chg_region = pd.DataFrame({aff_regions: prod_chg_region}).T
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

    def write_indicators(self, storage_path=None):
        if storage_path is None and not hasattr(self, "storage_path"):
            raise ValueError(
                f"You are attempting to save indicators but no storage path was specified either in the Indicators class or to this method."
            )
        storage_path = self.storage_path if storage_path is None else storage_path
        storage_path = pathlib.Path(storage_path).resolve()
        logger.info("Writing indicators to json")
        if hasattr(self, "prod_chg_region"):
            self.prod_chg_region.to_json(
                storage_path / "prod_chg.json", indent=4, orient="split"
            )
        if hasattr(self, "fd_loss_region"):
            self.fd_loss_region.to_json(
                storage_path / "fd_loss.json", indent=4, orient="split"
            )
        with (storage_path / "indicators.json").open("w") as f:
            json.dump(self.indicators, f, cls=misc.CustomNumpyEncoder, indent=4)

    def calc_fd_loss_region(self):
        fd_loss = self.final_demand_unmet_df.copy().round(6)
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
        self.fd_loss_region = pd.DataFrame({aff_regions: fd_loss_region}).T
        fd_loss_sect = fd_loss.sum()
        tmp = fd_loss_sect.sort_values(ascending=False, key=abs).head(5).to_dict()
        self.indicators["top_5_sector_fdloss"] = {str(k): v for k, v in tmp.items()}

    def save_dfs(self):
        logger.info("Saving computed dataframe to results folder")
        self.production_realised_df.to_parquet(
            self.parquets_path / "production_realised_df.parquet"
        )
        self.productive_capital_to_recover_df.to_parquet(
            self.parquets_path / "productive_capital_to_recover_df.parquet"
        )
        self.production_capacity_df.to_parquet(
            self.parquets_path / "production_capacity_df.parquet"
        )
        self.overproduction_df.to_parquet(
            self.parquets_path / "overproduction_df.parquet"
        )
        self.final_demand_unmet_df.to_parquet(
            self.parquets_path / "final_demand_unmet_df.parquet"
        )
        self.final_demand_df.to_parquet(self.parquets_path / "final_demand_df.parquet")
        self.intermediate_demand_df.to_parquet(
            self.parquets_path / "intermediate_demand_df.parquet"
        )
        self.rebuild_demand_df.to_parquet(
            self.parquets_path / "rebuild_demand_df.parquet"
        )
        self.treated_loss_df.to_parquet(self.parquets_path / "treated_df_loss.parquet")
        if self.inputs_stocks_df is not None:
            ddf = da.from_pandas(self.inputs_stocks_df, chunksize=10000000)
            ddf.to_parquet(
                self.parquets_path / "treated_df_stocks.parquet", engine="pyarrow"
            )
        if self.limiting_inputs_df is not None:
            # ddf_l = da.from_pandas(self.df_limiting, chunksize=10000000)
            # ddf_l = ddf_l.melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'variable_2':'stock of'})
            # ddf_l = ddf_l.reset_index()
            # ddf_l['step'] = ddf_l['step'].astype("uint16")
            # ddf_l['stock of'] = ddf_l['stock of'].astype("category")
            # ddf_l['region'] = ddf_l['region'].astype("category")
            # ddf_l['sector'] = ddf_l['sector'].astype("category")
            self.limiting_inputs_df.to_parquet(
                self.parquets_path / "treated_df_limiting_inputs.parquet",
                engine="pyarrow",
            )
        # self.df_limiting.to_feather(self.parquets_path/"treated_df_limiting.feather")
