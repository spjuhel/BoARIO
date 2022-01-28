"""
Default settings and initialisation of the app
"""

####################
# Default settings #
####################
import json
import pathlib

WORKING_DIR = pathlib.Path.cwd()

def load_defaults(default_storage_dir):
    default_storage_dir = WORKING_DIR/default_storage_dir
    default_storage_dir.mkdir(parents=True, exist_ok=True)

    if (default_storage_dir/"default_params.json").exists():
        with (default_storage_dir/"default_params.json").open('r') as fp:
            default_params=json.load(fp)
    else:
        default_params = {
            "is_default": True,
            "storage_dir": str(default_storage_dir),
            "results_storage": str(default_storage_dir/"/results/"),
            "bool_run_detailled": True,
            "psi_param": 0.8,
            "model_time_step": 1,
            "timestep_dividing_factor": 365,
            "inventory_restoration_time": 40,
            "alpha_base": 1.0,
            "alpha_max": 1.25,
            "alpha_tau": 365,
            "rebuild_tau": 30,
            "n_timesteps": 365,
            "min_duration": (365 // 100) * 25,
            "impacted_region_base_production_toward_rebuilding": 0.01,
            "row_base_production_toward_rebuilding": 0.0,
            "mrio_params_file":str(default_storage_dir/"default_mrio_params.json")
        }
        with (default_storage_dir/"default_params.json").open('w') as fp:
            json.dump(default_params,fp)

    current_params = default_params

    if (pathlib.Path(default_params["storage_dir"])/"indexes"/"indexes.json").exists():
        with (pathlib.Path(default_params["storage_dir"])/"indexes"/"indexes.json").open('r') as fp:
            indexes = json.load(fp)
    else:
        indexes = {
            "n_industries": 48,
            "n_regions": 6,
            "n_sectors": 8,
            "fd_cat": [
                "Changes in inventories",
                "Changes in valuables",
                "Export",
                "Final consumption expenditure by government",
                "Final consumption expenditure by households",
                "Final consumption expenditure by non-profit organisations serving households (NPISH)",
                "Gross fixed capital formation"
            ],
            "sectors": [
                "construction",
                "electricity",
                "food",
                "manufactoring",
                "mining",
                "other",
                "trade",
                "transport"
            ],
            "regions": [
                "reg1",
                "reg2",
                "reg3",
                "reg4",
                "reg5",
                "reg6"
            ],
            "is_default": True
        }
        (pathlib.Path(default_params["storage_dir"])/"indexes"/"indexes.json").parent.mkdir(parents=True, exist_ok=True)
        with (pathlib.Path(default_params["storage_dir"])/"indexes"/"indexes.json").open("w") as f:
            json.dump(indexes, f)

    if pathlib.Path(default_params["mrio_params_file"]).exists():
        with pathlib.Path(default_params["mrio_params_file"]).open('r') as fp:
            default_mrio_params = json.load(fp)
    else:
        default_mrio_params = {"is_default": True,
                               "monetary_unit": 1,
                               "exio_idx": {
                                   "sectors_for_rebuild_idx": {
                                       "construction": 0.30,
                                       "manufactoring": 0.70
                                   },
                                   "kapital_VA_idx": "Operating surplus: Consumption of fixed capital",
                                   "labor_VA_idx": "Compensation of employees"
                               },
                               "capital_ratio_dict": {
                                   "food": 2.9,
                                   "mining": 5.1,
                                   "manufactoring": 1.4,
                                   "electricity": 3,
                                   "construction": 0.5,
                                   "trade": 0.6,
                                   "transport": 2.5,
                                   "other": 1.2
                               },
                               "inventories_dict": {
                                   "food": 30,
                                   "mining": 30,
                                   "manufactoring": 30,
                                   "electricity": 7,
                                   "construction": 30,
                                   "trade": 30,
                                   "transport": 10,
                                   "other": 30
                               }}
        with pathlib.Path(default_params["mrio_params_file"]).open('w') as fp:
            json.dump(default_mrio_params, fp)

    current_mrio_params=default_mrio_params

    if (pathlib.Path(default_params["storage_dir"])/"params.json").exists():
        with (pathlib.Path(default_params["storage_dir"])/"params.json").open('r') as fp:
            current_params = json.load(fp)

    if (pathlib.Path(default_params["storage_dir"])/"mrio_params.json").exists():
        with (pathlib.Path(default_params["storage_dir"])/"mrio_params.json").open('r') as fp:
            current_mrio_params = json.load(fp)

    return current_params, current_mrio_params, indexes
    #print(WORKING_DIR)
    #print(current_params)
    #print(current_mrio_params)
    #print(indexes)
