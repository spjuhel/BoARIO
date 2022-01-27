import ario3.simulation.base as sim
import pathlib
import numpy as np
mrio_path = pathlib.Path("./dash-testing/mrio.pkl")
import pathlib
import json
import numpy as np
cwd = pathlib.Path().cwd()
dash_ressources = cwd / "dash-testing"

with (dash_ressources/"params.json").open('r') as fp:
    params = json.load(fp)
#ev_file = pathlib.Path('./ario3/test/mock_data/test_events3.json')
#params = {"alpha_base":1,"alpha_max":1.25,"alpha_tau":365,"bool_run_detailled":True,"exio_params_file":"dash-testing/mrio_params.json","impacted_region_base_production_toward_rebuilding":0.01,"inventory_restoration_time":20,"min_duration":300,"model_time_step":1,"n_timesteps":1200,"psi_param":0.8,"rebuild_tau":30,"results_storage":"dash-testing/results/","row_base_production_toward_rebuilding":0,"storage_dir":"dash-testing/","timestep_dividing_factor":365}
#events = [{"aff-regions":["reg1"],"aff-sectors":["construction"],"dmg-distrib-regions":[1],"dmg-distrib-sectors":[1],"duration":1,"name":"0","occur":0,"q_dmg":150000,"rebuilding-sectors":{"construction":1}}]

events = [{'aff-regions': ['reg1'], 'aff-sectors': ['construction', 'mining'], 'dmg-distrib-regions': [1], 'dmg-distrib-sectors': [0.5, 0.5], 'duration': 1, 'name': '0', 'occur': 5, 'q_dmg': 200000, 'rebuilding-sectors': {'construction': 0.5, 'manufactoring': 0.5}}]


#model = sim.Simulation(mrio_path, params)
#model.read_events_from_list(events)
#model.loop()
