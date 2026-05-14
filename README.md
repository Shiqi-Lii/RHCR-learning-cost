# RHCR
![test_ubuntu](https://github.com/Jiaoyang-Li/RHCR/actions/workflows/test_ubuntu.yml/badge.svg)
![test_macos](https://github.com/Jiaoyang-Li/RHCR/actions/workflows/test_macos.yml/badge.svg)

Rolling-Horizon Collision Resolution (RHCR) is an efficient algorithm for solving lifelong Multi-Agent Path Finding (MAPF) where we are aksed to plan collision-free paths for a large number of agents that are constanly engaged with new goal locations. RHCR calls a Windowed MAPF solver every h timesteps that resolves collisions only for the next w timesteps (w >= h). More details can be found in our extended abstract at AAMAS 2020 [1] and our full paper at AAAI 2021 [2].

The code requires the external library BOOST (https://www.boost.org/).    
Here is an easy way of installing BOOST in Linux:
```shell script
sudo apt install libboost-all-dev
```

After you installed BOOST and downloaded the source code, go into the directory of the source code and compile it with CMake: 
```
cmake .
make
```

If you clone this repository on a new machine, install the dependencies first:
```shell script
sudo apt update
sudo apt install build-essential cmake libboost-all-dev
```

Then compile the code:
```
cd RHCR
cmake .
make -j
```

You can check whether the executable is available with:
```
./lifelong --help
```

Then, you are able to run the code:
```
./lifelong -m maps/sorting_map.grid -k 800 --scenario=SORTING --simulation_window=5 --planning_window=10 --solver=PBS --seed=0
```
for running RHCR with PBS on the sorting center map; and
```
./lifelong -m maps/kiva.map -k 100 --scenario=KIVA --simulation_window=1 --solver=ECBS --suboptimal_bound=1.5 --dummy_path=1 --seed=0
```
for running ECBS(w=1.5) with dummy paths on the kiva map.

- m: the map file 
- k: the number of agents
- scenario: the simulation scenario (each scenario corresponding to a different task assigner). Use KIVA for the fulfillment warehouse scenario and SORTING for the sorting center scenario. 
- simulation_window: the replanning period h
- planning_window: the planning window w
- solver: the windowed MAPF solver (WHCA, ECBS, and PBS)
- seed: the random seed

You can find more details and explanations for all parameters with:
```
./lifelong --help
```

### Learned cost extension

This repository also supports running RHCR with a learned cost map. The C++ solver runs on CPU, while the learned cost model is evaluated by the Python bridge in `scripts/learned_cost_infer.py`.

To use learned cost, make sure the Python environment contains PyTorch and NumPy. The current bridge expects `learn-to-follow` to be located next to this repository, for example:
```
masterarbeit/
  RHCR/
  learn-to-follow/
```

Example command:
```
./lifelong \
  -m maps/wfi_warehouse.map \
  --scenario=KIVA \
  -k 128 \
  --simulation_window=1 \
  --planning_window=5 \
  --solver=PBS \
  --seed=10 \
  --simulation_time=256 \
  --dummy_paths=false \
  --use_learned_cost=true \
  --learned_cost_ckpt ../learn-to-follow/costmap/outputs/warehouse_outputs/20260424_133433/best_model.pt \
  --learned_cost_weight=1.0 \
  --gaussian_sigma=0.8 \
  --pred_bias=0.05 \
  --gaussian_ksize=5
```

If the heuristics table for a map does not exist, RHCR generates it during preprocessing. For large maps, this first run can take extra time.

## License
RHCR is released under USC – Research License. See license.md for further details.
 
## References
[1] Jiaoyang Li, Andrew Tinka, Scott Kiesel, Joseph W. Durham, T. K. Satish Kumar and Sven Koenig. Lifelong Multi-Agent Path Finding in Large-Scale Warehouses (extended abstract). In Proceedings of the International Joint Conference on Autonomous Agents and Multiagent Systems (AAMAS), pages 1898-1900, 2020.

[2] Jiaoyang Li, Andrew Tinka, Scott Kiesel, Joseph W. Durham, T. K. Satish Kumar and Sven Koenig. Lifelong Multi-Agent Path Finding in Large-Scale Warehouses. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), (in print), 2021.
