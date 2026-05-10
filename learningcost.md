cd /home/shiqi/masterarbeit/RHCR

```bash
./lifelong \
  -m maps/wfi_warehouse.map \
  --scenario=KIVA \
  -k 192 \
  --simulation_window=5 --planning_window=20 \
  --solver=ECBS \
  --suboptimal_bound=1.5 \
  --dummy_paths=false \
  --seed=0 \
  --use_learned_cost=true \
  --learned_cost_ckpt ../learn-to-follow/costmap/outputs/warehouse_outputs/20260424_133433/best_model.pt \
  --learned_cost_params_json ../learn-to-follow/costmap/cmaes_hparams_runs/wfi_warehouse/20260425_100543/best_candidate.json \
  --simulation_time=100 \
  -o ../exp/wfi_learned_seed0
```

```bash
cd /home/shiqi/masterarbeit/RHCR

./lifelong \
  -m maps/wfi_warehouse.map \
  --scenario=KIVA \
  -k 192 \
  --simulation_window=1\
  --solver=ECBS \
  --suboptimal_bound=1.5 \
  --dummy_paths=true \
  --seed=0 \
  --simulation_time=100 \
  -o ../exp/wfi_baseline_seed0
```


./lifelong \
  -m maps/wfi_warehouse.map \
  --scenario=KIVA \
  -k 192 \
  --simulation_window=5 --planning_window=20 \
  --solver=PBS \
  --dummy_paths=false \
  --seed=0 \
  --use_learned_cost=true \
  --learned_cost_ckpt ../learn-to-follow/costmap/outputs/warehouse_outputs/20260424_133433/best_model.pt \
  --learned_cost_params_json ../learn-to-follow/costmap/cmaes_hparams_runs/wfi_warehouse/20260425_100543/best_candidate.json \
  --simulation_time=100 \
  -o ../exp/wfi_learned_seed0

./lifelong \
  -m maps/wfi_warehouse.map \
  --scenario=KIVA \
  -k 192 \
  --simulation_window=5 --planning_window=20 \
  --solver=PBS \
  --dummy_paths=false \
  --seed=0 \
  --use_learned_cost=true \
  --learned_cost_ckpt ../learn-to-follow/costmap/outputs/warehouse_outputs/20260424_133433/best_model.pt \
  --learned_cost_weight=1.0 \
  --gaussian_sigma=0.8 \
  --pred_bias=0.05 \
  --simulation_time=100 \
  -o ../exp/wfi_learned_seed0


cd /home/shiqi/masterarbeit/RHCR

./lifelong \
  -m maps/wfi_warehouse.map \
  --scenario=KIVA \
  -k 192 \
  --simulation_window=5 --planning_window=20 \
  --solver=PBS \
  --dummy_paths=false \
  --seed=0 \
  --simulation_time=100 \
  -o ../exp/wfi_baseline_seed0
