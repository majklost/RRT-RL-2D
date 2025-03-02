# Details

Date : 2025-03-01 14:07:38

Directory /home/michal/Documents/Skola/bakalarka/RRT-RL-2D

Total : 116 files,  5339 codes, 206 comments, 1444 blanks, all 6989 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [README.md](/README.md) | Markdown | 2 | 0 | 1 | 3 |
| [bash\_scripts/autofetch.sh](/bash_scripts/autofetch.sh) | Shell Script | 9 | 5 | 4 | 18 |
| [bash\_scripts/conn\_tensorboard.sh](/bash_scripts/conn_tensorboard.sh) | Shell Script | 6 | 2 | 2 | 10 |
| [bash\_scripts/fetch\_results.sh](/bash_scripts/fetch_results.sh) | Shell Script | 6 | 3 | 1 | 10 |
| [bash\_scripts/loading.sh](/bash_scripts/loading.sh) | Shell Script | 2 | 1 | 2 | 5 |
| [bash\_scripts/on\_open.sh](/bash_scripts/on_open.sh) | Shell Script | 13 | 11 | 5 | 29 |
| [bash\_scripts/remote\_submit.sh](/bash_scripts/remote_submit.sh) | Shell Script | 3 | 0 | 1 | 4 |
| [bash\_scripts/submit.sh](/bash_scripts/submit.sh) | Shell Script | 4 | 3 | 4 | 11 |
| [bash\_scripts/to\_remote.sh](/bash_scripts/to_remote.sh) | Shell Script | 9 | 3 | 2 | 14 |
| [pyproject.toml](/pyproject.toml) | toml | 3 | 0 | 1 | 4 |
| [rrt\_rl\_2D/CLIENT/\_\_init\_\_.py](/rrt_rl_2D/CLIENT/__init__.py) | Python | 8 | 0 | 1 | 9 |
| [rrt\_rl\_2D/CLIENT/analyzable/analyzable.py](/rrt_rl_2D/CLIENT/analyzable/analyzable.py) | Python | 100 | 0 | 22 | 122 |
| [rrt\_rl\_2D/CLIENT/makers/\_\_init\_\_.py](/rrt_rl_2D/CLIENT/makers/__init__.py) | Python | 3 | 2 | 3 | 8 |
| [rrt\_rl\_2D/CLIENT/makers/makers.py](/rrt_rl_2D/CLIENT/makers/makers.py) | Python | 196 | 0 | 71 | 267 |
| [rrt\_rl\_2D/MPI/MPI.py](/rrt_rl_2D/MPI/MPI.py) | Python | 7 | 0 | 3 | 10 |
| [rrt\_rl\_2D/RL/\_\_init\_\_.py](/rrt_rl_2D/RL/__init__.py) | Python | 2 | 0 | 1 | 3 |
| [rrt\_rl\_2D/RL/players.py](/rrt_rl_2D/RL/players.py) | Python | 45 | 3 | 8 | 56 |
| [rrt\_rl\_2D/RL/training\_utils.py](/rrt_rl_2D/RL/training_utils.py) | Python | 116 | 1 | 28 | 145 |
| [rrt\_rl\_2D/\_\_init\_\_.py](/rrt_rl_2D/__init__.py) | Python | 12 | 0 | 1 | 13 |
| [rrt\_rl\_2D/analytics/\_\_init\_\_.py](/rrt_rl_2D/analytics/__init__.py) | Python | 1 | 0 | 1 | 2 |
| [rrt\_rl\_2D/analytics/standard\_analytics.py](/rrt_rl_2D/analytics/standard_analytics.py) | Python | 38 | 0 | 7 | 45 |
| [rrt\_rl\_2D/assets/\_\_init\_\_.py](/rrt_rl_2D/assets/__init__.py) | Python | 7 | 0 | 2 | 9 |
| [rrt\_rl\_2D/assets/boundings.py](/rrt_rl_2D/assets/boundings.py) | Python | 19 | 0 | 5 | 24 |
| [rrt\_rl\_2D/assets/cable.py](/rrt_rl_2D/assets/cable.py) | Python | 113 | 6 | 23 | 142 |
| [rrt\_rl\_2D/assets/multibody.py](/rrt_rl_2D/assets/multibody.py) | Python | 145 | 6 | 33 | 184 |
| [rrt\_rl\_2D/assets/random\_block.py](/rrt_rl_2D/assets/random_block.py) | Python | 18 | 1 | 5 | 24 |
| [rrt\_rl\_2D/assets/random\_obstacle\_group.py](/rrt_rl_2D/assets/random_obstacle_group.py) | Python | 36 | 0 | 9 | 45 |
| [rrt\_rl\_2D/assets/rectangle.py](/rrt_rl_2D/assets/rectangle.py) | Python | 19 | 0 | 5 | 24 |
| [rrt\_rl\_2D/assets/singlebody.py](/rrt_rl_2D/assets/singlebody.py) | Python | 146 | 2 | 39 | 187 |
| [rrt\_rl\_2D/controllers/cable\_controller.py](/rrt_rl_2D/controllers/cable_controller.py) | Python | 54 | 0 | 13 | 67 |
| [rrt\_rl\_2D/controllers/direct\_controller.py](/rrt_rl_2D/controllers/direct_controller.py) | Python | 11 | 0 | 4 | 15 |
| [rrt\_rl\_2D/controllers/env\_controller.py](/rrt_rl_2D/controllers/env_controller.py) | Python | 72 | 0 | 19 | 91 |
| [rrt\_rl\_2D/controllers/rect\_controller.py](/rrt_rl_2D/controllers/rect_controller.py) | Python | 29 | 0 | 7 | 36 |
| [rrt\_rl\_2D/envs/\_\_init\_\_.py](/rrt_rl_2D/envs/__init__.py) | Python | 4 | 0 | 1 | 5 |
| [rrt\_rl\_2D/envs/blend\_env.py](/rrt_rl_2D/envs/blend_env.py) | Python | 45 | 1 | 20 | 66 |
| [rrt\_rl\_2D/envs/cable\_env.py](/rrt_rl_2D/envs/cable_env.py) | Python | 138 | 0 | 42 | 180 |
| [rrt\_rl\_2D/envs/cable\_radius.py](/rrt_rl_2D/envs/cable_radius.py) | Python | 84 | 1 | 36 | 121 |
| [rrt\_rl\_2D/envs/debug\_radius.py](/rrt_rl_2D/envs/debug_radius.py) | Python | 131 | 3 | 36 | 170 |
| [rrt\_rl\_2D/envs/dodge\_env.py](/rrt_rl_2D/envs/dodge_env.py) | Python | 0 | 0 | 1 | 1 |
| [rrt\_rl\_2D/envs/rect.py](/rrt_rl_2D/envs/rect.py) | Python | 114 | 0 | 34 | 148 |
| [rrt\_rl\_2D/envs/rrt\_env.py](/rrt_rl_2D/envs/rrt_env.py) | Python | 128 | 2 | 40 | 170 |
| [rrt\_rl\_2D/export/\_\_init\_\_.py](/rrt_rl_2D/export/__init__.py) | Python | 2 | 0 | 1 | 3 |
| [rrt\_rl\_2D/export/vel\_path\_replayer.py](/rrt_rl_2D/export/vel_path_replayer.py) | Python | 41 | 3 | 9 | 53 |
| [rrt\_rl\_2D/export/vel\_path\_saver.py](/rrt_rl_2D/export/vel_path_saver.py) | Python | 55 | 0 | 11 | 66 |
| [rrt\_rl\_2D/importable/\_\_init\_\_.py](/rrt_rl_2D/importable/__init__.py) | Python | 1 | 0 | 1 | 2 |
| [rrt\_rl\_2D/importable/saved\_path\_replayer.py](/rrt_rl_2D/importable/saved_path_replayer.py) | Python | 83 | 4 | 15 | 102 |
| [rrt\_rl\_2D/manual\_models/\_\_init\_\_.py](/rrt_rl_2D/manual_models/__init__.py) | Python | 2 | 0 | 2 | 4 |
| [rrt\_rl\_2D/manual\_models/base\_model.py](/rrt_rl_2D/manual_models/base_model.py) | Python | 7 | 0 | 5 | 12 |
| [rrt\_rl\_2D/maps/\_\_init\_\_.py](/rrt_rl_2D/maps/__init__.py) | Python | 16 | 0 | 4 | 20 |
| [rrt\_rl\_2D/maps/empty.py](/rrt_rl_2D/maps/empty.py) | Python | 120 | 2 | 32 | 154 |
| [rrt\_rl\_2D/maps/non\_convex.py](/rrt_rl_2D/maps/non_convex.py) | Python | 31 | 0 | 9 | 40 |
| [rrt\_rl\_2D/maps/piped.py](/rrt_rl_2D/maps/piped.py) | Python | 61 | 0 | 9 | 70 |
| [rrt\_rl\_2D/maps/stones.py](/rrt_rl_2D/maps/stones.py) | Python | 30 | 0 | 12 | 42 |
| [rrt\_rl\_2D/node\_managers/\_\_init\_\_.py](/rrt_rl_2D/node_managers/__init__.py) | Python | 4 | 0 | 2 | 6 |
| [rrt\_rl\_2D/node\_managers/controllable\_manager.py](/rrt_rl_2D/node_managers/controllable_manager.py) | Python | 10 | 0 | 4 | 14 |
| [rrt\_rl\_2D/node\_managers/node\_manager.py](/rrt_rl_2D/node_managers/node_manager.py) | Python | 25 | 0 | 9 | 34 |
| [rrt\_rl\_2D/node\_managers/vel\_node\_manager.py](/rrt_rl_2D/node_managers/vel_node_manager.py) | Python | 20 | 0 | 6 | 26 |
| [rrt\_rl\_2D/nodes/\_\_init\_\_.py](/rrt_rl_2D/nodes/__init__.py) | Python | 3 | 0 | 2 | 5 |
| [rrt\_rl\_2D/nodes/goal\_node.py](/rrt_rl_2D/nodes/goal_node.py) | Python | 11 | 1 | 4 | 16 |
| [rrt\_rl\_2D/nodes/node.py](/rrt_rl_2D/nodes/node.py) | Python | 2 | 0 | 1 | 3 |
| [rrt\_rl\_2D/nodes/tree\_node.py](/rrt_rl_2D/nodes/tree_node.py) | Python | 21 | 0 | 8 | 29 |
| [rrt\_rl\_2D/planners/\_\_init\_\_.py](/rrt_rl_2D/planners/__init__.py) | Python | 4 | 0 | 2 | 6 |
| [rrt\_rl\_2D/planners/base\_planner.py](/rrt_rl_2D/planners/base_planner.py) | Python | 26 | 0 | 7 | 33 |
| [rrt\_rl\_2D/planners/learnable\_planner.py](/rrt_rl_2D/planners/learnable_planner.py) | Python | 7 | 0 | 4 | 11 |
| [rrt\_rl\_2D/planners/rl\_planner.py](/rrt_rl_2D/planners/rl_planner.py) | Python | 14 | 0 | 5 | 19 |
| [rrt\_rl\_2D/planners/vec\_env\_planner.py](/rrt_rl_2D/planners/vec_env_planner.py) | Python | 66 | 2 | 18 | 86 |
| [rrt\_rl\_2D/rendering/\_\_init\_\_.py](/rrt_rl_2D/rendering/__init__.py) | Python | 5 | 0 | 2 | 7 |
| [rrt\_rl\_2D/rendering/base\_renderer.py](/rrt_rl_2D/rendering/base_renderer.py) | Python | 19 | 0 | 5 | 24 |
| [rrt\_rl\_2D/rendering/debug\_renderer.py](/rrt_rl_2D/rendering/debug_renderer.py) | Python | 53 | 0 | 11 | 64 |
| [rrt\_rl\_2D/rendering/env\_renderer.py](/rrt_rl_2D/rendering/env_renderer.py) | Python | 42 | 0 | 11 | 53 |
| [rrt\_rl\_2D/rendering/null\_renderer.py](/rrt_rl_2D/rendering/null_renderer.py) | Python | 11 | 0 | 5 | 16 |
| [rrt\_rl\_2D/samplers/\_\_init\_\_.py](/rrt_rl_2D/samplers/__init__.py) | Python | 3 | 0 | 2 | 5 |
| [rrt\_rl\_2D/samplers/base\_sampler.py](/rrt_rl_2D/samplers/base_sampler.py) | Python | 8 | 0 | 5 | 13 |
| [rrt\_rl\_2D/samplers/bezier\_sampler.py](/rrt_rl_2D/samplers/bezier_sampler.py) | Python | 123 | 2 | 22 | 147 |
| [rrt\_rl\_2D/samplers/ndim\_sampler.py](/rrt_rl_2D/samplers/ndim_sampler.py) | Python | 15 | 0 | 5 | 20 |
| [rrt\_rl\_2D/simulator/collision\_data.py](/rrt_rl_2D/simulator/collision_data.py) | Python | 12 | 0 | 5 | 17 |
| [rrt\_rl\_2D/simulator/simulator.py](/rrt_rl_2D/simulator/simulator.py) | Python | 161 | 6 | 39 | 206 |
| [rrt\_rl\_2D/simulator/standard\_config.py](/rrt_rl_2D/simulator/standard_config.py) | Python | 30 | 2 | 5 | 37 |
| [rrt\_rl\_2D/storage\_wrappers/\_\_init\_\_.py](/rrt_rl_2D/storage_wrappers/__init__.py) | Python | 2 | 0 | 1 | 3 |
| [rrt\_rl\_2D/storage\_wrappers/base\_wrapper.py](/rrt_rl_2D/storage_wrappers/base_wrapper.py) | Python | 24 | 0 | 9 | 33 |
| [rrt\_rl\_2D/storage\_wrappers/rect\_end\_wrapper.py](/rrt_rl_2D/storage_wrappers/rect_end_wrapper.py) | Python | 17 | 1 | 4 | 22 |
| [rrt\_rl\_2D/storage\_wrappers/standard\_wrapper.py](/rrt_rl_2D/storage_wrappers/standard_wrapper.py) | Python | 54 | 1 | 12 | 67 |
| [rrt\_rl\_2D/storages/GNAT.py](/rrt_rl_2D/storages/GNAT.py) | Python | 143 | 22 | 32 | 197 |
| [rrt\_rl\_2D/storages/\_\_init\_\_.py](/rrt_rl_2D/storages/__init__.py) | Python | 5 | 0 | 2 | 7 |
| [rrt\_rl\_2D/storages/base\_storage.py](/rrt_rl_2D/storages/base_storage.py) | Python | 13 | 0 | 6 | 19 |
| [rrt\_rl\_2D/storages/brute\_force.py](/rrt_rl_2D/storages/brute_force.py) | Python | 19 | 0 | 6 | 25 |
| [rrt\_rl\_2D/storages/kd\_tree.py](/rrt_rl_2D/storages/kd_tree.py) | Python | 95 | 32 | 35 | 162 |
| [rrt\_rl\_2D/utils/common\_utils.py](/rrt_rl_2D/utils/common_utils.py) | Python | 7 | 0 | 7 | 14 |
| [rrt\_rl\_2D/utils/save\_manager.py](/rrt_rl_2D/utils/save_manager.py) | Python | 352 | 4 | 83 | 439 |
| [rrt\_rl\_2D/utils/seed\_manager.py](/rrt_rl_2D/utils/seed_manager.py) | Python | 32 | 0 | 10 | 42 |
| [scripts/RL/cable-PID.py](/scripts/RL/cable-PID.py) | Python | 32 | 0 | 10 | 42 |
| [scripts/RL/cable-blend.py](/scripts/RL/cable-blend.py) | Python | 29 | 0 | 10 | 39 |
| [scripts/RL/cable-radius.py](/scripts/RL/cable-radius.py) | Python | 94 | 3 | 26 | 123 |
| [scripts/RL/init\_hyperopt.py](/scripts/RL/init_hyperopt.py) | Python | 0 | 0 | 1 | 1 |
| [scripts/RL/play\_experiment.py](/scripts/RL/play_experiment.py) | Python | 51 | 0 | 15 | 66 |
| [scripts/debug/\_\_init\_\_.py](/scripts/debug/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [scripts/debug/outer\_model.py](/scripts/debug/outer_model.py) | Python | 101 | 13 | 44 | 158 |
| [scripts/debug/t1.py](/scripts/debug/t1.py) | Python | 3 | 0 | 1 | 4 |
| [scripts/debug/t2.py](/scripts/debug/t2.py) | Python | 12 | 0 | 5 | 17 |
| [scripts/development/cablePID.py](/scripts/development/cablePID.py) | Python | 21 | 1 | 5 | 27 |
| [scripts/development/cable\_resetable.py](/scripts/development/cable_resetable.py) | Python | 18 | 0 | 10 | 28 |
| [scripts/development/dirty.ipynb](/scripts/development/dirty.ipynb) | JSON | 365 | 0 | 1 | 366 |
| [scripts/development/playground.py](/scripts/development/playground.py) | Python | 11 | 0 | 8 | 19 |
| [scripts/development/rect\_env.py](/scripts/development/rect_env.py) | Python | 32 | 6 | 14 | 52 |
| [scripts/development/rrt\_rect.py](/scripts/development/rrt_rect.py) | Python | 76 | 2 | 30 | 108 |
| [scripts/development/test\_env.py](/scripts/development/test_env.py) | Python | 30 | 14 | 21 | 65 |
| [scripts/development/test\_maps.py](/scripts/development/test_maps.py) | Python | 38 | 2 | 14 | 54 |
| [scripts/multitool.ipynb](/scripts/multitool.ipynb) | JSON | 24 | 0 | 1 | 25 |
| [scripts/playground/PID.py](/scripts/playground/PID.py) | Python | 33 | 2 | 15 | 50 |
| [scripts/playground/PID2.py](/scripts/playground/PID2.py) | Python | 35 | 6 | 16 | 57 |
| [scripts/quantitative/qunatitative.py](/scripts/quantitative/qunatitative.py) | Python | 244 | 1 | 47 | 292 |
| [scripts/rrt\_blender.py](/scripts/rrt_blender.py) | Python | 98 | 6 | 39 | 143 |
| [scripts/rrt\_cable.py](/scripts/rrt_cable.py) | Python | 87 | 9 | 30 | 126 |
| [scripts/rrt\_cable\_radius.py](/scripts/rrt_cable_radius.py) | Python | 81 | 3 | 29 | 113 |
| [scripts/rrt\_plan.py](/scripts/rrt_plan.py) | Python | 0 | 0 | 1 | 1 |
| [setup.py](/setup.py) | Python | 19 | 0 | 1 | 20 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)