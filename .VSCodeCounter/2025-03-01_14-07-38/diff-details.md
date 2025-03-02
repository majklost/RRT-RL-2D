# Diff Details

Date : 2025-03-01 14:07:38

Directory /home/michal/Documents/Skola/bakalarka/RRT-RL-2D

Total : 72 files,  2701 codes, 99 comments, 681 blanks, all 3481 lines

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [bash\_scripts/autofetch.sh](/bash_scripts/autofetch.sh) | Shell Script | 9 | 5 | 4 | 18 |
| [bash\_scripts/conn\_tensorboard.sh](/bash_scripts/conn_tensorboard.sh) | Shell Script | 6 | 2 | 2 | 10 |
| [bash\_scripts/fetch\_results.sh](/bash_scripts/fetch_results.sh) | Shell Script | 6 | 3 | 1 | 10 |
| [bash\_scripts/loading.sh](/bash_scripts/loading.sh) | Shell Script | 2 | 1 | 2 | 5 |
| [bash\_scripts/on\_open.sh](/bash_scripts/on_open.sh) | Shell Script | 13 | 11 | 5 | 29 |
| [bash\_scripts/remote\_submit.sh](/bash_scripts/remote_submit.sh) | Shell Script | 3 | 0 | 1 | 4 |
| [bash\_scripts/submit.sh](/bash_scripts/submit.sh) | Shell Script | 4 | 3 | 4 | 11 |
| [bash\_scripts/to\_remote.sh](/bash_scripts/to_remote.sh) | Shell Script | 9 | 3 | 2 | 14 |
| [rrt\_rl\_2D/CLIENT/\_\_init\_\_.py](/rrt_rl_2D/CLIENT/__init__.py) | Python | 8 | 0 | 1 | 9 |
| [rrt\_rl\_2D/CLIENT/analyzable/analyzable.py](/rrt_rl_2D/CLIENT/analyzable/analyzable.py) | Python | 100 | 0 | 22 | 122 |
| [rrt\_rl\_2D/CLIENT/makers/\_\_init\_\_.py](/rrt_rl_2D/CLIENT/makers/__init__.py) | Python | 3 | 2 | 3 | 8 |
| [rrt\_rl\_2D/CLIENT/makers/makers.py](/rrt_rl_2D/CLIENT/makers/makers.py) | Python | 196 | 0 | 71 | 267 |
| [rrt\_rl\_2D/RL/\_\_init\_\_.py](/rrt_rl_2D/RL/__init__.py) | Python | 2 | 0 | 1 | 3 |
| [rrt\_rl\_2D/RL/players.py](/rrt_rl_2D/RL/players.py) | Python | 45 | 3 | 8 | 56 |
| [rrt\_rl\_2D/RL/training\_utils.py](/rrt_rl_2D/RL/training_utils.py) | Python | 116 | 1 | 28 | 145 |
| [rrt\_rl\_2D/\_\_init\_\_.py](/rrt_rl_2D/__init__.py) | Python | 3 | 0 | 0 | 3 |
| [rrt\_rl\_2D/analytics/\_\_init\_\_.py](/rrt_rl_2D/analytics/__init__.py) | Python | 1 | 0 | 1 | 2 |
| [rrt\_rl\_2D/analytics/analytics\_formatter.py](/rrt_rl_2D/analytics/analytics_formatter.py) | Python | 0 | 0 | -1 | -1 |
| [rrt\_rl\_2D/analytics/standard\_analytics.py](/rrt_rl_2D/analytics/standard_analytics.py) | Python | 38 | 0 | 7 | 45 |
| [rrt\_rl\_2D/assets/cable.py](/rrt_rl_2D/assets/cable.py) | Python | 4 | 0 | 1 | 5 |
| [rrt\_rl\_2D/envs/\_\_init\_\_.py](/rrt_rl_2D/envs/__init__.py) | Python | 4 | 0 | 1 | 5 |
| [rrt\_rl\_2D/envs/blend\_env.py](/rrt_rl_2D/envs/blend_env.py) | Python | 45 | 1 | 20 | 66 |
| [rrt\_rl\_2D/envs/cable\_env.py](/rrt_rl_2D/envs/cable_env.py) | Python | 138 | 0 | 42 | 180 |
| [rrt\_rl\_2D/envs/cable\_radius.py](/rrt_rl_2D/envs/cable_radius.py) | Python | -20 | 1 | 4 | -15 |
| [rrt\_rl\_2D/envs/debug\_radius.py](/rrt_rl_2D/envs/debug_radius.py) | Python | 131 | 3 | 36 | 170 |
| [rrt\_rl\_2D/envs/dodge\_env.py](/rrt_rl_2D/envs/dodge_env.py) | Python | 0 | 0 | 1 | 1 |
| [rrt\_rl\_2D/envs/rect.py](/rrt_rl_2D/envs/rect.py) | Python | 28 | 0 | 8 | 36 |
| [rrt\_rl\_2D/envs/rrt\_env.py](/rrt_rl_2D/envs/rrt_env.py) | Python | -5 | 0 | 4 | -1 |
| [rrt\_rl\_2D/export/\_\_init\_\_.py](/rrt_rl_2D/export/__init__.py) | Python | 2 | 0 | 1 | 3 |
| [rrt\_rl\_2D/export/vel\_path\_replayer.py](/rrt_rl_2D/export/vel_path_replayer.py) | Python | 4 | 0 | -1 | 3 |
| [rrt\_rl\_2D/export/vel\_path\_saver.py](/rrt_rl_2D/export/vel_path_saver.py) | Python | 55 | 0 | 11 | 66 |
| [rrt\_rl\_2D/import/policy\_loader.py](/rrt_rl_2D/import/policy_loader.py) | Python | -5 | 0 | -1 | -6 |
| [rrt\_rl\_2D/importable/\_\_init\_\_.py](/rrt_rl_2D/importable/__init__.py) | Python | 1 | 0 | 1 | 2 |
| [rrt\_rl\_2D/importable/saved\_path\_replayer.py](/rrt_rl_2D/importable/saved_path_replayer.py) | Python | 83 | 4 | 15 | 102 |
| [rrt\_rl\_2D/maps/\_\_init\_\_.py](/rrt_rl_2D/maps/__init__.py) | Python | 10 | 0 | 2 | 12 |
| [rrt\_rl\_2D/node\_managers/\_\_init\_\_.py](/rrt_rl_2D/node_managers/__init__.py) | Python | 1 | 0 | 0 | 1 |
| [rrt\_rl\_2D/node\_managers/controllable\_manager.py](/rrt_rl_2D/node_managers/controllable_manager.py) | Python | 10 | 0 | 4 | 14 |
| [rrt\_rl\_2D/node\_managers/node\_manager.py](/rrt_rl_2D/node_managers/node_manager.py) | Python | 1 | 0 | 0 | 1 |
| [rrt\_rl\_2D/node\_managers/vel\_node\_manager.py](/rrt_rl_2D/node_managers/vel_node_manager.py) | Python | 1 | 0 | 0 | 1 |
| [rrt\_rl\_2D/nodes/goal\_node.py](/rrt_rl_2D/nodes/goal_node.py) | Python | 1 | 1 | 0 | 2 |
| [rrt\_rl\_2D/nodes/tree\_node.py](/rrt_rl_2D/nodes/tree_node.py) | Python | 1 | 0 | 0 | 1 |
| [rrt\_rl\_2D/planners/vec\_env\_planner.py](/rrt_rl_2D/planners/vec_env_planner.py) | Python | 16 | 0 | 8 | 24 |
| [rrt\_rl\_2D/rendering/\_\_init\_\_.py](/rrt_rl_2D/rendering/__init__.py) | Python | 5 | 0 | 2 | 7 |
| [rrt\_rl\_2D/rendering/debug\_renderer.py](/rrt_rl_2D/rendering/debug_renderer.py) | Python | 5 | 0 | 1 | 6 |
| [rrt\_rl\_2D/rendering/env\_renderer.py](/rrt_rl_2D/rendering/env_renderer.py) | Python | 6 | 0 | -1 | 5 |
| [rrt\_rl\_2D/simulator/simulator.py](/rrt_rl_2D/simulator/simulator.py) | Python | 7 | 0 | 2 | 9 |
| [rrt\_rl\_2D/simulator/standard\_config.py](/rrt_rl_2D/simulator/standard_config.py) | Python | 0 | 1 | 0 | 1 |
| [rrt\_rl\_2D/utils/save\_manager.py](/rrt_rl_2D/utils/save_manager.py) | Python | 352 | 4 | 82 | 438 |
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
| [scripts/development/rect\_env.py](/scripts/development/rect_env.py) | Python | -3 | 5 | -6 | -4 |
| [scripts/development/rrt\_rect.py](/scripts/development/rrt_rect.py) | Python | -2 | -1 | -2 | -5 |
| [scripts/development/test\_env.py](/scripts/development/test_env.py) | Python | -3 | 5 | 1 | 3 |
| [scripts/development/test\_maps.py](/scripts/development/test_maps.py) | Python | 5 | 0 | 0 | 5 |
| [scripts/multitool.ipynb](/scripts/multitool.ipynb) | JSON | 24 | 0 | 1 | 25 |
| [scripts/playground/PID.py](/scripts/playground/PID.py) | Python | 33 | 2 | 15 | 50 |
| [scripts/playground/PID2.py](/scripts/playground/PID2.py) | Python | 35 | 6 | 16 | 57 |
| [scripts/quantitative/qunatitative.py](/scripts/quantitative/qunatitative.py) | Python | 244 | 1 | 47 | 292 |
| [scripts/rrt\_blender.py](/scripts/rrt_blender.py) | Python | 98 | 6 | 39 | 143 |
| [scripts/rrt\_cable.py](/scripts/rrt_cable.py) | Python | 7 | 6 | -1 | 12 |
| [scripts/rrt\_cable\_radius.py](/scripts/rrt_cable_radius.py) | Python | 81 | 3 | 29 | 113 |

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details