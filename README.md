# RRT-RL-2D

Combination of RRT and RL algorithm for planning of deformable cables in pymunk.

# Installation

1. Clone the repository via:

```bash
git clone git@github.com:majklost/RRT-RL-2D.git
```

2. Go to the cloned repository

```bash
cd RRT-RL-2D
```

3. (Optional)
   Create and activate python virtual environment here

```bash
python -m venv .venv
. ./.venv/bin/activate
```

4. Install the framework via

```sh
pip install .
```

5. (Optional)
   One can verify the installation by running
   You should be able to move the rectangle with arrow keys

```bash
python ./scripts/controllable_tests/rect_env.py
```

6. (Optional)
   For quick testing of planning via RRT, you can use

```bash
cd test_rrt
chmod +x ./test_rrt.sh
./test_rrt.sh
python ../scripts/play_rpath.py ./Test.rpath
```

7. (Optional)
   If one want to train RL agents, a manager and folder for experiments are created by running

```bash
python ./scripts/init_RL.py
```
