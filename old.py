from setuptools import setup, find_packages
setup(
    name='rrt_rl_2D',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
        'stable-baselines3',
        'pymunk',
        'pygame',
        'tensorboard',
        'optuna',
    ],
    author="Michal Mrkos",
    author_email="mrkosmic@fel.cvut.cz",
    description="2D RRT with Reinforcement Learning",
    url="https://github.com/majklost/RRT-RL-2D",
    python_requires='>=3.10',
)
