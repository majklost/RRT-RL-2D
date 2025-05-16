import numpy as np

MAX_INT = 10000
_MANAGER = None


class _SeedManager:
    """
    Manager of seeds for the environment and the planner.
    Registers components and whether they belong to environment or planner.
    """

    def __init__(self, seed_env, seed_plan, garden=dict, verbose=True):
        """
        :param seed_env: Seed for the environment
        :param seed_plan: Seed for the planner (planning function)
        :param garden: Dictionary of seeds for components (if exact seed is needed)
        """
        self.verbose = verbose
        if verbose and seed_env is None:
            print("Environment will not be deterministic")
        if verbose and seed_plan is None:
            print("Planning will not be deterministic")

        self.env_rng = np.random.default_rng(seed_env)
        self.plan_rng = np.random.default_rng(seed_plan)
        self.garden = {}
        self.garden.update(garden)

    def get_seed(self, name, is_plannning=True):
        if name in self.garden:
            return self.garden[name]
        if is_plannning:
            self.garden[name] = self.plan_rng.integers(0, MAX_INT)
        else:
            self.garden[name] = self.env_rng.integers(0, MAX_INT)
        if self.verbose:
            print("seed for ", name, " is ", self.garden[name])
        return self.garden[name]


def init_manager(seed_env, seed_plan, garden={}, verbose=True):
    global _MANAGER
    _MANAGER = _SeedManager(seed_env, seed_plan, garden, verbose)


def manager():
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = _SeedManager(None, None, {}, True)
    return _MANAGER
