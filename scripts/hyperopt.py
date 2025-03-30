import optuna
from pathlib import Path
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from rrt_rl_2D.simulator.standard_config import STANDARD_CONFIG
from rrt_rl_2D.CLIENT.makers import BlendMaker
from rrt_rl_2D.RL.hyperparams import give_args
from rrt_rl_2D.RL.training_utils import create_multi_env, create_callback_list, get_name, SaveNormalizeCallback

PATH = Path(__file__).parent.parent / "experiments" / "hyperopt"


class PruneCallback(BaseCallback):

    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because episodic reward
        threshold reached
    """

    parent: EvalCallback

    def __init__(self, verbose: int = 0, trial: optuna.Trial = None):
        super().__init__(verbose=verbose)
        self.trial = trial

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used with an ``EvalCallback``"
        intermediate_value = self.parent.best_mean_reward
        self.trial.report(intermediate_value, step=self.num_timesteps)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True


def objectiveBlend(trial):
    TIMESTEPS = 1_000_000
    cfg = STANDARD_CONFIG.copy()
    cfg['threshold'] = 20
    maker_factory = BlendMaker('AlmostEmpty', cfg, resetable=True)
    maker, maker_name, _ = maker_factory.first_try()
    n_envs = trial.suggest_categorical("n_envs", [4, 16, 32, 64, 128])
    env = create_multi_env(maker, n_envs, normalize=True)
    eval_env = create_multi_env(maker, 1, normalize=True)
    prune_clb = PruneCallback(trial=trial)
    save_norm_clb = SaveNormalizeCallback(PATH / "norm", save_freq=1)
    prune_clb = PruneCallback(trial=trial)
    eval_clb = EvalCallback(eval_env, best_model_save_path=PATH /
                            "best_model", callback_on_new_best=save_norm_clb, callback_after_eval=prune_clb, n_eval_episodes=25)

    args = give_args(trial)
    model = PPO("MlpPolicy", env, verbose=0, device='cpu', **args)
    model.learn(total_timesteps=TIMESTEPS, callback=[save_norm_clb, eval_clb])
    rew, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
    return rew


if __name__ == "__main__":
    STUDY_NAME = "Blend"
    study = optuna.load_study(
        study_name=STUDY_NAME, storage=f"sqlite:///{PATH / (STUDY_NAME + '.db')}")
    print("Starting optimization")
    study.optimize(objectiveBlend, n_trials=100,
                   timeout=None, show_progress_bar=False, gc_after_trial=True)
