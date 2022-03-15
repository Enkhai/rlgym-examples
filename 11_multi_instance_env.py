from typing import Union, List, Type, Sequence

from rlgym.envs.match import Match
from rlgym.utils import RewardFunction, TerminalCondition, ObsBuilder, StateSetter
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.reward_functions import common_rewards
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from rlgym_tools.sb3_utils.sb3_multiple_instance_env import SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

# we reuse the SimpleObs, presented in the previous example
from utils import SimpleObs

reward = SB3CombinedLogReward.from_zipped(
    (DiffReward(common_rewards.LiuDistancePlayerToBallReward()), 0.05),
    (DiffReward(common_rewards.LiuDistanceBallToGoalReward()), 10),
    (common_rewards.ConstantReward(), -0.004),
    (common_rewards.EventReward(touch=0.05, goal=10)),
)
reward_names = ["PlayerToBallDistDiff", "BallToGoalDistDiff", "ConstantNegative", "GoalOrTouch"]
models_folder = "models/"


# We need a method to create a match for each instance
def get_match(reward: RewardFunction,
              terminal_conditions: Union[TerminalCondition, Sequence[TerminalCondition]],
              obs_builder: ObsBuilder,
              action_parser: ActionParser,
              state_setter: StateSetter,
              team_size: int,
              self_play=True):
    """
    A function that returns an RLGym match
    """
    return Match(reward_function=reward,
                 terminal_conditions=terminal_conditions,
                 obs_builder=obs_builder,
                 state_setter=state_setter,
                 action_parser=action_parser,
                 team_size=team_size,
                 self_play=self_play,
                 game_speed=500)


def get_matches(reward: RewardFunction,
                terminal_conditions: Union[TerminalCondition, Sequence[TerminalCondition]],
                obs_builder_cls: Type[ObsBuilder],
                action_parser_cls: Type[ActionParser] = KBMAction,
                state_setter_cls: Type[StateSetter] = DefaultState,
                self_plays: Union[bool, Sequence[bool]] = True,
                sizes: List[int] = None):
    """
    A function useful for creating a number of matches for multi-instance environments.\n
    If sizes is None or empty a list of `[3, 3, 2, 2, 1, 1]` sizes is used instead.
    """
    if not sizes:
        sizes = [3, 3, 2, 2, 1, 1]
    if type(self_plays) == bool:
        self_plays = [self_plays] * len(sizes)
    # out of the three cls type arguments, observation builders should at least not be shared between matches
    # (class argument instead of object argument, initialization happens for each match)
    # that is because observation builders often maintain state data that is specific to each match
    return [get_match(reward,
                      terminal_conditions,
                      obs_builder_cls(),
                      action_parser_cls(),
                      state_setter_cls(),
                      size,
                      self_play)
            for size, self_play in zip(sizes, self_plays)]


if __name__ == '__main__':
    matches = get_matches(reward,
                          [common_conditions.TimeoutCondition(500),
                           common_conditions.GoalScoredCondition()],
                          obs_builder_cls=SimpleObs,
                          action_parser_cls=KBMAction,
                          state_setter_cls=DefaultState,
                          self_plays=False,
                          sizes=[1] * 6  # 6 game instances of 1 player each
                          )

    # Creating the environment may take some time, this is normal
    # Always make sure your computer can handle your multiple game instances
    # You can do this by checking your device RAM and making sure your pagefile size is large enough
    # To check and change the pagefile size consult the following
    # https://www.tomshardware.com/news/how-to-manage-virtual-memory-pagefile-windows-10,36929.html
    # Each instance takes about 3.5Gb space of RAM upon startup but only ~400Mb when minimized
    # Turning off unnecessary apps and services can also be useful
    env = SB3MultipleInstanceEnv(match_func_or_matches=matches,
                                 wait_time=20)
    # With multi-instance environments the mean reward and episode length are not be logged
    # To overcome this, wrap the environment with VecMonitor
    env = VecMonitor(env)

    policy_kwargs = dict(net_arch=[dict(vf=[256, 256, 256, 256],  # completely separate actor and critic architecture
                                        pi=[256, 256, 256, 256])
                                   ])
    model = PPO(policy="MlpPolicy",
                env=env,
                learning_rate=1e-4,
                tensorboard_log="./bin",
                policy_kwargs=policy_kwargs,
                verbose=1,
                device="cpu",
                )
    # Random seed doesn't work in multi-instance environments
    # model.set_random_seed(0)

    callbacks = [SB3CombinedLogRewardCallback(reward_names),
                 # The number of steps here are effectively multiplied by 6 (6 agent-controlled cars)
                 CheckpointCallback(model.n_steps * 100,
                                    save_path=models_folder + "MLP2",
                                    name_prefix="model")]
    model.learn(total_timesteps=100_000_000, callback=callbacks, tb_log_name="PPO_MLP2_4x256")
    model.save(models_folder + "MLP2_final")

    env.close()
