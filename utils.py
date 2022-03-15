from typing import Any, Union, Sequence, List, Type

import numpy as np
from rlgym.utils import ObsBuilder, common_values, RewardFunction, TerminalCondition, StateSetter
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.envs import Match
from rlgym.utils.state_setters import DefaultState
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction


class SimpleObs(ObsBuilder):
    """
    Simple observation builder for a ball and one car only\n
    Observation space is of shape 1 * 2 * 20:\n
    1 (batch)\n
    \* 2 (1 ball + 1 car)\n
    \* 20 (2 (car and ball flags)
    + 9 ((relative) standardized position, linear velocity and angular velocity 3-d vectors)
    + 6 (forward and upward rotation axes 3-d vectors)
    + 3 (boost, touching ground and has flip flags))

    If flatten is true, it simply returns a vector of length 40 (2 * 20)
    """
    POS_STD = 3000

    def __init__(self, flatten: bool = False):
        super(SimpleObs, self).__init__()
        # The `flatten` boolean is useful for MLP networks
        self.flatten = flatten

    def reset(self, initial_state: GameState):
        # build_obs is called automatically after environment reset
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        # We don't consider teams or inverted data, the observation builder
        # is built just for one blue team car and a ball

        ball = state.ball
        car = player.car_data

        ball_obs = np.concatenate([[1, 0],  # 1 for ball
                                   # standardized relative position
                                   (ball.position - car.position) / self.POS_STD,
                                   # standardized relative velocity
                                   (ball.linear_velocity - car.linear_velocity) / common_values.BALL_MAX_SPEED,
                                   # angular velocity not relative, car and ball share the same max angular velocities
                                   ball.angular_velocity / common_values.CAR_MAX_ANG_VEL,
                                   np.zeros(6),  # no rotation axes
                                   np.zeros(3)])  # no boost, touching ground and has flip flags
        car_obs = np.concatenate([[0, 1],  # 1 for car
                                  car.position / self.POS_STD,
                                  car.linear_velocity / common_values.CAR_MAX_SPEED,
                                  car.angular_velocity / common_values.CAR_MAX_ANG_VEL,
                                  car.forward(),
                                  car.up(),
                                  [player.boost_amount, player.on_ground, player.has_flip]])

        # In the case of an MLP policy network, return a concatenated 1-d array
        if self.flatten:
            return np.concatenate([ball_obs, car_obs])
        return np.stack([ball_obs, car_obs])


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
