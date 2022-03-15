import numpy as np
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
    EventReward
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_instantaneous_fps_callback import SB3InstantaneousFPSCallback
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecCheckNan, VecMonitor, VecNormalize

from stable_baselines3 import PPO
from utils import get_matches

# Following https://github.com/Impossibum/rlgym_quickstart_tutorial_bot/blob/main/youtube_examplebot.py

if __name__ == '__main__':
    # A. Configuration

    frame_skip = 8  # default frame skip
    half_life_seconds = 10  # the number of seconds it takes the gamma exponential to reach 0.5
    fps = 120 // frame_skip  # game physics engine runs at 120 Hz\fps
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # gamma computed through an inverse function
    agents_per_match = 1 * 2  # self-play
    num_instances = 3
    target_steps = 500_000  # number of total rollout steps
    steps = target_steps // (num_instances * agents_per_match)  # rollout steps per agent
    batch_size = steps // 10  # batch size used for training
    training_interval = 25_000_000  # number of training steps, multiplied by (num_instances * agents_per_match)

    # B. Making the rewards

    rewards = SB3CombinedLogReward(
        (
            VelocityPlayerToBallReward(),
            VelocityBallToGoalReward(),
            EventReward(
                team_goal=100.0,
                concede=-100.0,
                shot=5.0,
                save=30.0,
                demo=10.0,
            ),
        ),
        (0.1, 1.0, 1.0))
    reward_names = [fn.__class__.__name__ for fn in rewards.reward_functions]

    # C. Setting up the environment

    # get_matches our custom function for creating Matches from example 11, suitable for an SB3MultipleInstanceEnv
    matches = get_matches(reward=rewards,
                          terminal_conditions=[common_conditions.TimeoutCondition(fps * 300),
                                               common_conditions.NoTouchTimeoutCondition(fps * 45),
                                               common_conditions.GoalScoredCondition()],
                          obs_builder_cls=AdvancedObs,
                          action_parser_cls=DiscreteAction,
                          sizes=[1] * num_instances,
                          self_plays=True
                          )
    # Create the environment
    env = SB3MultipleInstanceEnv(matches)
    # Wrap the environment with VecCheckNan (useful for checking nan values, optional)
    env = VecCheckNan(env)
    # and VecMonitor (useful for logging mean reward and episode length)
    env = VecMonitor(env)
    # We can also normalize our environment rewards
    # This helps the model learn from small, normalized and scaled reward values and is often recommended
    # VecNormalize works by maintaining a moving average across all rewards and/or observations
    env = VecNormalize(env, norm_obs=False, gamma=gamma)

    # D. Initializing the model

    # This architecture is reported to be somewhat working
    policy_kwargs = dict(net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])])
    model = PPO("MlpPolicy",
                env,
                n_epochs=10,
                policy_kwargs=policy_kwargs,
                # a learning rate between 2e-4 and 5e-5 is considered optimal by the community
                # for the Adam optimizer (default)
                learning_rate=1e-4,
                ent_coef=0.01,  # considered good, taken from the Atari PPO
                vf_coef=1,  # considered good, taken from the Atari PPO
                gamma=gamma,
                verbose=3,
                batch_size=batch_size,
                n_steps=steps,
                tensorboard_log="bin")

    # E. Making the callbacks

    callbacks = [SB3CombinedLogRewardCallback(reward_names),
                 # Many times SB3 reports a wrong number of rollout fps due to the delay when launching instances
                 # SB3InstantaneousFPSCallback combats this
                 SB3InstantaneousFPSCallback(),
                 CheckpointCallback(model.n_steps * 30,
                                    save_path="models/custom_test",
                                    name_prefix="model")]

    # F. Training the model

    model.learn(training_interval, callbacks, tb_log_name="CustomRewTestPPO")
    model.save("models/custom_test_final")
    env.close()
