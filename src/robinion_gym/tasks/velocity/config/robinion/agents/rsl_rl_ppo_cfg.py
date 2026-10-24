# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class CappedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian output distribution with a bounded exploration standard deviation.

    The action clip is a guardrail, not a regularizer: PPO still scores the unclipped
    sample, so once the std grows past ``clip_actions`` every sample maps to the same
    clamped action and the policy gradient can no longer pull the std back down. Keep the
    std well inside the clip instead of relying on the clamp to bound it.
    """

    std_range: tuple[float, float] = (0.1, 0.75)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 10000
    save_interval = 100
    # Hard guardrail: clamp sampled actions before they reach the environment. Keep the
    # policy std below this value; clipping does not bound the network output or the PPO
    # objective, it only bounds the action the simulator sees.
    clip_actions = 1.2
    experiment_name = "robinion_velocity"
    obs_groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=CappedGaussianDistributionCfg(init_std=0.3),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # A small entropy bonus keeps the policy exploring instead of collapsing onto
        # the first stable gait it finds. The std floor above prevents the earlier
        # collapse to a near-deterministic policy; the cap prevents the old runaway.
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
