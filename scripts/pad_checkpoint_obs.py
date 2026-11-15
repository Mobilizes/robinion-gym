# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Append zero-initialized observation inputs to an RSL-RL checkpoint.

The stand/walk command adds a one-element gait-mode input to the end of the actor and critic
observation vectors, so a checkpoint trained before that command has first-layer weights that
are one column too narrow. This script pads those columns with zeros, which makes the policy
ignore the new input until fine-tuning learns to use it. The matching Adam state is padded too,
so the checkpoint can be resumed with or without ``--resume_optimizer_state``.

.. code-block:: bash

    uv run python scripts/pad_checkpoint_obs.py \
        --checkpoint logs/rsl_rl/robinion_velocity/walk-v6.1/model_19999.pt \
        --output logs/rsl_rl/robinion_velocity/walk-v6.1-gait-mode.pt \
        --std 0.3

    ./isaaclab.sh train --task RobinionGym-Velocity-Robinion \
        --checkpoint logs/rsl_rl/robinion_velocity/walk-v6.1-gait-mode.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _pad_last_input(tensor: torch.Tensor, extra: int) -> torch.Tensor:
    """Append ``extra`` zero columns to a 2-D weight tensor."""
    padding = tensor.new_zeros((tensor.shape[0], extra))
    return torch.cat((tensor, padding), dim=1)


def main() -> None:
    """Pad a checkpoint and write the result to ``--output``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Source RSL-RL checkpoint.")
    parser.add_argument("--output", type=str, required=True, help="Destination checkpoint.")
    parser.add_argument(
        "--extra-obs",
        type=int,
        default=1,
        help="Number of observation dimensions appended at the end of the policy input.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Iteration stored in the output checkpoint. Defaults to 0 so fine-tuning starts fresh.",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=None,
        help=(
            "Optional exploration standard deviation written into the actor distribution. A converged "
            "checkpoint restores its small std, which can prevent the policy from discovering a new behavior."
        ),
    )
    args = parser.parse_args()
    if args.extra_obs < 0:
        parser.error("--extra-obs must be non-negative")
    if args.std is not None and args.std <= 0.0:
        parser.error("--std must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    critic = checkpoint["critic_state_dict"]

    old_to_new: dict[int, int] = {}
    for name, state in (("actor", actor), ("critic", critic)):
        weight = state["mlp.0.weight"]
        old_dim = weight.shape[1]
        new_dim = old_dim + args.extra_obs
        state["mlp.0.weight"] = _pad_last_input(weight, args.extra_obs)
        old_to_new[old_dim] = new_dim
        print(f"{name}: {old_dim} -> {new_dim} observation inputs")

    if args.std is not None:
        std_param = actor.get("distribution.std_param")
        if std_param is None:
            raise KeyError("The actor checkpoint has no 'distribution.std_param' tensor; --std cannot be applied.")
        std_param.fill_(args.std)
        print(f"actor: exploration std -> {args.std}")

    optimizer = checkpoint.get("optimizer_state_dict")
    if optimizer is not None:
        num_padded = 0
        for state in optimizer["state"].values():
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if value is not None and value.ndim == 2 and value.shape[1] in old_to_new:
                    state[key] = _pad_last_input(value, old_to_new[value.shape[1]] - value.shape[1])
                    num_padded += 1
        print(f"optimizer: padded {num_padded} state tensors")

    checkpoint["iter"] = args.iteration
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
