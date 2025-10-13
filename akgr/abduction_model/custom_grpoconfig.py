#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Adapted from huggingface/open-r1: https://github.com/huggingface/open-r1/blob/6a0cd5c8ad031fc75118a4ce7f42a4860c3d8dea/src/open_r1/configs.py


from dataclasses import dataclass, field
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )
    # Diffusion generation specific arguments
    random_masking: bool = field(
        default=True,
        metadata={"help": "Whether to use random masking."},
    )
    p_mask_prompt: float = field(
        default=0.15,
        metadata={"help": "Probability of masking prompt tokens."},
    )
    diffusion_steps: int = field(
        default=128,
        metadata={"help": "Number of diffusion steps."},
    )
    generation_temperature: float = field(
        default=1.2,
        metadata={"help": "Temperature for generation."},
    )
    generation_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for generation."},
    )
    answer_len: int = field(
        default=32,
        metadata={"help": "Minimum number of characters in completion."},
    )
    query_len: int = field(
        default=10,
        metadata={"help": "Minimum number of characters in completion."},
    )

    scale_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to scale rewards."},
    )
    explore_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of exploration."},
    )
    deductive_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of deductive."},
    )

    