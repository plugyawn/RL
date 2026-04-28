# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from nemo_rl.algorithms.kondo import (
    _solve_stochastic_row_keep_probabilities,
    apply_kondo_mode,
    compute_required_row_multiple,
    resolve_kondo_config,
    screen_kondo_tokens,
)
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _make_train_data() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [1, 11, 12, 13],
                    [1, 21, 22, 23],
                    [1, 31, 32, 0],
                ],
                dtype=torch.long,
            ),
            "token_mask": torch.tensor(
                [
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 1.0, 1.0]),
            "advantages": torch.tensor(
                [
                    [0.0, 3.0, 3.0, 3.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 2.0, 2.0, 0.0],
                ]
            ),
            "prev_logprobs": torch.tensor(
                [
                    [0.0, -4.0, -3.0, -2.0],
                    [0.0, -1.0, -1.0, -1.0],
                    [0.0, -2.0, -2.0, 0.0],
                ]
            ),
            "generation_logprobs": torch.tensor(
                [
                    [0.0, -4.0, -3.0, -2.0],
                    [0.0, -1.0, -1.0, -1.0],
                    [0.0, -2.0, -2.0, 0.0],
                ]
            ),
        }
    )


def test_compute_required_row_multiple_respects_batching_mode():
    assert compute_required_row_multiple(2, 4, False, False) == 8
    assert compute_required_row_multiple(2, 4, True, False) == 2
    assert compute_required_row_multiple(2, 4, False, True) == 2


def test_low_contrast_fallback_smoothly_densifies_keep_probabilities():
    keep_prob = _solve_stochastic_row_keep_probabilities(
        row_signal=torch.tensor([1.0, 1.0], dtype=torch.float32),
        row_cost=torch.tensor([4, 4], dtype=torch.int64),
        target_token_fraction=0.7,
        total_valid_tokens=8,
        min_keep_probability=0.05,
        low_contrast_fallback_low=0.2,
        low_contrast_fallback_high=0.5,
    )

    torch.testing.assert_close(
        keep_prob,
        torch.ones_like(keep_prob, dtype=torch.float32),
    )


def test_resolve_kondo_config_rejects_removed_legacy_modes_and_priorities():
    with pytest.raises(ValueError, match="Unsupported Kondo mode"):
        resolve_kondo_config({"enabled": True, "mode": "response_dense_rows_v2"})

    with pytest.raises(ValueError, match="Unsupported Kondo priority mode"):
        resolve_kondo_config(
            {
                "enabled": True,
                "mode": "dense_reference",
                "priority_mode": "stratified_pg_kl_tempered_delight",
            }
        )

    with pytest.raises(
        ValueError,
        match=(
            "grpo.kondo.priority_mode=split_dual_tempered_row_sum is currently only "
            "supported with grpo.kondo.mode=stochastic_response_rows"
        ),
    ):
        resolve_kondo_config(
            {
                "enabled": True,
                "mode": "dense_reference",
                "priority_mode": "split_dual_tempered_row_sum",
            }
        )

    with pytest.raises(
        ValueError,
        match=(
            "grpo.kondo.priority_mode=split_dual_tempered_competence is currently only "
            "supported with grpo.kondo.mode=stochastic_response_rows"
        ),
    ):
        resolve_kondo_config(
            {
                "enabled": True,
                "mode": "dense_reference",
                "priority_mode": "split_dual_tempered_competence",
            }
        )

    with pytest.raises(
        ValueError,
        match=(
            "grpo.kondo.priority_mode=split_dual_tempered_error_anchor is currently only "
            "supported with grpo.kondo.mode=stochastic_response_rows"
        ),
    ):
        resolve_kondo_config(
            {
                "enabled": True,
                "mode": "dense_reference",
                "priority_mode": "split_dual_tempered_error_anchor",
            }
        )

    with pytest.raises(
        ValueError,
        match=(
            "grpo.kondo.priority_mode=split_dual_tempered_row_hierarchy is currently only "
            "supported with grpo.kondo.mode=stochastic_response_rows"
        ),
    ):
        resolve_kondo_config(
            {
                "enabled": True,
                "mode": "dense_reference",
                "priority_mode": "split_dual_tempered_row_hierarchy",
            }
        )


def test_screen_kondo_tokens_builds_exact_dense_reference_gate():
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "dense_reference",
            "target_backward_token_fraction": 0.5,
            "priority_mode": "delight",
        }
    )
    screening = screen_kondo_tokens(_make_train_data(), cfg)

    assert screening is not None
    assert screening.total_valid_tokens == 8
    assert screening.screenable_token_count == 8
    assert screening.reference_token_gate.sum().item() == 4.0
    assert screening.loss_token_mask.shape == torch.Size([3, 4])
    assert screening.loss_token_mask[:, 0].sum().item() == 0.0


def test_split_dual_tempered_delight_matches_sign_target_means():
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 11, 12, 13, 14, 15, 16]], dtype=torch.long),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
            "sample_mask": torch.tensor([1.0]),
            "advantages": torch.tensor([[0.0, 3.0, 2.0, -3.0, -2.0, 0.0, 0.0]]),
            "prev_logprobs": torch.tensor([[0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]),
            "generation_logprobs": torch.tensor(
                [[0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]
            ),
            "prompt_group_std": torch.tensor([1.0]),
        }
    )
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "dense_reference",
            "priority_mode": "split_dual_tempered_delight",
            "surprisal_temperature": 1.0,
            "gate_temperature": 1.0,
            "positive_keep_floor": 1.0,
            "negative_keep_floor": 0.5,
        }
    )

    screening = screen_kondo_tokens(train_data, cfg)

    assert screening is not None
    gate = screening.reference_token_gate[0]
    assert gate[0].item() == 1.0
    assert gate[1].item() == 1.0
    assert gate[2].item() > 0.0
    assert gate[3].item() > 0.0
    assert gate[4].item() == 0.0
    assert gate[5].item() == 0.0


def test_split_dual_tempered_delight_quarantines_zero_variance_groups():
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 11, 12], [1, 21, 22]], dtype=torch.long),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
            "sample_mask": torch.tensor([1.0, 1.0]),
            "advantages": torch.tensor([[0.0, 3.0, 3.0], [0.0, 1.0, -1.0]]),
            "prev_logprobs": torch.tensor([[0.0, -1.0, -1.0], [0.0, -1.0, -1.0]]),
            "generation_logprobs": torch.tensor(
                [[0.0, -1.0, -1.0], [0.0, -1.0, -1.0]]
            ),
            "prompt_group_std": torch.tensor([0.0, 1.0]),
        }
    )
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "dense_reference",
            "target_backward_token_fraction": 0.5,
            "priority_mode": "split_dual_tempered_delight",
            "surprisal_temperature": 1.0,
        }
    )

    screening = screen_kondo_tokens(train_data, cfg)

    assert screening is not None
    torch.testing.assert_close(
        screening.reference_token_gate.sum(dim=-1),
        torch.tensor([0.0, 1.2]),
    )


def test_split_dual_tempered_delight_uses_teacher_coverage_row_priority():
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [1, 11, 12],
                    [1, 21, 22],
                ],
                dtype=torch.long,
            ),
            "token_mask": torch.tensor(
                [
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 1.0]),
            "advantages": torch.tensor(
                [
                    [0.0, 3.0, 3.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            "prev_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "generation_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "prompt_group_std": torch.tensor([1.0, 1.0]),
        }
    )
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "stochastic_response_rows",
            "priority_mode": "split_dual_tempered_delight",
            "target_backward_token_fraction": 0.5,
            "surprisal_temperature": 1.0,
        }
    )
    screening = screen_kondo_tokens(train_data, cfg)

    assert screening is not None
    expected = torch.tensor(
        [
            2 * 3.0 * torch.log1p(torch.tensor(1.0)),
            0.0,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(screening.row_sampling_priority, expected)
    assert screening.row_sampling_priority[0].item() > screening.row_sampling_priority[1].item()


def test_split_dual_tempered_row_sum_uses_raw_cumulative_row_priority():
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [1, 11, 12, 13],
                    [1, 21, 22, 23],
                ],
                dtype=torch.long,
            ),
            "token_mask": torch.tensor(
                [
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 1.0]),
            "advantages": torch.tensor(
                [
                    [0.0, 10.0, 0.0, 0.0],
                    [0.0, 4.0, 4.0, 4.0],
                ]
            ),
            "prev_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0, -1.0],
                ]
            ),
            "generation_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0, -1.0],
                ]
            ),
        }
    )
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "stochastic_response_rows",
            "priority_mode": "split_dual_tempered_row_sum",
            "target_backward_token_fraction": 0.25,
            "surprisal_temperature": 1.0,
        }
    )
    screening = screen_kondo_tokens(train_data, cfg)

    assert screening is not None
    expected = torch.tensor(
        [
            10.0 * torch.log1p(torch.tensor(1.0)),
            12.0 * torch.log1p(torch.tensor(1.0)),
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(screening.row_sampling_priority, expected)
    assert screening.row_sampling_priority[1].item() > screening.row_sampling_priority[0].item()


def test_split_dual_tempered_row_hierarchy_preserves_solved_prompt_without_prompt_atomicity():
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [1, 11, 12],
                    [1, 21, 22],
                    [1, 31, 32],
                    [1, 41, 42],
                ],
                dtype=torch.long,
            ),
            "token_mask": torch.tensor(
                [
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 1.0, 1.0, 1.0]),
            "advantages": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.5, 0.0],
                ],
                dtype=torch.float32,
            ),
            "prev_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "generation_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "sample_reward": torch.tensor([1.0, 1.0, 0.25, 0.25]),
            "prompt_group_id": torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        }
    )
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "stochastic_response_rows",
            "priority_mode": "split_dual_tempered_row_hierarchy",
            "target_backward_token_fraction": 0.25,
            "surprisal_temperature": 1.0,
        }
    )
    screening = screen_kondo_tokens(train_data, cfg)

    assert screening is not None
    torch.testing.assert_close(
        screening.row_sampling_priority[:2],
        torch.full((2,), screening.row_sampling_priority[0], dtype=torch.float32),
    )
    assert screening.row_sampling_priority[0].item() > 0.0
    assert screening.row_sampling_priority[2].item() > screening.row_sampling_priority[3].item()
    assert screening.row_sampling_priority[3].item() > 0.0


def test_split_dual_tempered_competence_rescues_high_reward_zero_actor_group():
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [1, 11, 12],
                    [1, 21, 22],
                    [1, 31, 32],
                    [1, 41, 42],
                ],
                dtype=torch.long,
            ),
            "token_mask": torch.tensor(
                [
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 1.0, 1.0, 1.0]),
            "advantages": torch.zeros((4, 3), dtype=torch.float32),
            "prev_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "generation_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "sample_reward": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "prompt_group_id": torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        }
    )
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "stochastic_response_rows",
            "priority_mode": "split_dual_tempered_competence",
            "target_backward_token_fraction": 0.25,
            "surprisal_temperature": 1.0,
        }
    )
    screening = screen_kondo_tokens(train_data, cfg)

    assert screening is not None
    torch.testing.assert_close(
        screening.row_sampling_priority[:2],
        torch.full((2,), screening.row_sampling_priority[0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        screening.row_sampling_priority[2:],
        torch.zeros(2, dtype=torch.float32),
    )
    assert screening.row_sampling_priority[0].item() > 0.0


def test_split_dual_tempered_error_anchor_preserves_solved_group_without_rescuing_wrong_group():
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [1, 11, 12],
                    [1, 21, 22],
                    [1, 31, 32],
                    [1, 41, 42],
                    [1, 51, 52],
                    [1, 61, 62],
                ],
                dtype=torch.long,
            ),
            "token_mask": torch.tensor(
                [
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "advantages": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.25, 0.25],
                    [0.0, 0.25, 0.25],
                ],
                dtype=torch.float32,
            ),
            "prev_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "generation_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "sample_reward": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.875, 0.875]),
            "prompt_group_id": torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
        }
    )
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "stochastic_response_rows",
            "priority_mode": "split_dual_tempered_error_anchor",
            "target_backward_token_fraction": 0.25,
            "surprisal_temperature": 1.0,
        }
    )
    screening = screen_kondo_tokens(train_data, cfg)

    assert screening is not None
    solved_group = screening.row_sampling_priority[:2]
    wrong_group = screening.row_sampling_priority[2:4]
    nearly_solved_group = screening.row_sampling_priority[4:]
    torch.testing.assert_close(
        solved_group,
        torch.full((2,), solved_group[0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        wrong_group,
        torch.zeros(2, dtype=torch.float32),
    )
    torch.testing.assert_close(
        nearly_solved_group,
        torch.full((2,), nearly_solved_group[0], dtype=torch.float32),
    )
    assert solved_group[0].item() > 0.0
    assert nearly_solved_group[0].item() > 0.0


def test_apply_kondo_mode_dense_reference_preserves_full_batch():
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "dense_reference",
            "target_backward_token_fraction": 0.5,
        }
    )
    train_data = _make_train_data()
    screening = screen_kondo_tokens(train_data, cfg)
    policy_train_data, metrics = apply_kondo_mode(
        train_data=train_data,
        screening=screening,
        cfg=cfg,
        row_multiple=1,
    )

    assert policy_train_data["input_ids"].shape[0] == 3
    assert metrics["kondo_rows_total"] == 3.0
    assert metrics["kondo_rows_kept"] == 3.0
    assert metrics["kondo_selected_token_recall"] == 1.0
    assert metrics["kondo_actual_backward_token_fraction"] == 1.0
    assert metrics["kondo_kept_row_token_fraction"] == 1.0
    assert "loss_token_mask" in train_data


def test_kondo_bypasses_when_nonfinite_screen_tokens_are_present():
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "stochastic_response_rows",
            "target_backward_token_fraction": 0.5,
            "priority_mode": "delight",
            "bypass_on_nonfinite": True,
        }
    )
    train_data = _make_train_data()
    train_data["prev_logprobs"][0, 1] = float("-inf")
    screening = screen_kondo_tokens(train_data, cfg)
    policy_train_data, metrics = apply_kondo_mode(
        train_data=train_data,
        screening=screening,
        cfg=cfg,
        row_multiple=1,
    )

    assert screening is not None
    assert screening.should_bypass is True
    assert policy_train_data["input_ids"].shape[0] == 3
    assert metrics["kondo_bypass_step"] == 1.0
    assert metrics["kondo_nonfinite_screen_token_count"] == 1.0


def test_stochastic_response_rows_bypasses_all_zero_actor_batches():
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 11, 12], [1, 21, 22]], dtype=torch.long),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
            "sample_mask": torch.tensor([1.0, 1.0]),
            "advantages": torch.zeros((2, 3), dtype=torch.float32),
            "prev_logprobs": torch.tensor([[0.0, -1.0, -1.0], [0.0, -1.0, -1.0]]),
            "generation_logprobs": torch.tensor(
                [[0.0, -1.0, -1.0], [0.0, -1.0, -1.0]]
            ),
            "prompt_group_std": torch.tensor([0.0, 0.0]),
        }
    )
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "stochastic_response_rows",
            "priority_mode": "split_dual_tempered_delight",
            "target_backward_token_fraction": 0.7,
            "surprisal_temperature": 1.0,
        }
    )
    screening = screen_kondo_tokens(train_data, cfg)
    policy_train_data, metrics = apply_kondo_mode(
        train_data=train_data,
        screening=screening,
        cfg=cfg,
        row_multiple=1,
    )

    assert screening is not None
    assert torch.all(screening.row_sampling_priority == 0)
    assert policy_train_data["input_ids"].shape[0] == 2
    torch.testing.assert_close(
        policy_train_data["kondo_row_selected"],
        torch.ones(2, dtype=torch.float32),
    )
    torch.testing.assert_close(
        policy_train_data["kondo_row_keep_prob"],
        torch.ones(2, dtype=torch.float32),
    )
    assert "sample_loss_weight" not in policy_train_data
    assert metrics["kondo_bypass_step"] == 1.0
    assert metrics["kondo_actual_backward_token_fraction"] == 1.0


def test_stochastic_response_rows_samples_prompt_groups_atomically():
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [[1, 11, 12], [1, 21, 22], [1, 31, 32], [1, 41, 42]],
                dtype=torch.long,
            ),
            "token_mask": torch.tensor(
                [
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 1.0, 1.0, 1.0]),
            "advantages": torch.tensor(
                [
                    [0.0, 3.0, 3.0],
                    [0.0, -1.0, -1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "prev_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "generation_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "prompt_group_std": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "prompt_group_id": torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        }
    )
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "stochastic_response_rows",
            "priority_mode": "split_dual_tempered_delight",
            "target_backward_token_fraction": 0.3,
            "surprisal_temperature": 1.0,
            "positive_keep_floor": 0.95,
            "negative_keep_floor": 0.25,
            "min_keep_probability": 0.05,
        }
    )
    screening = screen_kondo_tokens(train_data, cfg)
    torch.manual_seed(0)
    _, metrics = apply_kondo_mode(
        train_data=train_data,
        screening=screening,
        cfg=cfg,
        row_multiple=1,
    )

    assert screening is not None
    group0_keep = train_data["kondo_row_keep_prob"][:2]
    group1_keep = train_data["kondo_row_keep_prob"][2:]
    torch.testing.assert_close(group0_keep, torch.full_like(group0_keep, group0_keep[0]))
    torch.testing.assert_close(group1_keep, torch.full_like(group1_keep, group1_keep[0]))
    assert group0_keep[0].item() > group1_keep[0].item()
    assert group1_keep[0].item() == pytest.approx(0.05)
    group0_selected = train_data["kondo_row_selected"][:2]
    group1_selected = train_data["kondo_row_selected"][2:]
    torch.testing.assert_close(
        group0_selected, torch.full_like(group0_selected, group0_selected[0])
    )
    torch.testing.assert_close(
        group1_selected, torch.full_like(group1_selected, group1_selected[0])
    )
    assert metrics["kondo_rows_kept"] in {0.0, 2.0, 4.0}


def test_row_sum_adds_kl_only_sentinel_for_dropped_solved_prompt(
    monkeypatch: pytest.MonkeyPatch,
):
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [[1, 11, 12], [1, 21, 22], [1, 31, 32], [1, 41, 42]],
                dtype=torch.long,
            ),
            "token_mask": torch.tensor(
                [
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 1.0, 1.0, 1.0]),
            "advantages": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.0, 0.5, 0.5],
                ],
                dtype=torch.float32,
            ),
            "prev_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "generation_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "sample_reward": torch.tensor([1.0, 1.0, 0.5, 0.5]),
            "prompt_group_id": torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        }
    )
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "stochastic_response_rows",
            "priority_mode": "split_dual_tempered_row_sum",
            "target_backward_token_fraction": 0.3,
            "surprisal_temperature": 1.0,
            "min_keep_probability": 0.05,
        }
    )
    screening = screen_kondo_tokens(train_data, cfg)
    assert screening is not None

    monkeypatch.setattr(
        "nemo_rl.algorithms.kondo._sample_stochastic_groups",
        lambda **kwargs: torch.tensor([2, 3], dtype=torch.long),
    )

    policy_train_data, _ = apply_kondo_mode(
        train_data=train_data,
        screening=screening,
        cfg=cfg,
        row_multiple=1,
    )

    zero_actor_rows = (
        policy_train_data["loss_token_mask"][:, 1:].sum(dim=-1) == 0
    )
    assert policy_train_data["input_ids"].shape[0] == 3
    assert int(zero_actor_rows.sum().item()) == 1
    torch.testing.assert_close(
        policy_train_data["sample_loss_weight"][zero_actor_rows],
        torch.zeros(1, dtype=torch.float32),
    )
    assert (
        policy_train_data["loss_token_mask"][~zero_actor_rows][:, 1:].sum().item()
        > 0.0
    )


def test_row_hierarchy_uses_row_sampling_even_with_prompt_groups(
    monkeypatch: pytest.MonkeyPatch,
):
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [[1, 11, 12], [1, 21, 22], [1, 31, 32], [1, 41, 42]],
                dtype=torch.long,
            ),
            "token_mask": torch.tensor(
                [
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 1.0, 1.0, 1.0]),
            "advantages": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.5, 0.0],
                ],
                dtype=torch.float32,
            ),
            "prev_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "generation_logprobs": torch.tensor(
                [
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                    [0.0, -1.0, -1.0],
                ]
            ),
            "sample_reward": torch.tensor([1.0, 1.0, 0.25, 0.25]),
            "prompt_group_id": torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        }
    )
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "stochastic_response_rows",
            "priority_mode": "split_dual_tempered_row_hierarchy",
            "target_backward_token_fraction": 0.3,
            "surprisal_temperature": 1.0,
            "min_keep_probability": 0.05,
        }
    )
    screening = screen_kondo_tokens(train_data, cfg)
    assert screening is not None

    def _fail_group_sampler(**kwargs):
        raise AssertionError("group sampler should not be used")

    monkeypatch.setattr(
        "nemo_rl.algorithms.kondo._sample_stochastic_groups",
        _fail_group_sampler,
    )
    monkeypatch.setattr(
        "nemo_rl.algorithms.kondo._sample_stochastic_rows",
        lambda **kwargs: torch.tensor([0, 2], dtype=torch.long),
    )

    policy_train_data, metrics = apply_kondo_mode(
        train_data=train_data,
        screening=screening,
        cfg=cfg,
        row_multiple=1,
    )

    assert policy_train_data["input_ids"].shape[0] == 2
    torch.testing.assert_close(
        train_data["kondo_row_selected"],
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    assert metrics["kondo_rows_kept"] == 2.0


def test_stochastic_response_rows_validates_keep_probability_and_mode():
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "stochastic_response_rows",
            "priority_mode": "split_dual_tempered_delight",
            "min_keep_probability": 0.2,
        }
    )

    assert cfg["mode"] == "stochastic_response_rows"
    assert cfg["min_keep_probability"] == 0.2

    with pytest.raises(ValueError, match="min_keep_probability"):
        resolve_kondo_config(
            {
                "enabled": True,
                "mode": "stochastic_response_rows",
                "priority_mode": "split_dual_tempered_delight",
                "min_keep_probability": 0.0,
            }
        )


def test_stochastic_response_rows_sets_inverse_propensity_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    train_data = _make_train_data()
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "stochastic_response_rows",
            "priority_mode": "split_dual_tempered_delight",
            "target_backward_token_fraction": 0.5,
            "surprisal_temperature": 1.0,
            "gate_temperature": 1.0,
            "positive_keep_floor": 0.9,
            "negative_keep_floor": 0.25,
            "min_keep_probability": 0.2,
        }
    )
    screening = screen_kondo_tokens(train_data, cfg)
    assert screening is not None

    monkeypatch.setattr(
        "nemo_rl.algorithms.kondo.torch.rand_like",
        lambda tensor: torch.tensor([0.0, 0.99, 0.0], dtype=tensor.dtype),
    )

    policy_train_data, metrics = apply_kondo_mode(
        train_data=train_data,
        screening=screening,
        cfg=cfg,
        row_multiple=1,
    )

    selected = train_data["kondo_row_selected"].bool()
    keep_prob = train_data["kondo_row_keep_prob"][selected]

    assert selected.sum().item() == 2
    torch.testing.assert_close(
        policy_train_data["sample_loss_weight"],
        1.0 / keep_prob,
    )
    torch.testing.assert_close(
        train_data["kondo_row_ht_weight"][selected],
        1.0 / keep_prob,
    )
    assert metrics["kondo_rows_kept"] == 2.0
    assert metrics["kondo_row_keep_probability_mean"] > 0.0
    assert metrics["kondo_row_ht_weight_mean"] >= 1.0


def test_clipped_pg_loss_respects_loss_token_mask_and_normalizer():
    cfg = {
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "ratio_clip_c": None,
        "disable_ppo_ratio": True,
        "reference_policy_kl_penalty": 0.0,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": 20.0,
        "kl_output_clamp_value": 10.0,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        "sequence_level_importance_ratios": False,
        "token_level_loss": True,
        "force_on_policy_ratio": False,
    }
    loss_fn = ClippedPGLossFn(cfg)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "advantages": torch.tensor([[0.0, 2.0, 2.0, 2.0]]),
            "prev_logprobs": torch.tensor([[0.0, -1.0, -2.0, -3.0]]),
            "generation_logprobs": torch.tensor([[0.0, -1.0, -2.0, -3.0]]),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
            "loss_token_mask": torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
            "loss_normalizer": torch.tensor([4.0]),
            "sample_mask": torch.tensor([1.0]),
        }
    )
    next_token_logprobs = torch.tensor([[-1.0, -2.0, -3.0]])

    loss, _ = loss_fn(
        next_token_logprobs=next_token_logprobs,
        data=data,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(3.0),
    )

    torch.testing.assert_close(loss, torch.tensor(2.0))


def test_clipped_pg_loss_applies_sample_loss_weight():
    cfg = {
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "ratio_clip_c": None,
        "disable_ppo_ratio": True,
        "reference_policy_kl_penalty": 0.0,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": 20.0,
        "kl_output_clamp_value": 10.0,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        "sequence_level_importance_ratios": False,
        "token_level_loss": True,
        "force_on_policy_ratio": False,
    }
    loss_fn = ClippedPGLossFn(cfg)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "advantages": torch.tensor([[0.0, 2.0, 2.0, 2.0]]),
            "prev_logprobs": torch.tensor([[0.0, -1.0, -2.0, -3.0]]),
            "generation_logprobs": torch.tensor([[0.0, -1.0, -2.0, -3.0]]),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
            "loss_normalizer": torch.tensor([3.0]),
            "sample_mask": torch.tensor([1.0]),
            "sample_loss_weight": torch.tensor([2.0]),
        }
    )
    next_token_logprobs = torch.tensor([[-1.0, -2.0, -3.0]])

    loss, _ = loss_fn(
        next_token_logprobs=next_token_logprobs,
        data=data,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(3.0),
    )

    torch.testing.assert_close(loss, torch.tensor(8.0))
