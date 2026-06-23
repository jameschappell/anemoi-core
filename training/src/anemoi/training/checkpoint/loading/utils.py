# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pure utility functions for state dict manipulation.

These are stateless helpers used by loading strategies. No model loading,
no async, no pipeline wiring — just pure functions on state dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import torch


@dataclass(frozen=True)
class MatchResult:
    """Result of comparing source and target state dict keys.

    Attributes
    ----------
    missing_in_source : set[str]
        Keys present in target but not in source
    unexpected_in_source : set[str]
        Keys present in source but not in target
    shape_mismatches : set[str]
        Keys present in both but with different tensor shapes
    """

    missing_in_source: frozenset[str] = field(default_factory=frozenset)
    unexpected_in_source: frozenset[str] = field(default_factory=frozenset)
    shape_mismatches: frozenset[str] = field(default_factory=frozenset)


def filter_state_dict(
    source: dict[str, Any],
    target: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Filter source state dict to only include keys compatible with target.

    Non-mutating: builds new dicts, never modifies the inputs.

    Parameters
    ----------
    source : dict
        Source state dictionary (e.g. from checkpoint)
    target : dict
        Target state dictionary (e.g. from model)

    Returns
    -------
    tuple[dict, dict]
        (filtered, skipped) where filtered contains compatible entries
        and skipped maps key to a reason string for incompatible entries
    """
    filtered: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    for key, value in source.items():
        if key not in target:
            skipped[key] = "Key not in target"
            continue

        if (
            isinstance(value, torch.Tensor)
            and isinstance(target[key], torch.Tensor)
            and value.shape != target[key].shape
        ):
            skipped[key] = f"Shape mismatch: {value.shape} vs {target[key].shape}"
            continue

        filtered[key] = value

    return filtered, skipped


def remap_dataset_keys(
    source: dict[str, Any],
    remap_dataset: dict[str, str],
    target_keys: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, list[str]]]:
    """Remap dataset-name segments in checkpoint parameter keys.

    The remapping is structure-agnostic and does not rely on hardcoded
    module prefixes. Instead, each key is tokenized by ``.`` and any token
    matching a ``remap_dataset`` source name is replaced with its target
    name if that candidate key exists in ``target_keys``. This keeps the
    behavior robust to model refactors.

    Parameters
    ----------
    source : dict[str, Any]
        Source checkpoint state dict.
    remap_dataset : dict[str, str]
        Mapping from checkpoint dataset name to target dataset name,
        e.g. ``{'era5': 'gm'}``.
    target_keys : set[str], optional
        Target model state-dict keys. When provided, remaps are only
        applied if the remapped key exists in this set.

    Returns
    -------
    tuple[dict[str, Any], dict[str, str], dict[str, list[str]]]
        ``(remapped_state_dict, remapped_keys, collisions)`` where
        ``remapped_keys`` maps old key names to new key names for changed
        entries, and ``collisions`` maps destination keys to source keys
        that were not
        applied because another key already mapped to the same destination.
    """
    if not remap_dataset:
        return dict(source), {}, {}

    target_keys = target_keys or set()

    def remap_key_against_target(old_key: str, target_keys: set[str]) -> str:
        parts = old_key.split(".")

        # Try single-token substitutions from remap_dataset and only accept
        # candidates that exist in the target model state dict.
        for i, token in enumerate(parts):
            for source_name, target_name in remap_dataset.items():
                if source_name == target_name:
                    continue

                candidate_tokens: list[str] = []

                # Case 1: whole token is the dataset name (e.g. ".era5.")
                if token == source_name:
                    candidate_tokens.append(target_name)

                # Case 2: dataset name appears as an underscore-delimited segment
                # (e.g. "latlons_era5" -> "latlons_gm").
                underscore_segments = token.split("_")
                if source_name in underscore_segments:
                    replaced_segments = [target_name if seg == source_name else seg for seg in underscore_segments]
                    remapped_token = "_".join(replaced_segments)
                    if remapped_token != token:
                        candidate_tokens.append(remapped_token)

                for candidate_token in candidate_tokens:
                    candidate_parts = list(parts)
                    candidate_parts[i] = candidate_token
                    candidate = ".".join(candidate_parts)

                    if candidate in target_keys:
                        return candidate

        return old_key

    remapped_keys: dict[str, str] = {}
    remapped_state: dict[str, Any] = {}
    collisions: dict[str, list[str]] = {}

    for old_key, value in source.items():
        new_key = remap_key_against_target(old_key, target_keys) if target_keys else old_key

        if new_key in remapped_state and old_key != new_key:
            collisions.setdefault(new_key, []).append(old_key)
            continue

        if new_key != old_key:
            remapped_keys[old_key] = new_key

        remapped_state[new_key] = value

    return remapped_state, remapped_keys, collisions


def match_state_dict_keys(
    source_dict: dict[str, Any],
    target_dict: dict[str, Any],
) -> MatchResult:
    """Compare keys and shapes between source and target state dicts.

    Parameters
    ----------
    source_dict : dict
        Source state dictionary
    target_dict : dict
        Target state dictionary

    Returns
    -------
    MatchResult
        Comparison result with missing, unexpected, and mismatched keys
    """
    source_keys = set(source_dict.keys())
    target_keys = set(target_dict.keys())

    missing_in_source = target_keys - source_keys
    unexpected_in_source = source_keys - target_keys

    shape_mismatches: set[str] = set()
    for key in source_keys & target_keys:
        src_val = source_dict[key]
        tgt_val = target_dict[key]
        if isinstance(src_val, torch.Tensor) and isinstance(tgt_val, torch.Tensor) and src_val.shape != tgt_val.shape:
            shape_mismatches.add(key)

    return MatchResult(
        missing_in_source=frozenset(missing_in_source),
        unexpected_in_source=frozenset(unexpected_in_source),
        shape_mismatches=frozenset(shape_mismatches),
    )
