# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torchinfo
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.diagnostics.profilers import BenchmarkProfiler
from anemoi.training.tasks import TemporalDownscaler
from anemoi.training.train.profiler import AnemoiProfiler


def _make_minimal_index_collection(name_to_index: dict[str, int]) -> IndexCollection:
    return IndexCollection(DictConfig({"forcing": [], "diagnostic": [], "target": []}), name_to_index)


def test_profiler_example_input_uses_task_num_input_timesteps() -> None:
    """Profiler example inputs slice with the instantiated task, not forecaster-only config keys."""
    profiler = AnemoiProfiler.__new__(AnemoiProfiler)
    profiler.task = TemporalDownscaler(input_timestep="18h", output_timestep="6h")
    profiler.config = DictConfig({"task": {}, "dataloader": {"read_group_size": 1}})
    profiler.data_indices = {"data": _make_minimal_index_collection({"A": 0, "B": 1})}

    batch = {"data": torch.arange(16, dtype=torch.float32).reshape(1, 4, 1, 2, 2)}

    class _DataModule:
        def train_dataloader(self) -> list[dict[str, torch.Tensor]]:
            return [batch]

    profiler.datamodule = _DataModule()

    example_input_array = profiler.get_example_input_array()

    torch.testing.assert_close(
        example_input_array["data"],
        batch["data"][:, : profiler.task.num_input_timesteps, ..., profiler.data_indices["data"].data.input.full],
    )


@pytest.mark.parametrize(("rank", "filename"), [(0, "model_summary.txt"), (4, "model_summary_rank4.txt")])
def test_model_summary_written_per_rank(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rank: int,
    filename: str,
) -> None:
    profiler = SimpleNamespace(
        dirpath=tmp_path,
        _save_model_summary=lambda text, path: BenchmarkProfiler._save_model_summary(None, text, path),
    )
    model = MagicMock()
    model.to.return_value = model
    tensor = MagicMock()
    tensor.to.return_value = tensor
    example_input = {"data": tensor}

    def fake_summary(model_arg: MagicMock, *, input_data: tuple[dict[str, MagicMock]], **_kwargs: object) -> str:
        assert model_arg is model
        assert input_data == (example_input,)
        return f"Summary for rank {rank}"

    monkeypatch.setattr(torchinfo, "summary", fake_summary)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)

    result = BenchmarkProfiler.get_model_summary(profiler, model, example_input)

    assert result == f"Summary for rank {rank}"
    assert profiler.model_summary_fname == tmp_path / filename
    assert profiler.model_summary_fname.read_text() == result


def test_nonzero_rank_logs_model_summary(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    profiler = MagicMock(spec=AnemoiProfiler)
    profiler.model_summary = "Model summary for rank 4"
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 4)

    with caplog.at_level("INFO", logger="anemoi.training.train.profiler"):
        AnemoiProfiler.report(profiler)

    assert "Model Summary (rank 4):\nModel summary for rank 4" in caplog.text
