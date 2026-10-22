# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot depth units, lossless offline preprocessing, and cached teacher targets."""

from pathlib import Path

import av
import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from physicalai.data import FeatureType
from physicalai.data.datamodules import _collate_observations
from physicalai.data.lerobot.converters import FormatConverter
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter
from physicalai.data.lerobot.feature_cache import TeacherFeatureCache, dataset_fingerprint
from physicalai.data.lerobot.preprocessing import preprocess_dataset

RGB = "observation.images.front"
DEPTH = "observation.images.front_depth"


def make_dataset(root: Path, *, depth: bool = False, cameras: tuple[str, ...] = (RGB,)) -> Path:
    features: dict[str, dict[str, object]] = {
        "observation.state": {"dtype": "float32", "shape": [6], "names": None},
        "action": {"dtype": "float32", "shape": [6], "names": None},
    }
    for camera in cameras:
        features[camera] = {"dtype": "image", "shape": [32, 32, 3], "names": ["height", "width", "channels"]}
    if depth:
        features[DEPTH] = {
            "dtype": "image",
            "shape": [32, 32, 1],
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True, "depth_unit": "mm", "aligned_to": RGB},
        }
    dataset = LeRobotDataset.create(repo_id="test/depth", root=root, fps=10, features=features, use_videos=False)
    for episode in range(2):
        for frame in range(2):
            value = (0, 100, 200, 250)[2 * episode + frame]
            sample = {
                "observation.state": np.ones(6, dtype=np.float32) * frame,
                "action": np.ones(6, dtype=np.float32),
                "task": "move",
            }
            sample.update({camera: np.full((32, 32, 3), value, dtype=np.uint8) for camera in cameras})
            if depth:
                sample[DEPTH] = np.full((32, 32, 1), 1500, dtype=np.uint16)
            dataset.add_frame(sample)
        dataset.save_episode()
    dataset.finalize()
    return root


class DepthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def infer(self, images, **kwargs):
        self.batch_sizes.append(images.shape[0])
        return {"depth": images.mean(1) + 1.0}


class Targets:
    references = {"teacher": "fixture"}

    def __call__(self, batch):
        current = batch["teacher_current_rgb"].mean().item()
        future = batch["teacher_future_rgb"].mean().item()
        return {
            f"{name}_target": torch.full((1, 256, 1024), future if name.startswith("future") else current)
            for name in ("depth", "future_depth", "current_video", "future_video")
        }


def test_native_depth_is_metric_and_never_an_rgb_camera(tmp_path):
    root = make_dataset(tmp_path / "source", depth=True)
    adapter = _LeRobotDatasetAdapter(repo_id="test/depth", root=root)
    sample = adapter[0]
    assert set(sample.images) == {"front"}
    assert set(sample.depths) == {DEPTH}
    torch.testing.assert_close(sample.depths[DEPTH], torch.full((1, 32, 32), 1.5))
    assert adapter.observation_features[f"depths.{DEPTH}"].ftype == FeatureType.DEPTH
    batch = _collate_observations([sample, adapter[1]])
    assert batch.depths[DEPTH].shape == (2, 1, 32, 32)
    assert batch[0].depths[DEPTH].shape == (1, 32, 32)
    assert FormatConverter.to_lerobot_dict(batch)[DEPTH].shape == (2, 1, 32, 32)
    torch.testing.assert_close(batch.to_numpy().to_torch().depths[DEPTH], batch.depths[DEPTH])


def test_offline_depth_copy_preserves_rgb_and_source(tmp_path):
    wrist = "observation.images.wrist"
    source = make_dataset(tmp_path / "source", cameras=(RGB, wrist))
    before = dataset_fingerprint(source)
    progress = []
    output = preprocess_dataset(
        source, tmp_path / "depth", operation="depth", device="cpu", models=DepthModel(), progress=progress.append
    )
    assert dataset_fingerprint(source) == before
    dataset = LeRobotDataset(repo_id="test/depth", root=output, depth_output_unit="m")
    assert set(dataset.meta.depth_keys) == {DEPTH, f"{wrist}_depth"}
    assert progress[-1] == len(dataset)
    assert progress == sorted(progress)
    for camera in (RGB, wrist):
        for episode in range(2):
            path = output / dataset.features[f"{camera}_depth"]["info"]["preview_path"].format(episode_index=episode)
            with av.open(str(path)) as preview:
                assert len(list(preview.decode(video=0))) == 2
    assert dataset.features[DEPTH]["info"]["aligned_to"] == RGB
    torch.testing.assert_close(dataset[0][DEPTH], torch.ones(1, 32, 32))
    torch.testing.assert_close(dataset[1][RGB], LeRobotDataset(repo_id="test/source", root=source)[1][RGB])
    with pytest.raises(ValueError, match="new directory"):
        preprocess_dataset(source, output, operation="depth", models=DepthModel(), device="cpu")
    repeated = preprocess_dataset(output, tmp_path / "again", operation="depth", models=object(), device="cpu")
    assert dataset_fingerprint(repeated) == dataset_fingerprint(output)


def test_depth_batch_size_reaches_inference_and_keeps_the_final_partial_batch(tmp_path):
    source = make_dataset(tmp_path / "source")
    model = DepthModel()
    output = preprocess_dataset(source, tmp_path / "depth", operation="depth", batch_size=3, models=model, device="cpu")
    assert model.batch_sizes == [3, 1]
    dataset = LeRobotDataset(repo_id="test/depth", root=output, depth_output_unit="m")
    assert len(dataset) == 4
    torch.testing.assert_close(dataset[0][DEPTH], torch.ones(1, 32, 32))
    torch.testing.assert_close(dataset[3][DEPTH], torch.full((1, 32, 32), 1 + 250 / 255))


@pytest.mark.parametrize("batch_size", [0, -1, 65])
def test_invalid_depth_batch_size_is_rejected(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch size"):
        preprocess_dataset(tmp_path / "source", tmp_path / "depth", operation="depth", batch_size=batch_size)


def test_existing_depth_is_preserved_while_other_cameras_get_depth(tmp_path):
    wrist = "observation.images.wrist"
    source = make_dataset(tmp_path / "source", depth=True, cameras=(RGB, wrist))
    output = preprocess_dataset(source, tmp_path / "depth", operation="depth", models=DepthModel(), device="cpu")
    dataset = LeRobotDataset(repo_id="test/depth", root=output, depth_output_unit="m")
    assert set(dataset.meta.depth_keys) == {DEPTH, f"{wrist}_depth"}
    assert dataset.features[DEPTH]["info"]["depth_unit"] == "mm"
    for key in dataset.meta.depth_keys:
        preview_path = output / dataset.features[key]["info"]["preview_path"].format(episode_index=0)
        with av.open(str(preview_path)) as preview:
            assert len(list(preview.decode(video=0))) == 2
    torch.testing.assert_close(dataset[0][DEPTH], torch.full((1, 32, 32), 1.5))
    torch.testing.assert_close(dataset[0][f"{wrist}_depth"], torch.ones(1, 32, 32))


def test_source_changes_are_not_overwritten(tmp_path):
    source = make_dataset(tmp_path / "source")

    def change_source(done):
        with (source / "meta/info.json").open("a") as file:
            file.write(" ")

    with pytest.raises(ValueError, match="Source dataset changed"):
        preprocess_dataset(
            source, tmp_path / "depth", operation="depth", models=DepthModel(), device="cpu", progress=change_source
        )
    assert not (tmp_path / "depth").exists()


def test_teacher_cache_preserves_episode_boundaries_and_absolute_indices(tmp_path):
    source = make_dataset(tmp_path / "source", depth=True)
    output = preprocess_dataset(
        source,
        tmp_path / "targets",
        operation="teacher_targets",
        rgb_key=RGB,
        depth_key=DEPTH,
        chunk_size=2,
        models=Targets(),
        device="cpu",
    )
    cache = TeacherFeatureCache(output, chunk_size=2, tokens=256, feature_dim=1024)
    assert cache(1)["teacher_features.future_depth"].mean() == 100
    assert cache(2)["teacher_features.future_depth"].mean() == 250
    adapter = _LeRobotDatasetAdapter(repo_id="test/targets", root=output, episodes=[1])
    adapter.extra_features = cache
    assert adapter[0].extra["teacher_features.future_depth"].mean() == 250
    with pytest.raises(ValueError, match="horizon"):
        TeacherFeatureCache(output, chunk_size=50, tokens=256, feature_dim=1024)
    with (output / "meta/info.json").open("a") as file:
        file.write(" ")
    with pytest.raises(ValueError, match="stale"):
        TeacherFeatureCache(output, chunk_size=2, tokens=256, feature_dim=1024)


def test_external_image_references_are_rejected(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = make_dataset(tmp_path / "source")
    path = next((source / "data").glob("*/*.parquet"))
    table = pq.read_table(path)
    values = pa.array(
        [{"path": str(tmp_path / "private.png"), "bytes": None}] * table.num_rows, type=table.schema.field(RGB).type
    )
    table = table.set_column(table.schema.get_field_index(RGB), RGB, values)
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="Image references"):
        preprocess_dataset(source, tmp_path / "result", operation="depth", models=DepthModel(), device="cpu")
    assert not (tmp_path / "result").exists()


def test_failed_preprocessing_does_not_publish_or_mutate(tmp_path):
    source = make_dataset(tmp_path / "source")
    before = dataset_fingerprint(source)

    def cancel(done):
        raise RuntimeError("cancel")

    with pytest.raises(RuntimeError, match="cancel"):
        preprocess_dataset(
            source, tmp_path / "result", operation="depth", models=DepthModel(), device="cpu", progress=cancel
        )
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".preprocess-*"))
    assert dataset_fingerprint(source) == before
