import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from lerobot.configs import RGBEncoderConfig

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.lerobot.streaming_encoding_settings import StreamingEncodingSettings
from runtime.dataset_features import build_lerobot_dataset_features
from services.dataset_download_service import DatasetDownloadService
from settings import Settings


def test_streaming_settings_translate_to_lerobot_kwargs() -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=4,
        encoder_queue_maxsize=60,
    )

    kwargs = settings.to_lerobot_write_kwargs()

    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_threads"] == 4
    assert kwargs["encoder_queue_maxsize"] == 60
    assert isinstance(kwargs["rgb_encoder"], RGBEncoderConfig)
    assert kwargs["rgb_encoder"].vcodec == "h264"
    assert kwargs["rgb_encoder"].g is None
    assert "vcodec" not in kwargs


def test_create_uses_rgb_encoder_and_not_vcodec(tmp_path: Path) -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.READ_ONLY

    with (
        patch.object(InternalLeRobotDataset, "_check_repository_exists", return_value=False),
        patch(
            "internal_datasets.lerobot.lerobot_dataset.LeRobotDataset.create", return_value=MagicMock()
        ) as create_mock,
    ):
        dataset.create(fps=30, features={}, robot_type="so100")

    kwargs = create_mock.call_args.kwargs
    assert isinstance(kwargs["rgb_encoder"], RGBEncoderConfig)
    assert kwargs["rgb_encoder"].vcodec == "h264"
    assert kwargs["rgb_encoder"].g is None
    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_threads"] == 2
    assert kwargs["encoder_queue_maxsize"] == 60
    assert "vcodec" not in kwargs


def test_load_dataset_is_read_only_and_does_not_pass_write_kwargs(tmp_path: Path) -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.READ_ONLY

    with (
        patch.object(InternalLeRobotDataset, "_check_repository_exists", return_value=True),
        patch(
            "internal_datasets.lerobot.lerobot_dataset.LeRobotDataset",
            return_value=MagicMock(num_episodes=1),
        ) as init_mock,
    ):
        dataset.load_dataset()

    kwargs = init_mock.call_args.kwargs
    assert "rgb_encoder" not in kwargs
    assert "streaming_encoding" not in kwargs
    assert "encoder_threads" not in kwargs
    assert "encoder_queue_maxsize" not in kwargs
    assert "vcodec" not in kwargs


def test_resume_dataset_uses_write_kwargs_and_not_vcodec(tmp_path: Path) -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.RECORDING_MUTATION

    with (
        patch.object(InternalLeRobotDataset, "_check_repository_exists", return_value=True),
        patch(
            "internal_datasets.lerobot.lerobot_dataset.LeRobotDataset.resume",
            return_value=MagicMock(num_episodes=1),
        ) as resume_mock,
    ):
        dataset.resume_dataset()

    kwargs = resume_mock.call_args.kwargs
    assert isinstance(kwargs["rgb_encoder"], RGBEncoderConfig)
    assert kwargs["rgb_encoder"].vcodec == "h264"
    assert kwargs["rgb_encoder"].g is None
    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_threads"] == 2
    assert kwargs["encoder_queue_maxsize"] == 60
    assert "vcodec" not in kwargs


def test_resume_dataset_raises_in_read_only_mode(tmp_path: Path) -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.READ_ONLY

    with patch.object(InternalLeRobotDataset, "_resume_for_writing") as resume_mock:
        try:
            dataset.resume_dataset()
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "RECORDING_MUTATION" in str(exc)
    resume_mock.assert_not_called()


def _dataset_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    settings = Settings(STORAGE_DIR=str(tmp_path / "storage"))
    monkeypatch.setattr("internal_datasets.lerobot.lerobot_dataset.get_settings", lambda: settings)
    return settings


def _features(*, intervention: bool = True) -> dict:
    features = build_lerobot_dataset_features(joint_names=["joint", "joint2"], camera_specs={}, use_videos=False)
    if not intervention:
        features.pop("intervention")
    return features


def _dataset(path: Path, *, intervention: bool = True) -> InternalLeRobotDataset:
    dataset = InternalLeRobotDataset(
        path,
        streaming_encoding_settings=StreamingEncodingSettings(streaming_encoding=False),
    )
    dataset.create(fps=10, features=_features(intervention=intervention), robot_type="test_robot")
    return dataset


def _write_episode(dataset: InternalLeRobotDataset, intervention_flags: list[bool]) -> None:
    for index, intervention in enumerate(intervention_flags):
        observation = {"joint.pos": float(index), "joint2.pos": float(index + 1)}
        action = {"joint.pos": float(index), "joint2.pos": float(index + 1)}
        dataset.add_frame(observation, action, "move the joint", intervention)
    dataset.save_episode()


def _read_intervention(dataset: InternalLeRobotDataset) -> list[bool]:
    return [bool(row["intervention"]) for row in dataset._dataset.hf_dataset.select_columns("intervention")]


def test_new_dataset_records_false_intervention_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _dataset_settings(monkeypatch, tmp_path)
    root = tmp_path / "new-dataset"
    source = InternalLeRobotDataset(
        root,
        streaming_encoding_settings=StreamingEncodingSettings(streaming_encoding=False),
    )
    with patch.object(InternalLeRobotDataset, "prepare_for_writing"):
        mutation = source.start_recording_mutation(fps=10, features=_features(), robot_type="test_robot")
    observation = {"joint.pos": 1.0, "joint2.pos": 2.0}
    action = {"joint.pos": 1.0, "joint2.pos": 2.0}
    mutation.add_frame(observation, action, "move the joint")
    mutation.save_episode()
    mutation.teardown()

    recorded = InternalLeRobotDataset(
        root,
        streaming_encoding_settings=StreamingEncodingSettings(streaming_encoding=False),
    )
    assert recorded._dataset.features["intervention"] == {"dtype": "bool", "shape": (1,), "names": None}
    assert _read_intervention(recorded) == [False]
    assert recorded.get_episode_infos()[0].intervention is False
    assert recorded.find_episode(0).intervention is False


def test_existing_dataset_migrates_and_keeps_per_frame_labels_after_copyback_and_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dataset_settings(monkeypatch, tmp_path)
    root = tmp_path / "legacy-dataset"
    legacy_writer = _dataset(root, intervention=False)
    _write_episode(legacy_writer, [False, False])
    legacy_writer.finalize()

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["data_path"] = "data/chunk-{chunk_index:03d}/frame-{file_index:03d}.parquet"
    info["chunks_size"] = 32
    info["data_files_size_in_mb"] = 128
    info["video_files_size_in_mb"] = 256
    info["tools"] = [{"type": "function", "name": "test_tool"}]
    (root / "data/chunk-000/file-000.parquet").rename(root / "data/chunk-000/frame-000.parquet")
    info_path.write_text(json.dumps(info), encoding="utf-8")

    legacy = InternalLeRobotDataset(
        root,
        streaming_encoding_settings=StreamingEncodingSettings(streaming_encoding=False),
    )
    assert [episode.intervention for episode in legacy.get_episode_infos()] == [False]
    assert [episode.intervention for episode in legacy.get_episodes()] == [False]
    assert legacy.find_episode(0).intervention is False
    original_state_stats = {key: value.copy() for key, value in legacy._dataset.meta.stats["observation.state"].items()}
    old_episode_metadata = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")

    with patch.object(InternalLeRobotDataset, "prepare_for_writing"):
        mutation = legacy.start_recording_mutation(fps=10, features=_features(), robot_type="test_robot")
    migrated_cache = mutation.cache_dataset._dataset
    assert migrated_cache.features["intervention"] == {"dtype": "bool", "shape": (1,), "names": None}
    assert migrated_cache.meta.info.data_path == "data/chunk-{chunk_index:03d}/frame-{file_index:03d}.parquet"
    assert migrated_cache.meta.info.chunks_size == 32
    assert migrated_cache.meta.info.data_files_size_in_mb == 128
    assert migrated_cache.meta.info.video_files_size_in_mb == 256
    assert migrated_cache.meta.info.tools == [{"type": "function", "name": "test_tool"}]
    assert (mutation.cache_dataset.path / "data/chunk-000/frame-000.parquet").is_file()
    migrated_values = [bool(row["intervention"]) for row in migrated_cache.hf_dataset.select_columns("intervention")]
    assert migrated_values == [False, False]
    assert set(migrated_cache.meta.stats["intervention"]) == {
        "min",
        "max",
        "mean",
        "std",
        "count",
        "q01",
        "q10",
        "q50",
        "q90",
        "q99",
    }
    assert migrated_cache.meta.stats["intervention"]["count"].tolist() == [2]
    for key, value in original_state_stats.items():
        np.testing.assert_array_equal(migrated_cache.meta.stats["observation.state"][key], value)
    migrated_episode_metadata = pd.read_parquet(
        mutation.cache_dataset.path / "meta/episodes/chunk-000/file-000.parquet"
    )
    for key in ("stats/action/mean", "stats/observation.state/mean"):
        np.testing.assert_equal(old_episode_metadata.at[0, key], migrated_episode_metadata.at[0, key])
    assert migrated_episode_metadata.at[0, "stats/intervention/count"] == [2]

    mutation.add_frame(
        {"joint.pos": 0.5, "joint2.pos": 1.5},
        {"joint.pos": 0.5, "joint2.pos": 1.5},
        "move the joint",
        False,
    )
    mutation.add_frame(
        {"joint.pos": 0.75, "joint2.pos": 1.75},
        {"joint.pos": 0.75, "joint2.pos": 1.75},
        "move the joint",
        True,
    )
    mutation.save_episode()
    mutation.teardown()

    recorded = InternalLeRobotDataset(
        root,
        streaming_encoding_settings=StreamingEncodingSettings(streaming_encoding=False),
    )
    assert _read_intervention(recorded) == [False, False, False, True]
    assert [episode.intervention for episode in recorded.get_episode_infos()] == [False, True]
    assert [episode.intervention for episode in recorded.get_episodes()] == [False, True]
    assert recorded.find_episode(1).intervention is True
    assert recorded._dataset.meta.stats["intervention"]["count"].tolist() == [4]

    archive = DatasetDownloadService().create_dataset_archive(root)
    imported_path = tmp_path / "export-import-roundtrip"
    imported_path.mkdir()
    with zipfile.ZipFile(archive) as dataset_archive:
        dataset_archive.extractall(imported_path)
    archive.unlink()
    imported = InternalLeRobotDataset(
        imported_path,
        streaming_encoding_settings=StreamingEncodingSettings(streaming_encoding=False),
    )
    assert _read_intervention(imported) == [False, False, False, True]
    assert [episode.intervention for episode in imported.get_episode_infos()] == [False, True]

    from physicalai.data import LeRobotDataModule

    data_module = LeRobotDataModule(
        dataset=recorded._dataset,
        train_batch_size=2,
        num_workers=0,
        persistent_workers=False,
    )
    batch = next(iter(data_module.train_dataloader()))
    assert batch.extra is not None and "intervention" in batch.extra
    assert batch.extra["intervention"].dtype == torch.bool
    assert batch.extra["intervention"].shape[0] == 2
    assert "intervention" not in data_module.train_dataset.observation_features
    assert "intervention" not in data_module.train_dataset.action_features

    deleted = recorded.delete_episodes([0], tmp_path / "dataset-after-delete")
    assert [(episode.episode_index, episode.intervention) for episode in deleted.get_episode_infos()] == [(0, True)]
    assert deleted.get_episodes()[0].intervention is True
    assert _read_intervention(deleted) == [False, True]


def test_migration_failure_leaves_original_dataset_intact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _dataset_settings(monkeypatch, tmp_path)
    root = tmp_path / "legacy-failure"
    writer = _dataset(root, intervention=False)
    _write_episode(writer, [False, False])
    writer.finalize()

    original_info = (root / "meta/info.json").read_bytes()
    original_data = {path.relative_to(root): path.read_bytes() for path in (root / "data").rglob("*") if path.is_file()}
    source = InternalLeRobotDataset(
        root,
        streaming_encoding_settings=StreamingEncodingSettings(streaming_encoding=False),
    )

    def fail_after_partial_output(*_args, output_dir: Path, **_kwargs) -> None:
        output_dir.mkdir(parents=True)
        (output_dir / "partial-migration").touch()
        raise RuntimeError("feature migration failed")

    with (
        patch(
            "internal_datasets.lerobot.lerobot_dataset.lerobot_add_features",
            side_effect=fail_after_partial_output,
        ),
        pytest.raises(RuntimeError, match="feature migration failed"),
    ):
        source.start_recording_mutation(fps=10, features=_features(), robot_type="test_robot")

    assert (root / "meta/info.json").read_bytes() == original_info
    current_data = {path.relative_to(root): path.read_bytes() for path in (root / "data").rglob("*") if path.is_file()}
    assert current_data == original_data
    assert not list(settings.cache_dir.glob("*/partial-migration"))
