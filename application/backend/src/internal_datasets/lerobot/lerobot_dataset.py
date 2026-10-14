import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.dataset_tools import add_features as lerobot_add_features
from lerobot.datasets.dataset_tools import delete_episodes as lerobot_delete_episodes
from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.processor.pipeline import RobotProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from loguru import logger

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.dataset_client import DatasetClient
from internal_datasets.lerobot.streaming_encoding_settings import StreamingEncodingSettings
from internal_datasets.mutations.recording_mutation import RecordingMutation
from schemas import Episode, EpisodeInfo, EpisodeVideo
from settings import get_settings

EpisodeMetadata = dict[str, Any]


class InternalLeRobotDataset(DatasetClient):
    type: str = "lerobot"
    path: Path
    _dataset: LeRobotDataset

    _teleop_action_processor: RobotProcessorPipeline
    _robot_action_processor: RobotProcessorPipeline
    _robot_observation_processor: RobotProcessorPipeline
    _streaming_encoding_settings: StreamingEncodingSettings
    _access_mode: DatasetAccessMode

    def __init__(
        self,
        dataset_path: Path,
        *,
        streaming_encoding_settings: StreamingEncodingSettings = StreamingEncodingSettings(),
        access_mode: DatasetAccessMode = DatasetAccessMode.READ_ONLY,
    ):
        self.path = dataset_path
        self._streaming_encoding_settings = streaming_encoding_settings
        self._access_mode = access_mode
        self.load_dataset()

        self._teleop_action_processor, self._robot_action_processor, self._robot_observation_processor = (
            make_default_processors()
        )

    def load_dataset(self) -> None:
        """Load dataset."""
        if self._check_repository_exists(self.path):
            if self._access_mode is DatasetAccessMode.RECORDING_MUTATION:
                self._dataset = LeRobotDataset.resume(
                    repo_id=str(uuid4()),
                    root=self.path,
                    **self._resolved_streaming_encoding_settings_write(),
                )
            else:
                self._dataset = LeRobotDataset(
                    str(uuid4()),
                    self.path,
                )
            self.has_episodes = self._dataset.num_episodes > 0

    def _resolved_streaming_encoding_settings_write(self) -> dict:
        settings = self._streaming_encoding_settings.with_resolved_vcodec()
        return settings.to_lerobot_write_kwargs()

    def _resume_for_writing(self, repo_id: str | None = None) -> None:
        if not self._check_repository_exists(self.path):
            raise ValueError(f"Cannot resume non-existing dataset at {self.path}")

        resolved_repo_id = repo_id or getattr(getattr(self, "_dataset", None), "repo_id", str(uuid4()))
        self._dataset = LeRobotDataset.resume(
            repo_id=resolved_repo_id,
            root=self.path,
            **self._resolved_streaming_encoding_settings_write(),
        )
        self.has_episodes = self._dataset.num_episodes > 0

    def resume_dataset(self) -> None:
        """Load dataset in write mode for appending episodes."""
        if self._access_mode is not DatasetAccessMode.RECORDING_MUTATION:
            raise ValueError("Cannot resume dataset in write mode unless access_mode is RECORDING_MUTATION")
        self._resume_for_writing()

    def create(
        self,
        fps: int,
        features: dict,
        robot_type: str,
    ) -> None:
        """Create LeRobot dataset."""
        if self._check_repository_exists(self.path):
            raise Exception(f"Dataset already exists at {self.path}")
        self._dataset = LeRobotDataset.create(
            repo_id=str(uuid4()),
            root=self.path,
            fps=fps,
            features=features,
            robot_type=robot_type,
            use_videos=True,
            **self._resolved_streaming_encoding_settings_write(),
        )
        self.has_episodes = False

    def delete_episodes(self, episode_indices: list[int], output_path: Path) -> DatasetClient:
        """Copy over repo without given episode_indices to output_path."""
        lerobot_delete_episodes(dataset=self._dataset, episode_indices=episode_indices, output_dir=output_path)
        return InternalLeRobotDataset(output_path, streaming_encoding_settings=self._streaming_encoding_settings)

    def get_tasks(self) -> list[str]:
        """Get Tasks in dataset."""
        if not self.exists_on_disk:
            return []
        return list(self._dataset.meta.tasks.to_dict()["task_index"].keys())

    def get_video_path(self, episode: int, camera: str) -> Path:
        """Get Video path of specific episode and camera."""
        metadata = self._dataset.meta
        full_camera_name = f"observation.images.{camera}"
        return Path(metadata.root) / Path(metadata.get_video_file_path(episode, full_camera_name))

    def get_video_keys(self) -> list[str]:
        return list(self._dataset.meta.video_keys)

    def get_episode_thumbnail_png(
        self,
        episode_index: int,
        video_key: str,
        width: int = 320,
        height: int = 240,
    ) -> tuple[bytes, Path] | None:
        if not self.exists_on_disk:
            return None

        metadata = self._dataset.meta
        episode = self._find_episode_metadata(episode_index)
        if episode is None:
            return None

        video_path = Path(metadata.root) / Path(metadata.get_video_file_path(episode_index, video_key))
        if not video_path.is_file():
            return None

        image_key = video_key if video_key in self._dataset.meta.camera_keys else f"observation.images.{video_key}"
        thumbnail_png = self._build_thumbnail_png_bytes(episode, image_key, width, height)
        if thumbnail_png is None:
            return None

        return thumbnail_png, video_path

    def prepare_for_writing(self) -> None:
        """Start image writer &"""
        if self._dataset.writer is None:
            return
        if getattr(self._dataset.writer, "_streaming_encoder", None) is not None:
            logger.info("Streaming encoding enabled; skipping image writer startup")
            return
        num_threads = 4 * len(self._dataset.meta.camera_keys)
        self._dataset.writer.start_image_writer(
            num_processes=0,
            num_threads=num_threads,
        )

    def overwrite(self, source: DatasetClient) -> None:
        """Overwrite this dataset with the given dataset."""
        if not isinstance(source, InternalLeRobotDataset):
            raise ValueError(f"Cannot overwrite lerobot dataset with {source.__class__}")

        temporary_path = self.path.with_name(f"{self.path.name}.{uuid4()}.tmp")
        backup_path = self.path.with_name(f"{self.path.name}.{uuid4()}.backup")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(source.path, temporary_path)
            if self.path.exists():
                os.replace(self.path, backup_path)
            try:
                os.replace(temporary_path, self.path)
                self.load_dataset()
            except Exception:
                if self.path.exists():
                    shutil.rmtree(self.path)
                if backup_path.exists():
                    os.replace(backup_path, self.path)
                raise
            if backup_path.exists():
                shutil.rmtree(backup_path)
        finally:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)

    def get_episodes(self) -> list[Episode]:
        """Get episodes of dataset."""

        if not self.exists_on_disk:
            return []

        episodes = self._get_episode_metadata_list()
        interventions = self._get_episode_interventions()
        return [
            self._build_episode_from_metadata(episode, interventions.get(episode["episode_index"], False))
            for episode in episodes
        ]

    def find_episode(self, episode_index: int) -> Episode | None:
        """Find episode by index or return None."""
        if not self.exists_on_disk:
            return None

        episode = self._find_episode_metadata(episode_index)
        if episode is None:
            return None

        intervention = self._get_episode_interventions().get(episode_index, False)
        return self._build_episode_from_metadata(episode, intervention)

    def get_episode_infos(self) -> list[EpisodeInfo]:
        """Get lightweight episode summaries."""
        if not self.exists_on_disk:
            return []

        metadata = self._dataset.meta
        episodes = self._get_episode_metadata_list()
        interventions = self._get_episode_interventions()
        return [
            EpisodeInfo(
                episode_index=episode["episode_index"],
                tasks=episode["tasks"],
                length=episode["length"],
                fps=metadata.fps,
                intervention=interventions.get(episode["episode_index"], False),
            )
            for episode in episodes
        ]

    def add_frame(self, obs: dict, act: dict, task: str, intervention: bool = False) -> None:
        """Add frame to recording buffer."""
        frame = self._process_frame(obs, act, task, intervention)
        self._dataset.add_frame(frame)

    def save_episode(self) -> None:
        """Save current recording buffer as episode."""
        self._dataset.save_episode()

    def discard_buffer(self) -> None:
        """Discard current recording buffer."""
        self._dataset.clear_episode_buffer()

    def teardown(self) -> None:
        """Finalize dataset or delete if no episodes."""
        # TODO: Implement a wait for when an episode is still being written, but teardown is called.
        if self._dataset.num_episodes == 0:
            logger.info("Removing dataset since it has no episodes")
            self.delete()
        else:
            logger.info("Finalizing")
            self.finalize()

    def delete(self) -> None:
        """Delete dataset."""
        shutil.rmtree(self.path)

    def finalize(self) -> None:
        """Finalize changes to dataset."""
        logger.info(f"Finalizing dataset {self.path}")
        if self._dataset.writer is not None:
            self._dataset.writer.stop_image_writer()
        self._dataset.finalize()

    def _process_frame(self, obs: dict, act: dict, task: str, intervention: bool = False) -> dict:
        obs_processed = self._robot_observation_processor(obs)
        act_processed_teleop = self._teleop_action_processor((act, obs))
        action_frame = build_dataset_frame(self._dataset.features, act_processed_teleop, prefix=ACTION)
        observation_frame = build_dataset_frame(self._dataset.features, obs_processed, prefix=OBS_STR)
        frame = {**observation_frame, **action_frame, "task": task}
        if "intervention" in self._dataset.features:
            frame["intervention"] = np.array([intervention], dtype=bool)
        return frame

    def start_recording_mutation(self, fps: int, features: dict, robot_type: str) -> RecordingMutation:
        """Start recording mutation."""
        settings = get_settings()
        cache_dir = settings.cache_dir / str(uuid4())

        logger.info(f"Creating cache dataset {cache_dir}")
        cache_dataset = InternalLeRobotDataset(
            cache_dir,
            streaming_encoding_settings=self._streaming_encoding_settings,
        )
        if self.exists_on_disk:
            try:
                if "intervention" in self._dataset.features:
                    shutil.copytree(self.path, cache_dir)
                else:
                    intervention_feature = features.get("intervention")
                    if intervention_feature is None:
                        raise ValueError("Cannot migrate dataset without intervention feature metadata")
                    total_frames = int(self._dataset.meta.total_frames)
                    lerobot_add_features(
                        self._dataset,
                        {
                            "intervention": (
                                np.zeros((total_frames, 1), dtype=bool),
                                intervention_feature,
                            )
                        },
                        output_dir=cache_dir,
                        repo_id=getattr(self._dataset, "repo_id", None),
                    )
                    self._restore_migrated_metadata(self.path, cache_dir)
                    self._add_false_intervention_stats(cache_dir, total_frames)
                cache_dataset._resume_for_writing(getattr(self._dataset, "repo_id", None))
            except Exception:
                shutil.rmtree(cache_dir, ignore_errors=True)
                raise
        else:
            cache_dataset.create(fps=fps, features=features, robot_type=robot_type)

        return RecordingMutation(self, cache_dataset)

    @property
    def exists_on_disk(self) -> bool:
        """Check if repo exists."""
        return self._check_repository_exists(self.path)

    @staticmethod
    def _check_repository_exists(repo_path: Path) -> bool:
        """Check if repository path contains info and therefor exists."""
        return (repo_path / "meta/info.json").is_file()

    def _build_episode_from_buffer(self, episode: dict | None, task: str) -> Episode:
        """Build Episode object from buffer and episode dict."""
        data = self._build_episode_data_from_buffer()
        if data is None or self._dataset is None:
            raise Exception("No dataset loaded.")

        end = data["timestamp"][-1]
        episode_index = data["episode_index"].tolist()[0]
        video_timestamps = {
            video_key: EpisodeVideo(start=0, end=end, path="")  # TODO: Implement path
            for video_key in self._dataset.meta.video_keys
        }
        if episode is not None:
            for video_key in self._dataset.meta.video_keys:
                offset = episode[f"videos/{video_key}/to_timestamp"][-1]
                video_timestamps[video_key].start += offset
                video_timestamps[video_key].end += offset

        action_feature_names = self._dataset.features.get("action", {}).get("names", [])
        return Episode(
            episode_index=episode_index,
            length=len(data["frame_index"]),
            fps=self._dataset.fps,
            tasks=[task],
            actions=data["action"].tolist(),
            videos=video_timestamps,
            action_keys=action_feature_names,
            intervention=bool(np.any(data.get("intervention", False))),
        )

    def _build_episode_data_from_buffer(self) -> dict:
        """Build episode data from the buffer.

        LeRobotDataset V3 doesnt update episode data on save.
        In order to get the episode data we duplicate the actions that happen inside.
        """
        if self._dataset is None:
            raise Exception("No dataset loaded.")

        episode_buffer = self._get_episode_buffer_snapshot()
        if episode_buffer is None:
            raise Exception("Attempting to save episode, but no episode in buffer.")

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self._dataset.meta.total_frames, self._dataset.meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Update tasks and task indices with new tasks if any
        self._dataset.meta.save_episode_tasks(episode_tasks)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self._dataset.meta.get_task_index(task) for task in tasks])

        for key, ft in self._dataset.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        return episode_buffer

    def _get_episode_actions(self, episode: dict) -> torch.Tensor:
        """Get episode actions tensor from specific episode."""
        from_idx = episode["dataset_from_index"]
        to_idx = episode["dataset_to_index"]
        actions = self._dataset.hf_dataset["action"][from_idx:to_idx]
        return torch.stack(actions)

    def _build_episode_from_metadata(self, episode: EpisodeMetadata, intervention: bool = False) -> Episode:
        metadata = self._dataset.meta
        episode_index = episode["episode_index"]
        action_feature_names = self._dataset.features.get("action", {}).get("names", [])

        return Episode(
            actions=self._get_episode_actions(episode).tolist(),
            fps=metadata.fps,
            intervention=intervention,
            videos={
                video_key: EpisodeVideo(
                    start=episode[f"videos/{video_key}/from_timestamp"],
                    end=episode[f"videos/{video_key}/to_timestamp"],
                    path=str(metadata.get_video_file_path(episode_index, video_key)),
                )
                for video_key in self._dataset.meta.video_keys
            },
            action_keys=action_feature_names,
            **episode,
        )

    def _find_episode_metadata(self, episode_index: int) -> EpisodeMetadata | None:
        episodes = self._get_episode_metadata_list()
        for episode in episodes:
            if episode["episode_index"] == episode_index:
                return episode
        return None

    def _get_episode_metadata_list(self) -> list[EpisodeMetadata]:
        return [episode for episode in self._dataset.meta.episodes if isinstance(episode, dict)]

    def _get_episode_interventions(self) -> dict[int, bool]:
        if "intervention" not in self._dataset.features:
            return {}

        interventions: dict[int, bool] = {}
        missing_stats: set[int] = set()
        for episode in self._get_episode_metadata_list():
            episode_index = int(episode["episode_index"])
            max_values = episode.get("stats/intervention/max")
            if max_values is None:
                missing_stats.add(episode_index)
            elif bool(np.any(max_values)):
                interventions[episode_index] = True

        # Imported/partially converted LeRobot datasets may have the frame
        # column but no per-episode stats. Scan only those episodes as fallback.
        if missing_stats:
            frames = self._dataset.hf_dataset.select_columns(["episode_index", "intervention"]).with_format(None)
            for batch in frames.iter(batch_size=8192):
                for episode_index, intervention in zip(batch["episode_index"], batch["intervention"], strict=True):
                    if int(episode_index) in missing_stats and bool(np.any(intervention)):
                        interventions[int(episode_index)] = True
        return interventions

    @staticmethod
    def _restore_migrated_metadata(source_path: Path, migrated_path: Path) -> None:
        """Keep LeRobot settings that add_features recreates with defaults."""
        source_info_path = source_path / "meta/info.json"
        migrated_info_path = migrated_path / "meta/info.json"
        source_info = json.loads(source_info_path.read_text(encoding="utf-8"))
        migrated_info = json.loads(migrated_info_path.read_text(encoding="utf-8"))

        source_data_path = source_info.get("data_path", migrated_info["data_path"])
        if source_data_path != migrated_info["data_path"]:
            data_dir = migrated_path / "data"
            for parquet_path in sorted(data_dir.glob("chunk-*/file-*.parquet")):
                chunk_index = int(parquet_path.parent.name.removeprefix("chunk-"))
                file_index = int(parquet_path.stem.removeprefix("file-"))
                relative_path = Path(source_data_path.format(chunk_index=chunk_index, file_index=file_index))
                destination = migrated_path / relative_path
                if relative_path.is_absolute() or not destination.resolve().is_relative_to(migrated_path.resolve()):
                    raise ValueError(f"Unsafe LeRobot data_path in dataset metadata: {source_data_path}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(parquet_path, destination)
            migrated_info["data_path"] = source_data_path

        for key in ("codebase_version", "chunks_size", "data_files_size_in_mb", "video_files_size_in_mb", "tools"):
            if key in source_info:
                migrated_info[key] = source_info[key]
            elif key == "tools":
                migrated_info.pop(key, None)
        migrated_info["video_path"] = source_info.get("video_path", migrated_info.get("video_path"))
        migrated_info_path.write_text(json.dumps(migrated_info, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _add_false_intervention_stats(dataset_path: Path, total_frames: int) -> None:
        """Add the all-false legacy column's stats omitted by LeRobot 0.6.0 add_features."""
        if not total_frames:
            return

        feature_stats = {
            stat: np.array([total_frames if stat == "count" else 0.0])
            for stat in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
        }
        stats = load_stats(dataset_path) or {}
        stats["intervention"] = feature_stats
        write_stats(stats, dataset_path)

        for episodes_path in sorted((dataset_path / "meta/episodes").rglob("*.parquet")):
            episodes = pd.read_parquet(episodes_path)
            for stat in feature_stats:
                episodes[f"stats/intervention/{stat}"] = [
                    np.array([int(row["length"]) if stat == "count" else 0.0]) for _, row in episodes.iterrows()
                ]
            episodes.to_parquet(episodes_path, index=False)

    def _build_thumbnail_png_bytes(self, episode: dict, image_key: str, width: int, height: int) -> bytes | None:
        if image_key not in self._dataset.meta.camera_keys:
            logger.warning("Unknown thumbnail camera key '{}'", image_key)
            return None

        from_idx = int(episode["dataset_from_index"])
        item = self._read_dataset_item_for_thumbnail(from_idx)
        if item is None:
            logger.warning("Could not read dataset item for thumbnail at index {}", from_idx)
            return None

        try:
            image = item[image_key].permute(1, 2, 0).detach().numpy()
        except Exception:
            logger.exception("Could not extract image '{}' for thumbnail", image_key)
            return None

        rescaled = (image * 255).clip(0, 255).astype(np.uint8)
        resized = cv2.resize(rescaled, (width, height))
        bgr_image = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        encoded, imagebytes = cv2.imencode(".png", bgr_image)
        if not encoded:
            logger.warning("Failed to encode thumbnail PNG for image key '{}'", image_key)
            return None

        return imagebytes.tobytes()

    def _get_episode_buffer_snapshot(self) -> dict | None:
        writer = self._dataset.writer
        if writer is None:
            return None
        return copy.deepcopy(writer.episode_buffer)

    def _read_dataset_item_for_thumbnail(self, index: int) -> dict | None:
        try:
            return self._dataset[index]
        except RuntimeError as exc:
            if "Cannot read from a dataset that is being recorded" not in str(exc):
                logger.exception("Could not read dataset item for thumbnail")
                return None

            logger.warning("Dataset is in recording mode during thumbnail read; using reader fallback")
            try:
                if self._dataset.reader is None:
                    _ = self._dataset.hf_dataset

                reader = self._dataset.reader
                if reader is None:
                    return None

                if reader.hf_dataset is None:
                    reader.load_and_activate()

                return reader.get_item(index)
            except Exception:
                logger.exception("Could not read dataset item from reader fallback")
                return None
        except Exception:
            logger.exception("Could not read dataset item for thumbnail")
            return None
