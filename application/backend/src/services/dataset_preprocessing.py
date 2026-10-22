from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from loguru import logger

from db import get_async_db_session_ctx
from schemas import Dataset
from schemas.base_job import JobStatus, JobType
from schemas.dataset_preprocessing import DatasetPreprocessingInfo, DatasetPreprocessingRequest, PreprocessingFeature
from services.dataset_service import DatasetService
from services.event_processor import EventType
from services.job_service import JobService
from settings import get_settings

if TYPE_CHECKING:
    import multiprocessing as mp
    from collections.abc import Callable

    from schemas.job import DatasetPreprocessingJob


def preprocessing_info(dataset: Dataset) -> DatasetPreprocessingInfo:
    """Read camera modalities without decoding images or allocating GPU models."""
    info = json.loads((Path(dataset.path) / "meta/info.json").read_text())
    features = []
    for key, feature in info["features"].items():
        if feature.get("dtype") not in {"image", "video"}:
            continue
        metadata = feature.get("info") or {}
        legacy = feature.get("video_info") or {}
        features.append(
            PreprocessingFeature(
                key=key,
                shape=feature["shape"],
                is_depth=bool(
                    metadata.get("is_depth_map")
                    or metadata.get("video.is_depth_map")
                    or legacy.get("video.is_depth_map")
                ),
                unit=metadata.get("depth_unit"),
                aligned_to=metadata.get("aligned_to"),
            )
        )
    return DatasetPreprocessingInfo(
        features=features,
        frames=info["total_frames"],
        has_teacher_cache=(Path(dataset.path) / ".physicalai/lingbotvla2/manifest.json").is_file(),
    )


def validate_preprocessing(dataset: Dataset, request: DatasetPreprocessingRequest) -> None:
    """Reject empty datasets, conflicting depth names, and undeclared alignment."""
    info = preprocessing_info(dataset)
    by_name = {feature.key: feature for feature in info.features}
    rgb = [feature for feature in info.features if not feature.is_depth]
    if info.frames < 1 or not rgb:
        raise ValueError("Select a nonempty dataset with RGB cameras")
    if request.operation == "depth":
        raw = json.loads((Path(dataset.path) / "meta/info.json").read_text())
        for feature in rgb:
            if any(depth.is_depth and depth.aligned_to == feature.key for depth in info.features):
                continue
            if f"{feature.key}_depth" in raw["features"]:
                raise ValueError(f"Depth feature name conflicts with existing data: {feature.key}_depth")
        return
    if request.rgb_key is None or request.rgb_key not in by_name or by_name[request.rgb_key].is_depth:
        raise ValueError("Select an RGB source camera")
    if request.depth_key is not None:
        depth = by_name.get(request.depth_key)
        if depth is None or not depth.is_depth or depth.unit not in {"m", "mm"}:
            raise ValueError("Selected depth needs LeRobot depth metadata and explicit units")
        if depth.aligned_to != request.rgb_key and not request.depth_aligned:
            raise ValueError("Confirm calibrated alignment between depth and RGB")


def recover_preprocessing_job(job: DatasetPreprocessingJob) -> bool:
    """Restore an interrupted directory swap; return whether publication finished."""
    settings = get_settings()
    root = settings.datasets_dir / f".preprocessing-{job.id}"
    source = settings.datasets_dir / str(job.payload.dataset_id)
    backup = root / "original"
    published = backup.exists() and source.exists() and not (root / "dataset").exists()
    if backup.exists() and not source.exists():
        backup.rename(source)
    if root.exists():
        shutil.rmtree(root)
    return published


async def run_preprocessing_job(  # noqa: C901, PLR0912, PLR0915 -- shared cleanup/publication lifecycle
    job: DatasetPreprocessingJob, queue: mp.Queue, should_stop: Callable[[], bool]
) -> None:
    """Use the local training execution slot and an isolated GPU subprocess."""
    settings = get_settings()
    root = settings.datasets_dir / f".preprocessing-{job.id}"
    output = root / "dataset"
    backup = root / "original"
    process = None
    source: Dataset | None = None
    root_created = False
    last_progress = -1

    async def update(status: JobStatus, message: str, progress: int | None = None) -> None:
        async with get_async_db_session_ctx() as session:
            service = JobService(session)
            current = await service.get_job_by_id(job.id)
            if current.status == JobStatus.CANCELED and status != JobStatus.CANCELED:
                raise asyncio.CancelledError
            result = await service.update_job_status(job.id, status, message=message, progress=progress)
        queue.put((EventType.JOB_UPDATE, result))
        logger.info("{}: {} ({}%)", status.value, message, progress if progress is not None else result.progress)

    try:
        async with get_async_db_session_ctx() as session:
            if await JobService(session).repo.claim_pending_job(job.id, JobType.DATASET_PREPROCESSING) is None:
                return
            source = await DatasetService(session).get_dataset_by_id(job.payload.dataset_id)
        if source.project_id != job.project_id:
            raise ValueError("Dataset does not belong to this job's project")
        validate_preprocessing(source, job.payload)
        root.mkdir(parents=True, exist_ok=False)
        root_created = True
        await update(JobStatus.RUNNING, "Preparing dataset preprocessing", 0)
        command = [
            sys.executable,
            "-m",
            "physicalai.data.lerobot.preprocessing",
            source.path,
            str(output),
            "--operation",
            job.payload.operation,
            "--chunk-size",
            str(job.payload.chunk_size),
            "--batch-size",
            str(job.payload.batch_size),
            "--device",
            f"cuda:{job.payload.device_index}",
            "--parent-pid",
            str(os.getpid()),
        ]
        if job.payload.rgb_key is not None:
            command.extend(["--rgb-key", job.payload.rgb_key])
        if job.payload.depth_key is not None:
            command.extend(["--depth-key", job.payload.depth_key])
        if job.payload.depth_aligned:
            command.append("--depth-aligned")
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        while process.returncode is None:
            if should_stop():
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                await update(JobStatus.CANCELED, "Dataset preprocessing canceled; source unchanged")
                return
            try:
                line = await asyncio.wait_for(cast("asyncio.StreamReader", process.stdout).readline(), timeout=0.5)
            except TimeoutError:
                continue
            if not line:
                await asyncio.sleep(0.05)
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            progress = event.get("progress") if isinstance(event, dict) else None
            if type(progress) is int and 0 <= progress <= 99 and progress != last_progress:
                last_progress = progress
                await update(
                    JobStatus.RUNNING,
                    "Computing depth" if job.payload.operation == "depth" else "Caching teacher features",
                    progress,
                )
        if process.returncode != 0:
            raise RuntimeError("Preprocessing subprocess failed; check dataset metadata and model availability")
        if should_stop():
            await update(JobStatus.CANCELED, "Dataset preprocessing canceled; source unchanged")
            return
        original = Path(source.path)
        original.rename(backup)
        try:
            output.rename(original)
        except BaseException:
            backup.rename(original)
            raise
        await update(JobStatus.COMPLETED, "Preprocessing ready in this dataset", 100)
    except Exception as error:
        message = (
            str(error) if isinstance(error, ValueError) else f"Dataset preprocessing failed ({type(error).__name__})"
        )
        await update(JobStatus.FAILED, message)
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                process.kill()
                await process.wait()
        if root_created and root.exists():
            if source is not None and backup.exists() and not Path(source.path).exists():
                backup.rename(source.path)
            shutil.rmtree(root)
