import json
import queue
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_dataset_service, get_job_service
from main import app
from schemas import Dataset
from schemas.base_job import JobStatus
from schemas.dataset_preprocessing import DatasetPreprocessingPayload, DatasetPreprocessingRequest
from schemas.job import DatasetPreprocessingJob
from services.dataset_preprocessing import recover_preprocessing_job, run_preprocessing_job
from settings import Settings

RGB = "observation.images.front"
DEPTH = "observation.images.depth"


@pytest.fixture
def data(tmp_path, monkeypatch):
    settings = Settings(STORAGE_DIR=str(tmp_path))
    monkeypatch.setattr("schemas.dataset.get_settings", lambda: settings)
    monkeypatch.setattr("services.dataset_preprocessing.get_settings", lambda: settings)
    dataset = Dataset(name="source", default_task="move", project_id=uuid4(), environment_id=uuid4())
    root = Path(dataset.path)
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "total_frames": 4,
                "features": {
                    RGB: {"dtype": "video", "shape": [32, 32, 3]},
                    DEPTH: {"dtype": "image", "shape": [32, 32, 1], "info": {"is_depth_map": True, "depth_unit": "mm"}},
                },
            }
        )
    )
    return settings, dataset


def test_api_preserves_depth_metadata_and_requires_alignment(data):
    _, dataset = data
    service = SimpleNamespace(get_dataset_by_id=AsyncMock(return_value=dataset))
    jobs = SimpleNamespace(create_job=AsyncMock(side_effect=lambda job: job))
    app.dependency_overrides[get_dataset_service] = lambda: service
    app.dependency_overrides[get_job_service] = lambda: jobs
    try:
        client = TestClient(app)
        info = client.get(f"/api/dataset/{dataset.id}/preprocessing")
        assert info.status_code == 200
        assert info.json()["features"][1]["unit"] == "mm"
        depth_response = client.post(
            f"/api/dataset/{dataset.id}/preprocessing", json={"operation": "depth", "batch_size": 4}
        )
        assert depth_response.status_code == 202
        assert depth_response.json()["payload"]["dataset_id"] == str(dataset.id)
        assert depth_response.json()["payload"]["batch_size"] == 4
        assert "result_dataset_id" not in depth_response.json()["payload"]
        body = {"operation": "teacher_targets", "rgb_key": RGB, "depth_key": DEPTH}
        assert client.post(f"/api/dataset/{dataset.id}/preprocessing", json=body).status_code == 422
        response = client.post(f"/api/dataset/{dataset.id}/preprocessing", json={**body, "depth_aligned": True})
        assert response.status_code == 202
        job = response.json()
        assert job["type"] == "dataset_preprocessing" and job["status"] == "pending"
        assert job["payload"]["dataset_id"] == str(dataset.id)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("operation", ["depth", "teacher_targets"])
def test_stored_derived_jobs_remain_readable_but_new_requests_reject_old_fields(operation):
    legacy = {
        "dataset_id": uuid4(),
        "operation": operation,
        "output_name": "derived",
        "result_dataset_id": uuid4(),
        "rgb_key": RGB,
        "depth_key": DEPTH,
    }
    payload = DatasetPreprocessingPayload.model_validate(legacy)
    assert payload.rgb_key == (None if operation == "depth" else RGB)
    assert payload.depth_key == (None if operation == "depth" else DEPTH)
    assert "output_name" not in payload.model_dump()
    assert "result_dataset_id" not in payload.model_dump()
    with pytest.raises(ValueError):
        DatasetPreprocessingRequest.model_validate({key: value for key, value in legacy.items() if key != "dataset_id"})


@pytest.mark.parametrize("batch_size", [0, -1, 65, 1.5, True])
def test_invalid_preprocessing_batch_size_is_rejected(batch_size):
    with pytest.raises(ValueError):
        DatasetPreprocessingRequest(operation="depth", batch_size=batch_size)


def test_teacher_targets_cannot_silently_ignore_a_batch_size():
    with pytest.raises(ValueError, match="batch size 1"):
        DatasetPreprocessingRequest(operation="teacher_targets", rgb_key=RGB, batch_size=2)


@pytest.mark.parametrize("key", ["../depth", "observation.images/../../depth", "observation.images\\depth"])
def test_feature_names_cannot_escape_dataset(key):
    with pytest.raises(ValueError):
        DatasetPreprocessingRequest(operation="teacher_targets", rgb_key=RGB, depth_key=key)


@pytest.mark.parametrize("phase", ["computing", "backed_up", "published"])
def test_restart_recovers_dataset_swap(data, phase):
    settings, dataset = data
    job = DatasetPreprocessingJob(
        project_id=dataset.project_id,
        payload=DatasetPreprocessingPayload(dataset_id=dataset.id, operation="depth"),
    )
    source = Path(dataset.path)
    before = (source / "meta/info.json").read_bytes()
    root = settings.datasets_dir / f".preprocessing-{job.id}"
    root.mkdir()
    output = root / "dataset"
    shutil.copytree(source, output)
    (output / "processed").touch()
    if phase != "computing":
        source.rename(root / "original")
    if phase == "published":
        output.rename(source)

    assert recover_preprocessing_job(job) == (phase == "published")
    assert (source / "meta/info.json").read_bytes() == before
    assert (source / "processed").exists() == (phase == "published")
    assert not root.exists()


@pytest.mark.anyio
@pytest.mark.parametrize("outcome", ["success", "cancel", "process_failure", "publication_failure"])
async def test_worker_updates_same_dataset_and_preserves_original_on_failure(data, monkeypatch, outcome):
    settings, dataset = data
    payload = DatasetPreprocessingPayload(
        dataset_id=dataset.id,
        operation="depth",
        batch_size=3,
    )
    job = DatasetPreprocessingJob(project_id=dataset.project_id, payload=payload)
    final = Path(dataset.path)
    before = (final / "meta/info.json").read_bytes()
    updates = []

    async def update(job_id, status, **kwargs):
        updates.append(status)
        job.status = status
        return job

    jobs = SimpleNamespace(
        repo=SimpleNamespace(claim_pending_job=AsyncMock(return_value=job)),
        get_job_by_id=AsyncMock(return_value=job),
        update_job_status=AsyncMock(side_effect=update),
    )
    datasets = SimpleNamespace(get_dataset_by_id=AsyncMock(return_value=dataset), create_dataset=AsyncMock())
    original_rename = Path.rename

    def rename(path, target):
        if outcome == "publication_failure" and path.name == "dataset":
            raise OSError("publication failed")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", rename)

    @asynccontextmanager
    async def session():
        yield None

    monkeypatch.setattr("services.dataset_preprocessing.get_async_db_session_ctx", session)
    monkeypatch.setattr("services.dataset_preprocessing.JobService", lambda session: jobs)
    monkeypatch.setattr("services.dataset_preprocessing.DatasetService", lambda session: datasets)

    class Process:
        returncode = None

        def __init__(self):
            self.stdout = self
            self.lines = [b'{"progress": 50}\n']

        async def readline(self):
            if self.lines:
                return self.lines.pop()
            self.returncode = 1 if outcome == "process_failure" else 0
            return b""

        def terminate(self):
            self.returncode = -15

        async def wait(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = Process()

    async def spawn(*command, **kwargs):
        output = Path(command[4])
        assert "--rgb-key" not in command
        assert command[command.index("--batch-size") + 1] == "3"
        shutil.copytree(final, output)
        (output / "processed").write_text("ready")
        return process

    monkeypatch.setattr("services.dataset_preprocessing.asyncio.create_subprocess_exec", spawn)
    await run_preprocessing_job(job, queue.Queue(), lambda: outcome == "cancel")
    if outcome == "success":
        assert updates[-1] == JobStatus.COMPLETED
        assert (final / "processed").read_text() == "ready"
    else:
        assert updates[-1] == (JobStatus.CANCELED if outcome == "cancel" else JobStatus.FAILED)
        assert not (final / "processed").exists()
    datasets.create_dataset.assert_not_awaited()
    assert (final / "meta/info.json").read_bytes() == before
    assert not (settings.datasets_dir / f".preprocessing-{job.id}").exists()
