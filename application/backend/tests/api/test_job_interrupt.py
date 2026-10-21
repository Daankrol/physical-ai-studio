# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from api.job import interrupt_job
from core.scheduler import Scheduler
from schemas.base_job import JobStatus
from schemas.job import RemoteTrainJobPayload, TrainJob


@pytest.mark.anyio
async def test_pending_reattachment_stop_signals_worker_to_cancel_remote_job() -> None:
    remote_job_id = uuid4()
    job = TrainJob(
        project_id=uuid4(),
        payload=RemoteTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            policy="act",
            model_name="remote",
            remote_trainer_id=uuid4(),
            remote_trainer_url="http://127.0.0.1:8001",
            remote_job_id=remote_job_id,
        ),
    )
    service = MagicMock()
    service.get_job_by_id = AsyncMock(return_value=job)
    service.cancel_pending_train_job = AsyncMock(return_value=None)
    scheduler = MagicMock(spec=Scheduler)

    await interrupt_job(job.id, service, scheduler)

    service.cancel_pending_train_job.assert_awaited_once_with(job.id)
    scheduler.cancel_job.assert_called_once_with(job.id)


@pytest.mark.anyio
@pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.RUNNING])
async def test_stop_signals_claimed_worker(status: JobStatus) -> None:
    job = TrainJob(
        project_id=uuid4(),
        status=status,
        payload=RemoteTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            policy="act",
            model_name="remote",
            remote_trainer_id=uuid4(),
        ),
    )
    service = MagicMock()
    service.get_job_by_id = AsyncMock(return_value=job)
    service.cancel_pending_train_job = AsyncMock(return_value=None)
    scheduler = MagicMock(spec=Scheduler)

    await interrupt_job(job.id, service, scheduler)

    service.cancel_pending_train_job.assert_awaited_once_with(job.id)
    scheduler.cancel_job.assert_called_once_with(job.id)
