import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.schema import Base, ProjectDB
from repositories.job_repo import JobRepository
from repositories.mappers.job_mapper import JobMapper
from schemas.base_job import JobStatus, JobType
from schemas.dataset_preprocessing import DatasetPreprocessingPayload
from schemas.job import DatasetPreprocessingJob, RemoteTrainJobPayload, TrainJob


def test_duplicate_remote_job_payload_with_uuids_is_json_serializable() -> None:
    """Duplicate detection matches the persisted JSON representation of UUID fields."""

    async def check_duplicate() -> None:
        engine = create_async_engine("sqlite+aiosqlite://")
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        project_id = uuid4()
        payload = RemoteTrainJobPayload(
            project_id=project_id,
            dataset_id=uuid4(),
            policy="act",
            model_name="test-model",
            base_model_id=uuid4(),
            snapshot_id=uuid4(),
            remote_trainer_id=uuid4(),
            remote_trainer_url="https://trainer.example.test",
        )

        async with engine.begin() as connection:
            await connection.run_sync(lambda sync_connection: Base.metadata.create_all(sync_connection))

        async with session_factory() as session:
            session.add(ProjectDB(id=str(project_id), name="Test project"))
            await session.commit()
            job = TrainJob(project_id=project_id, payload=payload, created_at=datetime.now(tz=UTC))
            session.add(JobMapper.to_schema(job))
            await session.commit()

            repository = JobRepository(session)
            assert await repository.is_job_duplicate(project_id, payload)

            remote_job_id = uuid4()
            updated_payload = payload.model_copy(update={"remote_job_id": remote_job_id})
            updated_job = await repository.update(job, {"payload": updated_payload.model_dump(mode="json")})
            assert updated_job.payload.remote_job_id == remote_job_id

        await engine.dispose()

    asyncio.run(check_duplicate())


def test_preprocessing_claim_is_exclusive_and_does_not_revive_canceled_jobs() -> None:
    async def check():
        engine = create_async_engine("sqlite+aiosqlite://")
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            project_id = uuid4()
            session.add(ProjectDB(id=str(project_id), name="project"))
            await session.commit()
            repository = JobRepository(session)
            payload = DatasetPreprocessingPayload(
                dataset_id=uuid4(),
                operation="depth",
            )
            job = DatasetPreprocessingJob(project_id=project_id, payload=payload, created_at=datetime.now(UTC))
            await repository.save(job)
            claimed = await repository.claim_pending_job(job.id, JobType.DATASET_PREPROCESSING)
            assert claimed.status == JobStatus.RUNNING
            assert await repository.claim_pending_job(job.id, JobType.DATASET_PREPROCESSING) is None
            canceled = DatasetPreprocessingJob(
                project_id=project_id, payload=payload, status=JobStatus.CANCELED, created_at=datetime.now(UTC)
            )
            await repository.save(canceled)
            assert await repository.claim_pending_job(canceled.id, JobType.DATASET_PREPROCESSING) is None
        await engine.dispose()

    asyncio.run(check())
