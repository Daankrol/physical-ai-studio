import { SchemaDatasetPreprocessingJob } from '../../api/openapi-spec';
import { getMockedTrainJob } from './mock-train-job';

export const getMockedDatasetPreprocessingJob = (
    overrides: Partial<SchemaDatasetPreprocessingJob> = {}
): SchemaDatasetPreprocessingJob => ({
    ...getMockedTrainJob(),
    type: 'dataset_preprocessing',
    payload: {
        dataset_id: 'dataset-1',
        operation: 'depth',
        rgb_key: null,
        depth_key: null,
        depth_aligned: false,
        chunk_size: 50,
        batch_size: 1,
        device_index: 0,
    },
    ...overrides,
});
