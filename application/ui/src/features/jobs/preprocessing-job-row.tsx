// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, ProgressBar, Text, View } from '@geti-ui/ui';

import { SchemaDatasetPreprocessingJob } from '../../api/openapi-spec';
import { Table } from '../../components/table/table';
import { useDatasetQuery, useEnvironmentQuery } from '../models/api/queries';
import { JobMenu } from '../models/job-table/job-table';
import { SingleBadge } from '../models/shared/split-badge';
import { getJobName } from './job-name';

export const PreprocessingJobRow = ({
    job,
    onInterrupt,
    onViewLogs,
}: {
    job: SchemaDatasetPreprocessingJob;
    onInterrupt: () => void;
    onViewLogs: () => void;
}) => {
    const { data: dataset } = useDatasetQuery(job.payload.dataset_id);
    const { data: environment } = useEnvironmentQuery(job.project_id, dataset?.environment_id);
    const isActive = job.status === 'running' || job.status === 'pending';

    return (
        <Table.Row
            id={job.id}
            after={
                job.status === 'running' && (
                    <ProgressBar
                        aria-label={`Preprocessing progress for ${getJobName(job)}`}
                        size='S'
                        width='100%'
                        value={job.progress}
                    />
                )
            }
        >
            <div />
            <Flex direction='column' gap='size-50'>
                <Text>{getJobName(job)}</Text>
                <SingleBadge
                    color={job.status === 'failed' ? 'var(--spectrum-negative-visual-color)' : 'var(--energy-blue)'}
                    text={job.status}
                />
                <Text>{job.message}</Text>
            </Flex>
            <Text>-</Text>
            <Text>{job.payload.operation === 'depth' ? 'MoGe' : 'LingBot teachers'}</Text>
            <Text>{dataset?.name ?? '-'}</Text>
            <Text>{environment?.name ?? '-'}</Text>
            <Text>{`Local · CUDA ${job.payload.device_index}`}</Text>
            <View>
                {isActive && (
                    <Button variant='secondary' onPress={onInterrupt}>
                        Cancel task
                    </Button>
                )}
            </View>
            <View>
                <JobMenu trainJob={job} onViewLogs={onViewLogs} />
            </View>
        </Table.Row>
    );
};
