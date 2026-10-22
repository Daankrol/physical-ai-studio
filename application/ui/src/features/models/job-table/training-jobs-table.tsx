import { ReactNode } from 'react';

import { $api } from '../../../api/client';
import { Table, TableColumn } from '../../../components/table/table';
import { VisibleJob } from '../../jobs/job-name';
import { PreprocessingJobRow } from '../../jobs/preprocessing-job-row';
import { TrainingRow } from './job-table';

const JOB_COLUMNS: TableColumn[] = [
    { width: 'max-content' },
    { width: '2fr', header: 'Name' },
    { width: '1fr', header: 'Loss' },
    { width: '1fr', header: 'Architecture' },
    { width: '1fr', header: 'Dataset' },
    { width: '1fr', header: 'Environment' },
    { width: '1fr', header: 'Trainer' },
    { width: '1fr' },
    { width: 'auto', align: 'end' },
];

const getTimestamp = (job: VisibleJob): number => {
    if (!job.created_at) return 0;
    const time = new Date(job.created_at).getTime();
    return Number.isNaN(time) ? 0 : time;
};

interface TrainingJobsTableProps<T extends VisibleJob> {
    jobs: T[];
    onViewLogs: (job: T) => void;
    emptyMessage?: ReactNode;
}

export const TrainingJobsTable = <T extends VisibleJob>({
    jobs,
    onViewLogs,
    emptyMessage,
}: TrainingJobsTableProps<T>) => {
    const sortedJobs = jobs.toSorted((a, b) => getTimestamp(b) - getTimestamp(a));

    const interruptMutation = $api.useMutation('post', '/api/jobs/{job_id}:interrupt', {
        meta: { invalidates: [['get', '/api/jobs']] },
    });
    const onInterrupt = (job: VisibleJob) => {
        if (job.id !== undefined) {
            interruptMutation.mutate({ params: { path: { job_id: job.id } } });
        }
    };

    if (sortedJobs.length === 0) {
        return <>{emptyMessage ?? null}</>;
    }

    return (
        <Table columns={JOB_COLUMNS}>
            {sortedJobs.map((job) =>
                job.type === 'training' ? (
                    <TrainingRow
                        key={job.id}
                        trainJob={job}
                        onInterrupt={() => onInterrupt(job)}
                        onViewLogs={() => onViewLogs(job)}
                    />
                ) : (
                    <PreprocessingJobRow
                        key={job.id}
                        job={job}
                        onInterrupt={() => onInterrupt(job)}
                        onViewLogs={() => onViewLogs(job)}
                    />
                )
            )}
        </Table>
    );
};
