import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    Checkbox,
    Content,
    Dialog,
    Divider,
    Flex,
    Heading,
    Item,
    Loading,
    NumberField,
    Picker,
    ProgressBar,
    Text,
} from '@geti-ui/ui';
import { Link } from 'react-router';

import { $api } from '../../api/client';
import { getApiErrorMessage } from '../../api/errors';
import { SchemaDatasetOutput, SchemaDatasetPreprocessingJob } from '../../api/openapi-spec';
import { paths } from '../../paths';

export const PreprocessDatasetDialog = ({ dataset, close }: { dataset: SchemaDatasetOutput; close: () => void }) => {
    const datasetId = dataset.id ?? '';
    const [operation, setOperation] = useState<'depth' | 'teacher_targets'>('depth');
    const [rgbKey, setRgbKey] = useState<string | null>(null);
    const [depthSource, setDepthSource] = useState('estimate');
    const [aligned, setAligned] = useState(false);
    const [chunkSize, setChunkSize] = useState(50);
    const [batchSize, setBatchSize] = useState(1);
    const [deviceIndex, setDeviceIndex] = useState(0);
    const info = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}/preprocessing',
        {
            params: { path: { dataset_id: datasetId } },
        },
        { enabled: datasetId.length > 0 }
    );
    const jobs = $api.useQuery('get', '/api/jobs', {}, { refetchInterval: 1000 });
    const submit = $api.useMutation('post', '/api/dataset/{dataset_id}/preprocessing', {
        meta: { invalidates: [['get', '/api/jobs']] },
    });
    const cancel = $api.useMutation('post', '/api/jobs/{job_id}:interrupt', {
        meta: { invalidates: [['get', '/api/jobs']] },
    });
    const features = info.data?.features ?? [];
    const rgb = features.filter((feature) => !feature.is_depth);
    const depths = features.filter((feature) => feature.is_depth);
    const selectedRgb = rgbKey ?? rgb[0]?.key ?? '';
    const measured = depths.find((feature) => feature.key === depthSource);
    const needsAlignment =
        operation === 'teacher_targets' && depthSource !== 'estimate' && measured?.aligned_to !== selectedRgb;
    const ownJobs = (jobs.data ?? [])
        .filter(
            (job): job is SchemaDatasetPreprocessingJob =>
                job.type === 'dataset_preprocessing' && job.payload.dataset_id === datasetId
        )
        .slice()
        .reverse();
    const invalidBatch = !Number.isInteger(batchSize) || batchSize < 1 || batchSize > 64;
    const invalid =
        (info.data?.frames ?? 0) < 1 ||
        !selectedRgb ||
        (needsAlignment && !aligned) ||
        (operation === 'depth' && invalidBatch);
    const maxPixels = rgb.reduce((largest, feature) => Math.max(largest, feature.shape[0] * feature.shape[1]), 0);
    // shortcut: H100-calibrated MoGe budget at 256 tokens; profile per device if exact VRAM is needed.
    const vramGiB = 2 + batchSize * (0.125 + (maxPixels * 64) / 1024 ** 3);

    const queueJob = () =>
        submit.mutate({
            params: { path: { dataset_id: datasetId } },
            body: {
                operation,
                rgb_key: operation === 'depth' ? null : selectedRgb,
                depth_key: operation === 'depth' || depthSource === 'estimate' ? null : depthSource,
                depth_aligned: aligned,
                chunk_size: chunkSize,
                batch_size: operation === 'depth' ? batchSize : 1,
                device_index: deviceIndex,
            },
        });

    return (
        <Dialog width='size-6000'>
            <Heading>Preprocess dataset</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-150'>
                    <Text>
                        Saves preprocessing results in this dataset and preserves RGB recordings. Jobs share the local
                        training slot. You can close this popup and track progress in Jobs.
                    </Text>
                    {info.isPending ? (
                        <Loading mode='inline' />
                    ) : info.isError ? (
                        <div role='alert'>
                            <Text>Could not read dataset features.</Text>
                        </div>
                    ) : (
                        <>
                            <Picker
                                label='Operation'
                                selectedKey={operation}
                                onSelectionChange={(key) => setOperation(key as 'depth' | 'teacher_targets')}
                            >
                                <Item key='depth'>Compute depth maps</Item>
                                <Item key='teacher_targets'>Cache LingBot teacher targets</Item>
                            </Picker>
                            <Text>
                                {operation === 'depth'
                                    ? `Computes depth for all ${rgb.length} RGB cameras. Camera names are assigned ` +
                                      'automatically and existing aligned depth is preserved.'
                                    : 'Caches depth/video targets in this dataset so training needs no teachers. ' +
                                      'Uses about 2 MiB per frame.'}
                            </Text>
                            {operation === 'depth' && (
                                <>
                                    <NumberField
                                        label='Batch size'
                                        description='Frames per GPU inference. Larger batches need more VRAM.'
                                        value={batchSize}
                                        onChange={setBatchSize}
                                        minValue={1}
                                        maxValue={64}
                                    />
                                    {!invalidBatch && (
                                        <Text>
                                            Estimated VRAM budget: ~{vramGiB.toFixed(1)} GiB. This is a planning
                                            estimate, not a reservation; actual use varies by GPU. Cameras run
                                            sequentially.
                                        </Text>
                                    )}
                                </>
                            )}
                            {operation === 'teacher_targets' && (
                                <>
                                    <Picker
                                        label='RGB camera'
                                        selectedKey={selectedRgb}
                                        onSelectionChange={(key) => {
                                            setRgbKey(String(key));
                                            setAligned(false);
                                        }}
                                    >
                                        {rgb.map((feature) => (
                                            <Item key={feature.key}>{feature.key}</Item>
                                        ))}
                                    </Picker>
                                    <Picker
                                        label='Depth source'
                                        selectedKey={depthSource}
                                        onSelectionChange={(key) => {
                                            setDepthSource(String(key));
                                            setAligned(false);
                                        }}
                                        disabledKeys={depths
                                            .filter((feature) => feature.unit !== 'm' && feature.unit !== 'mm')
                                            .map((feature) => feature.key)}
                                        items={[
                                            { key: 'estimate', label: 'Estimate from RGB with MoGe' },
                                            ...depths.map((feature) => ({
                                                key: feature.key,
                                                label: `${feature.key} (${feature.unit ?? 'missing units'})`,
                                            })),
                                        ]}
                                    >
                                        {(item) => <Item key={item.key}>{item.label}</Item>}
                                    </Picker>
                                    {needsAlignment && (
                                        <Checkbox isSelected={aligned} onChange={setAligned}>
                                            Depth is calibrated and aligned to this RGB camera
                                        </Checkbox>
                                    )}
                                    <NumberField
                                        label='Action chunk size'
                                        value={chunkSize}
                                        onChange={setChunkSize}
                                        minValue={1}
                                        maxValue={1000}
                                    />
                                </>
                            )}
                            <NumberField
                                label='Local CUDA GPU index'
                                value={deviceIndex}
                                onChange={setDeviceIndex}
                                minValue={0}
                                maxValue={63}
                            />
                        </>
                    )}
                    {submit.isError && (
                        <div role='alert'>
                            <Text>
                                {getApiErrorMessage(submit.error) ?? 'Could not queue preprocessing. Try again.'}
                            </Text>
                        </div>
                    )}
                    {ownJobs.map((job) => (
                        <Flex direction='column' gap='size-75' key={job.id}>
                            <Text>
                                {job.payload.operation === 'depth' ? 'Depth maps' : 'Teacher targets'}: {job.status}
                            </Text>
                            <Text>{job.message}</Text>
                            {job.status === 'running' && (
                                <ProgressBar label='Preprocessing progress' value={job.progress} />
                            )}
                            {(job.status === 'running' || job.status === 'pending') && (
                                <Button
                                    variant='secondary'
                                    isPending={cancel.isPending}
                                    onPress={() => cancel.mutate({ params: { path: { job_id: job.id! } } })}
                                >
                                    Cancel task
                                </Button>
                            )}
                            {job.status === 'completed' && (
                                <Link
                                    to={paths.project.datasets.show({
                                        project_id: dataset.project_id,
                                        dataset_id: datasetId,
                                    })}
                                    onClick={close}
                                >
                                    Open dataset
                                </Link>
                            )}
                        </Flex>
                    ))}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Close
                </Button>
                <Button
                    variant='accent'
                    onPress={queueJob}
                    isDisabled={invalid || info.isError || info.isPending}
                    isPending={submit.isPending}
                >
                    {operation === 'depth' ? 'Compute depth maps' : 'Cache teacher targets'}
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
