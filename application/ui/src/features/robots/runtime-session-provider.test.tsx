import { rs } from '@rstest/core';
import { act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { SchemaModel } from '../../api/openapi-spec';
import useWebSocketWithResponse from '../../components/websockets/use-websocket-with-response';
import { getMockedDataset } from '../../test-utils/mocks/mock-dataset';
import { getMockedEnvironment } from '../../test-utils/mocks/mock-environment';
import { render } from '../../test-utils/render';
import { RuntimeSessionProvider, useRuntimeSession } from './runtime-session-provider';

rs.mock('../../components/websockets/use-websocket-with-response', () => ({ default: rs.fn() }));

const sendJsonMessageAndWait = rs.fn(async () => ({ event: 'state', data: { is_recording: true } }));
const environment = getMockedEnvironment({
    robots: [
        {
            robot: {
                id: 'follower-1',
                name: 'Follower',
                type: 'SO101_Follower',
                payload: { connection_string: '', serial_number: 'SO101-001', calibration: {} },
                cameras: [],
            },
            tele_operator: { type: 'none' },
        },
    ],
} as never);

const StartEpisode = () => {
    const { startEpisode } = useRuntimeSession();
    return <button onClick={() => void startEpisode.mutateAsync('pick the cube')}>Start episode</button>;
};

const model = {
    id: 'model-1',
    name: 'ACT',
    path: '/models/act',
    policy: 'act',
    properties: {},
    project_id: 'project-1',
    dataset_id: 'dataset-1',
    snapshot_id: null,
    train_job_id: 'job-1',
    parent_model_id: null,
    version: 1,
    created_at: '2026-07-14T12:00:00Z',
    available_backends: ['pytorch'],
    lora_enabled: false,
    lora_use_dora: false,
    snapflow_enabled: false,
} as SchemaModel;

const renderProvider = (collect: boolean, withModel = false, withDataset = withModel) =>
    render(
        <RuntimeSessionProvider
            environment={environment}
            collect={collect}
            model={withModel ? model : undefined}
            dataset={withDataset ? getMockedDataset() : undefined}
            inferenceDevice={withModel ? ({ backend: 'pytorch', device: 'cpu' } as never) : undefined}
            onError={rs.fn()}
        >
            <StartEpisode />
        </RuntimeSessionProvider>,
        { route: '/projects/project-1', path: '/projects/:project_id' }
    );

describe('RuntimeSessionProvider correction recording command', () => {
    beforeEach(() => {
        rs.clearAllMocks();
        rs.mocked(useWebSocketWithResponse).mockReturnValue({
            sendJsonMessage: rs.fn(),
            sendJsonMessageAndWait,
            readyState: 1,
        } as never);
    });

    it('marks correction recordings as interventions', async () => {
        const user = userEvent.setup();
        renderProvider(true);

        await user.click(screen.getByRole('button', { name: 'Start episode' }));

        expect(sendJsonMessageAndWait).toHaveBeenCalledWith(
            { event: 'start_recording', data: { task: 'pick the cube', intervention: true } },
            expect.any(Function)
        );
    });

    it('loads the model and its training dataset without auto-entering teleop', async () => {
        renderProvider(true, true);
        const onOpen = rs.mocked(useWebSocketWithResponse).mock.calls[0]?.[1]?.onOpen;

        await act(async () => {
            onOpen?.(new Event('open'));
        });

        expect(sendJsonMessageAndWait).toHaveBeenCalledWith(
            {
                event: 'load_model',
                data: { model_id: model.id, inference_device: { backend: 'pytorch', device: 'cpu' } },
            },
            expect.any(Function)
        );
        expect(sendJsonMessageAndWait).toHaveBeenCalledWith(
            { event: 'load_dataset', data: { dataset_id: 'dataset-1' } },
            expect.any(Function)
        );
        expect(sendJsonMessageAndWait).not.toHaveBeenCalledWith(
            expect.objectContaining({ event: 'set_follower_source' }),
            expect.any(Function)
        );
    });

    it('requests guarded teleop for a dataset-only recording session', async () => {
        renderProvider(false, false, true);
        const onOpen = rs.mocked(useWebSocketWithResponse).mock.calls[0]?.[1]?.onOpen;

        await act(async () => {
            onOpen?.(new Event('open'));
        });

        expect(sendJsonMessageAndWait).toHaveBeenCalledWith(
            { event: 'set_follower_source', data: { follower_source: 'teleop' } },
            expect.any(Function)
        );
    });

    it('keeps dataset-only recordings as ordinary episodes', async () => {
        const user = userEvent.setup();
        renderProvider(false);

        await user.click(screen.getByRole('button', { name: 'Start episode' }));

        expect(sendJsonMessageAndWait).toHaveBeenCalledWith(
            { event: 'start_recording', data: { task: 'pick the cube' } },
            expect.any(Function)
        );
    });
});
