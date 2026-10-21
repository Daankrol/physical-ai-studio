import { createRef } from 'react';

import { rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { SchemaModel } from '../../../api/openapi-spec';
import { getMockedEnvironment } from '../../../test-utils/mocks/mock-environment';
import { render } from '../../../test-utils/render';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { InferenceViewer } from './inference-viewer';

rs.mock('../../robots/runtime-session-provider', () => ({
    useRuntimeSession: rs.fn(),
}));

rs.mock('../../robots/robot-control/robot-control-view', () => ({
    RobotControlView: () => <div>robot-control</div>,
}));

const PROJECT_ID = 'project-1';

const model = {
    id: 'model-1',
    name: 'ACT',
    path: '/models/act',
    policy: 'act',
    properties: {},
    project_id: PROJECT_ID,
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

const environmentWithLeader = getMockedEnvironment({
    robots: [
        {
            robot: {
                id: 'follower-1',
                name: 'Follower',
                type: 'SO101_Follower' as const,
                payload: { connection_string: '', serial_number: 'SO101-001', calibration: {} },
            },
            tele_operator: { type: 'robot', robot_id: 'leader-1' },
        },
    ],
});

type MockResult = { data: { follower_source?: string; handover_state?: string; [key: string]: unknown } };

const mutation = (
    mutateAsync = rs.fn(async (): Promise<MockResult> => ({
        data: { follower_source: 'teleop', handover_state: 'idle' },
    }))
) => ({
    mutate: rs.fn(),
    mutateAsync,
    isPending: false,
});

type SessionOptions = {
    followerSource?: 'hold' | 'teleop' | 'policy';
    hasLeader?: boolean;
    readyForInference?: boolean;
    readyForRecording?: boolean;
    handoverState?: ReturnType<typeof useRuntimeSession>['state']['handover_state'];
    isRecording?: boolean;
    episodesRecorded?: number;
};

const mockSession = ({
    followerSource = 'hold',
    hasLeader = true,
    readyForInference = true,
    readyForRecording = true,
    handoverState = 'idle',
    isRecording = false,
    episodesRecorded = 0,
}: SessionOptions = {}) => {
    const startTask = mutation();
    const stopTask = mutation();
    const setFollowerSource = mutation();
    const startEpisode = mutation();
    const saveEpisode = mutation();
    const discardEpisode = mutation();

    rs.mocked(useRuntimeSession).mockReturnValue({
        model,
        readyForInference,
        readyForRecording,
        state: {
            connected: true,
            follower_source: followerSource,
            has_leader: hasLeader,
            model_loaded: true,
            task: null,
            dataset_loaded: readyForRecording,
            is_recording: isRecording,
            episodes_recorded: episodesRecorded,
            handover_state: handoverState,
        },
        startTask,
        stopTask,
        setFollowerSource,
        startEpisode,
        saveEpisode,
        discardEpisode,
        environment: environmentWithLeader,
        observation: createRef<Record<string, number> | undefined>(),
        inferenceDevice: { backend: 'pytorch', device: 'cpu' },
    } as unknown as ReturnType<typeof useRuntimeSession>);

    return { startTask, stopTask, setFollowerSource, startEpisode, saveEpisode, discardEpisode };
};

const renderViewer = (tasksOrProps: string[] | { collect?: boolean } = ['pick the cube']) =>
    render(
        <InferenceViewer
            tasks={Array.isArray(tasksOrProps) ? tasksOrProps : ['pick the cube']}
            {...(Array.isArray(tasksOrProps) ? {} : tasksOrProps)}
        />,
        {
            route: `/projects/${PROJECT_ID}/models/model-1/inference/pytorch`,
            path: '/projects/:project_id/models/:model_id/inference/:backend',
        }
    );

describe('InferenceViewer', () => {
    beforeEach(() => {
        rs.clearAllMocks();
    });

    it('enables the teleoperate toggle when the session has a leader robot', async () => {
        mockSession();

        renderViewer();

        expect(await screen.findByRole('switch', { name: 'Teleoperate' })).not.toBeChecked();
        expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
    });

    it('disables the teleoperate toggle when the session has no leader robot', async () => {
        mockSession({ hasLeader: false });

        renderViewer();

        expect(await screen.findByRole('button', { name: /play/i })).toBeInTheDocument();
        expect(screen.getByRole('switch', { name: 'Teleoperate' })).toBeDisabled();
    });

    it('selects the toggle while teleoperating', async () => {
        mockSession({ followerSource: 'teleop' });

        renderViewer();

        expect(await screen.findByRole('switch', { name: 'Teleoperate' })).toBeChecked();
        expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
    });

    it('leaves the toggle off during policy inference', async () => {
        mockSession({ followerSource: 'policy' });

        renderViewer();

        expect(await screen.findByRole('switch', { name: 'Teleoperate' })).not.toBeChecked();
        expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
    });

    it('opts into powered alignment from the model-run switch', async () => {
        const user = userEvent.setup();
        const { setFollowerSource } = mockSession();
        renderViewer();

        await user.click(await screen.findByRole('switch', { name: 'Teleoperate' }));

        expect(setFollowerSource.mutateAsync).toHaveBeenCalledWith({
            follower_source: 'teleop',
            auto_align_leader: true,
        });
    });

    it('requests hold when ordinary teleoperation is turned off', async () => {
        const user = userEvent.setup();
        const { setFollowerSource } = mockSession({ followerSource: 'teleop' });
        renderViewer();

        await user.click(await screen.findByRole('switch', { name: 'Teleoperate' }));

        expect(setFollowerSource.mutateAsync).toHaveBeenCalledWith('hold');
    });

    it('takes over from policy without issuing an unsafe resume or separate stop', async () => {
        const user = userEvent.setup();
        const { setFollowerSource, stopTask } = mockSession({ followerSource: 'policy' });

        renderViewer();

        await user.click(await screen.findByRole('switch', { name: 'Teleoperate' }));

        expect(setFollowerSource.mutateAsync).toHaveBeenCalledWith({
            follower_source: 'teleop',
            auto_align_leader: true,
        });
        expect(stopTask.mutate).not.toHaveBeenCalled();
    });

    it('waits for confirmed alignment before starting a correction episode', async () => {
        const user = userEvent.setup();
        const order: string[] = [];
        const { setFollowerSource, startEpisode } = mockSession();
        setFollowerSource.mutateAsync.mockImplementation(async () => {
            order.push('teleop-confirmed');
            return { data: { follower_source: 'teleop', handover_state: 'idle' } };
        });
        startEpisode.mutateAsync.mockImplementation(async () => {
            order.push('recording-started');
            return { data: {} };
        });

        renderViewer({ collect: true });
        await user.click(await screen.findByRole('button', { name: /take over/i }));

        await waitFor(() => expect(order).toEqual(['teleop-confirmed', 'recording-started']));
        expect(setFollowerSource.mutateAsync).toHaveBeenCalledWith({
            follower_source: 'teleop',
            auto_align_leader: true,
        });
        expect(startEpisode.mutateAsync).toHaveBeenCalledWith('pick the cube');
    });

    it('waits for the teleop confirmation before starting the correction', async () => {
        const user = userEvent.setup();
        let finishAlignment:
            ((value: { data: { follower_source: string; handover_state: string } }) => void) | undefined;
        const { setFollowerSource, startEpisode } = mockSession();
        setFollowerSource.mutateAsync.mockImplementation(
            () =>
                new Promise((resolve) => {
                    finishAlignment = resolve;
                })
        );

        renderViewer({ collect: true });
        await user.click(await screen.findByRole('button', { name: /take over/i }));
        expect(startEpisode.mutateAsync).not.toHaveBeenCalled();

        finishAlignment?.({ data: { follower_source: 'teleop', handover_state: 'idle' } });
        await waitFor(() => expect(startEpisode.mutateAsync).toHaveBeenCalledWith('pick the cube'));
    });

    it('shows the aligning state and blocks take over and policy resume', async () => {
        mockSession({ handoverState: 'aligning' });
        renderViewer({ collect: true });

        expect(screen.getByRole('status')).toHaveTextContent('Aligning leader');
        expect(screen.getByRole('button', { name: /take over/i })).toBeDisabled();
        expect(screen.getByRole('button', { name: /play/i })).toBeDisabled();
    });

    it('does not start a correction while alignment waits for the operator', async () => {
        const user = userEvent.setup();
        const { setFollowerSource, startEpisode } = mockSession();
        setFollowerSource.mutateAsync.mockResolvedValue({
            data: { follower_source: 'hold', handover_state: 'awaiting_release' },
        });
        const { unmount } = renderViewer({ collect: true });
        await user.click(await screen.findByRole('button', { name: /take over/i }));
        expect(startEpisode.mutateAsync).not.toHaveBeenCalled();

        unmount();
        const released = mockSession({ handoverState: 'awaiting_release' });
        renderViewer({ collect: true });
        expect(screen.getByRole('status')).toHaveTextContent('Take hold of the leader');
        expect(screen.getByRole('button', { name: /play/i })).toBeDisabled();
        await user.click(screen.getByRole('button', { name: 'Release alignment' }));

        expect(released.setFollowerSource.mutateAsync).toHaveBeenCalledWith({
            follower_source: 'teleop',
            auto_align_leader: false,
            release_alignment: true,
        });
        await waitFor(() => expect(released.startEpisode.mutateAsync).toHaveBeenCalledWith('pick the cube'));
    });

    it('releases a missed alignment and starts the correction without manual confirmation', async () => {
        const user = userEvent.setup();
        const { setFollowerSource, startEpisode } = mockSession({ handoverState: 'alignment_failed' });
        renderViewer({ collect: true });

        expect(screen.getByRole('status')).toHaveTextContent('did not reach alignment tolerance');
        expect(screen.getByRole('status')).toHaveTextContent('release alignment to take over');
        expect(screen.queryByRole('button', { name: 'Confirm manual alignment' })).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: /take over/i })).toBeDisabled();
        expect(screen.getByRole('button', { name: /play/i })).toBeDisabled();
        await user.click(screen.getByRole('button', { name: 'Release alignment' }));

        expect(setFollowerSource.mutateAsync).toHaveBeenCalledWith({
            follower_source: 'teleop',
            auto_align_leader: false,
            release_alignment: true,
        });
        await waitFor(() => expect(startEpisode.mutateAsync).toHaveBeenCalledWith('pick the cube'));
    });

    it('offers explicit release outside correction collection and blocks the teleoperate toggle', async () => {
        const user = userEvent.setup();
        const { setFollowerSource, startEpisode } = mockSession({ handoverState: 'awaiting_release' });
        renderViewer();

        expect(screen.getByRole('switch', { name: 'Teleoperate' })).toBeDisabled();
        await user.click(screen.getByRole('button', { name: 'Release alignment' }));

        expect(setFollowerSource.mutateAsync).toHaveBeenCalledWith({
            follower_source: 'teleop',
            auto_align_leader: false,
            release_alignment: true,
        });
        expect(startEpisode.mutateAsync).not.toHaveBeenCalled();
    });

    it('shows an error if releasing alignment fails and does not start recording', async () => {
        const user = userEvent.setup();
        const { setFollowerSource, startEpisode } = mockSession({ handoverState: 'awaiting_release' });
        setFollowerSource.mutateAsync.mockRejectedValue(new Error('Torque-off failed'));
        renderViewer({ collect: true });

        await user.click(screen.getByRole('button', { name: 'Release alignment' }));

        expect(await screen.findByRole('alert')).toHaveTextContent('Torque-off failed');
        expect(startEpisode.mutateAsync).not.toHaveBeenCalled();
    });

    it('offers takeover instead of manual pose confirmation when powered alignment is unavailable', async () => {
        const user = userEvent.setup();
        const { setFollowerSource, startEpisode } = mockSession({ handoverState: 'manual_required' });
        renderViewer({ collect: true });

        expect(screen.queryByRole('button', { name: 'Confirm manual alignment' })).not.toBeInTheDocument();
        expect(screen.getByRole('status')).toHaveTextContent('Automatic alignment is unavailable');
        await user.click(screen.getByRole('button', { name: 'Release alignment' }));

        expect(setFollowerSource.mutateAsync).toHaveBeenCalledWith({
            follower_source: 'teleop',
            auto_align_leader: false,
            release_alignment: true,
        });
        await waitFor(() => expect(startEpisode.mutateAsync).toHaveBeenCalledWith('pick the cube'));
    });

    it('does not start recording after a failed alignment', async () => {
        const user = userEvent.setup();
        const { setFollowerSource, startEpisode } = mockSession();
        setFollowerSource.mutateAsync.mockRejectedValue(new Error('leader alignment failed'));
        renderViewer({ collect: true });

        await user.click(await screen.findByRole('button', { name: /take over/i }));

        expect(await screen.findByRole('alert')).toHaveTextContent('leader alignment failed');
        expect(startEpisode.mutateAsync).not.toHaveBeenCalled();
    });

    it('holds and waits for confirmation before saving or discarding a correction', async () => {
        const user = userEvent.setup();
        const order: string[] = [];
        const { setFollowerSource, saveEpisode, discardEpisode } = mockSession({
            followerSource: 'teleop',
            isRecording: true,
        });
        setFollowerSource.mutateAsync.mockImplementation(async () => {
            order.push('hold-confirmed');
            return { data: { follower_source: 'hold', handover_state: 'idle' } };
        });
        saveEpisode.mutateAsync.mockImplementation(async () => {
            order.push('saved');
            return { data: {} };
        });

        renderViewer({ collect: true });
        await user.click(await screen.findByRole('button', { name: /save correction/i }));
        await waitFor(() => expect(order).toEqual(['hold-confirmed', 'saved']));
        expect(discardEpisode.mutateAsync).not.toHaveBeenCalled();
    });

    it('keeps discard available in hold after a failed save', async () => {
        const user = userEvent.setup();
        const { setFollowerSource, saveEpisode } = mockSession({ followerSource: 'teleop', isRecording: true });
        setFollowerSource.mutateAsync.mockResolvedValue({
            data: { follower_source: 'hold', handover_state: 'idle' },
        });
        saveEpisode.mutateAsync.mockRejectedValue(new Error('save failed'));

        renderViewer({ collect: true });
        await user.click(await screen.findByRole('button', { name: /save correction/i }));

        expect(await screen.findByRole('alert')).toHaveTextContent('save failed');
        expect(screen.getByRole('button', { name: /discard/i })).toBeEnabled();
    });

    it('holds before discarding a correction', async () => {
        const user = userEvent.setup();
        const order: string[] = [];
        const { setFollowerSource, discardEpisode } = mockSession({ followerSource: 'teleop', isRecording: true });
        setFollowerSource.mutateAsync.mockImplementation(async () => {
            order.push('hold-confirmed');
            return { data: { follower_source: 'hold', handover_state: 'idle' } };
        });
        discardEpisode.mutateAsync.mockImplementation(async () => {
            order.push('discarded');
            return { data: {} };
        });

        renderViewer({ collect: true });
        await user.click(await screen.findByRole('button', { name: /discard/i }));

        await waitFor(() => expect(order).toEqual(['hold-confirmed', 'discarded']));
    });

    it('requires a leader for correction collection and disables its controls', async () => {
        mockSession({ hasLeader: false });
        renderViewer({ collect: true });

        expect(await screen.findByText('Collecting corrections needs a leader arm.')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /take over/i })).toBeDisabled();
        expect(screen.getByRole('button', { name: /play/i })).toBeDisabled();
    });

    it('disables policy resume while recording or alignment is in progress', async () => {
        mockSession({ isRecording: true });
        const { unmount } = renderViewer({ collect: true });
        expect(await screen.findByRole('button', { name: /play/i })).toBeDisabled();
        unmount();

        mockSession({ handoverState: 'aligning' });
        renderViewer({ collect: true });
        expect(await screen.findByRole('button', { name: /play/i })).toBeDisabled();
    });
});

describe('InferenceViewer task prompt', () => {
    const originalScreenWidth = window.screen.width;

    beforeEach(() => {
        rs.clearAllMocks();
        // React Spectrum renders the mobile combo box on narrow screens; jsdom reports width 0.
        Object.defineProperty(window.screen, 'width', { value: 1920, configurable: true });
    });

    afterEach(() => {
        Object.defineProperty(window.screen, 'width', { value: originalScreenWidth, configurable: true });
    });

    it('keeps the task after the prompt field loses focus', async () => {
        const { startTask } = mockSession();
        renderViewer();

        const field = await screen.findByRole('combobox', { name: 'Task prompt' });
        expect(field).toHaveValue('pick the cube');

        await userEvent.click(field);
        await userEvent.click(screen.getByRole('button', { name: /play/i }));

        expect(field).toHaveValue('pick the cube');
        expect(startTask.mutate).toHaveBeenCalledWith('pick the cube');
    });

    it('sends the task picked from the list and keeps showing it', async () => {
        const { startTask } = mockSession();
        renderViewer(['pick the cube', 'stack the blocks']);

        const field = await screen.findByRole('combobox', { name: 'Task prompt' });
        await userEvent.click(screen.getByRole('button', { name: /show suggestions/i }));
        await userEvent.click(await screen.findByRole('option', { name: 'stack the blocks' }));
        await userEvent.click(screen.getByRole('button', { name: /play/i }));

        expect(field).toHaveValue('stack the blocks');
        expect(startTask.mutate).toHaveBeenCalledWith('stack the blocks');
    });

    it('sends a custom prompt typed into the field', async () => {
        const { startTask } = mockSession();
        renderViewer();

        const field = await screen.findByRole('combobox', { name: 'Task prompt' });
        await userEvent.clear(field);
        await userEvent.type(field, 'put the red cube in the box');
        await userEvent.click(screen.getByRole('button', { name: /play/i }));

        expect(field).toHaveValue('put the red cube in the box');
        expect(startTask.mutate).toHaveBeenCalledWith('put the red cube in the box');
    });

    it('asks for confirmation when the prompt is cleared', async () => {
        const { startTask } = mockSession();
        renderViewer();

        const field = await screen.findByRole('combobox', { name: 'Task prompt' });
        await userEvent.clear(field);
        await userEvent.keyboard('{Escape}');
        await userEvent.tab();
        await userEvent.click(screen.getByRole('button', { name: /play/i }));

        expect(await screen.findByRole('alertdialog')).toHaveTextContent('Start without a task prompt?');
        expect(startTask.mutate).not.toHaveBeenCalled();
    });

    it('asks for confirmation before starting with an empty prompt', async () => {
        const { startTask } = mockSession();
        renderViewer([]);

        await userEvent.click(await screen.findByRole('button', { name: /play/i }));

        expect(await screen.findByRole('alertdialog')).toHaveTextContent('Start without a task prompt?');
        expect(startTask.mutate).not.toHaveBeenCalled();

        await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
        expect(startTask.mutate).not.toHaveBeenCalled();

        await userEvent.click(screen.getByRole('button', { name: /play/i }));
        await userEvent.click(await screen.findByRole('button', { name: 'Start anyway' }));
        expect(startTask.mutate).toHaveBeenCalledWith('');
    });
});
