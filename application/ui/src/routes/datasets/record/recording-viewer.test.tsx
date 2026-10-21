import { createRef } from 'react';

import { rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { useRuntimeSession } from '../../../features/robots/runtime-session-provider';
import { getMockedDataset } from '../../../test-utils/mocks/mock-dataset';
import { getMockedEnvironment } from '../../../test-utils/mocks/mock-environment';
import { render } from '../../../test-utils/render';
import { RecordingViewer } from './recording-viewer';

rs.mock('../../../features/robots/runtime-session-provider', () => ({
    useRuntimeSession: rs.fn(),
}));

rs.mock('../../../features/robots/robot-control/robot-control-view', () => ({
    RobotControlView: () => <div>robot-control</div>,
}));

const mockSession = (hasLeader = true) => {
    const setFollowerSource = {
        mutateAsync: rs.fn(async () => ({ data: { follower_source: 'teleop', handover_state: 'idle' } })),
        isPending: false,
    };
    const startEpisode = { mutate: rs.fn(), isPending: false };
    rs.mocked(useRuntimeSession).mockReturnValue({
        dataset: getMockedDataset(),
        environment: getMockedEnvironment(),
        observation: createRef(),
        actions: createRef(),
        readyForRecording: true,
        state: {
            connected: true,
            follower_source: 'hold',
            handover_state: 'idle',
            has_leader: hasLeader,
            is_recording: false,
        },
        setFollowerSource,
        startEpisode,
        saveEpisode: { isPending: false },
        discardEpisode: { isPending: false },
    } as unknown as ReturnType<typeof useRuntimeSession>);
    return { setFollowerSource, startEpisode };
};

const renderViewer = () =>
    render(<RecordingViewer />, {
        route: '/projects/project-1/datasets/dataset-1/record',
        path: '/projects/:project_id/datasets/:dataset_id/record',
    });

describe('RecordingViewer', () => {
    it('resumes passive teleoperation before starting another episode from hold', async () => {
        const { setFollowerSource, startEpisode } = mockSession();
        let confirmTeleop:
            ((result: { data: { follower_source: string; handover_state: string } }) => void) | undefined;
        setFollowerSource.mutateAsync.mockImplementation(
            () =>
                new Promise((resolve) => {
                    confirmTeleop = resolve;
                })
        );
        const user = userEvent.setup();
        renderViewer();

        await user.click(await screen.findByRole('button', { name: /Start episode/i }));

        expect(startEpisode.mutate).not.toHaveBeenCalled();
        confirmTeleop?.({ data: { follower_source: 'teleop', handover_state: 'idle' } });
        await waitFor(() => expect(startEpisode.mutate).toHaveBeenCalledWith('manipulation'));
        expect(setFollowerSource.mutateAsync).toHaveBeenCalledWith({
            follower_source: 'teleop',
            auto_align_leader: false,
        });
        expect(setFollowerSource.mutateAsync.mock.invocationCallOrder[0]).toBeLessThan(
            startEpisode.mutate.mock.invocationCallOrder[0]
        );
    });

    it('does not start recording when teleoperation fails', async () => {
        const { setFollowerSource, startEpisode } = mockSession();
        setFollowerSource.mutateAsync.mockRejectedValue(new Error('Leader disconnected'));
        const user = userEvent.setup();
        renderViewer();

        await user.click(await screen.findByRole('button', { name: /Start episode/i }));

        expect(setFollowerSource.mutateAsync).toHaveBeenCalled();
        expect(startEpisode.mutate).not.toHaveBeenCalled();
    });

    it('disables recording without a leader', async () => {
        mockSession(false);
        renderViewer();

        expect(await screen.findByRole('button', { name: /Start episode/i })).toBeDisabled();
    });
});
