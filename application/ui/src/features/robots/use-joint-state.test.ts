import { beforeEach, describe, expect, it, rs } from '@rstest/core';
import { act } from '@testing-library/react';
import useWebSocket from 'react-use-websocket';

import { renderHook } from '../../test-utils/render';
import { isRecoverableRobotControlError, runtimeSocketUrl, useJointState } from './use-joint-state';

rs.mock('react-use-websocket', () => ({ default: rs.fn() }));

describe('isRecoverableRobotControlError', () => {
    it('classifies a leader connection loss as recoverable', () => {
        expect(isRecoverableRobotControlError('leader_connection_lost')).toBe(true);
    });

    it('keeps session connection failures fatal', () => {
        expect(isRecoverableRobotControlError('robot_connection_failed')).toBe(false);
        expect(isRecoverableRobotControlError(undefined)).toBe(false);
    });
});

describe('useJointState environment teleoperation', () => {
    const sendJsonMessage = rs.fn();

    beforeEach(() => {
        rs.clearAllMocks();
        rs.mocked(useWebSocket).mockReturnValue({ sendJsonMessage } as never);
    });

    it('requests passive teleop without powering or aligning the leader', () => {
        const { result } = renderHook(() => useJointState('project-1', 'follower-1', 'leader-1'));

        act(() => result.current.setFollowerSource('teleop'));

        expect(sendJsonMessage).toHaveBeenCalledWith({
            event: 'set_follower_source',
            data: { follower_source: 'teleop', auto_align_leader: false },
        });
    });

    it('requests hold when teleoperation is switched off', () => {
        const { result } = renderHook(() => useJointState('project-1', 'follower-1', 'leader-1'));

        act(() => result.current.setFollowerSource('hold'));

        expect(sendJsonMessage).toHaveBeenCalledWith({
            event: 'set_follower_source',
            data: { follower_source: 'hold' },
        });
    });
});

describe('runtimeSocketUrl', () => {
    it('uses a distinct URL per follower so share:true does not collide', () => {
        const left = runtimeSocketUrl('project-1', 'follower-left');
        const right = runtimeSocketUrl('project-1', 'follower-right');
        expect(left).toContain('follower-left');
        expect(right).toContain('follower-right');
        expect(left).not.toBe(right);
    });
});
