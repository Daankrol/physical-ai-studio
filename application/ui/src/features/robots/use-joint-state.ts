import { useCallback, useEffect, useRef, useState } from 'react';

import useWebSocket from 'react-use-websocket';

import { fetchClient } from '../../api/client';
import { useRobotCatalogDefinitionQuery } from './robot-catalog.hooks';
import { mapJointToURDFJoint, useLoadModelQuery } from './robot-models-context';
import { SchemaRobotType } from './robot-types';

type JointsState = Array<{
    name: string;
    value: number;
}>;

// Mirrors runtime.contract.PoseLandmarkData: one skeleton overlay point.
export interface PoseLandmark {
    x: number;
    y: number;
    z: number;
    visibility: number;
}

// Mirrors runtime.contract.PoseEvent. Addressed by camera_id (not a feature
// key) so the browser can match it to the right camera panel.
export interface PoseOverlay {
    cameraId: string;
    landmarks: PoseLandmark[];
}

const getNewJointState = (newJoints: Record<string, number>) => {
    return Object.keys(newJoints).map((joint_name) => {
        return {
            name: joint_name,
            value: Number(newJoints[joint_name]),
        };
    });
};

export const useSynchronizeModelJoints = (joints: JointsState, robotType: SchemaRobotType) => {
    const { data: definition } = useRobotCatalogDefinitionQuery(robotType);
    const jointMap = definition.joint_map;

    const { data: model } = useLoadModelQuery(robotType);

    useEffect(() => {
        if (!model) return;

        joints.forEach((joint) => {
            mapJointToURDFJoint(joint, model, jointMap);
        });
    }, [model, joints, jointMap]);
};

// Mirrors runtime.contract.FollowerSource. Inference uses 'policy'. 'pose' is
// the PoC human-pose-teleop mode (browser MediaPipe landmarks -> joint angles).
export type FollowerSource = 'hold' | 'teleop' | 'policy' | 'pose';

const RECOVERABLE_ERROR_CODES = new Set([
    'leader_connection_lost',
    'pose_connection_lost',
    'pose_init_failed',
    'pose_not_supported',
]);
const EMPTY_CAMERA_IDS: string[] = [];

export const isRecoverableRobotControlError = (errorCode: unknown): errorCode is string =>
    typeof errorCode === 'string' && RECOVERABLE_ERROR_CODES.has(errorCode);

interface RobotControlState {
    connected: boolean;
    follower_source: FollowerSource;
    pose_available?: boolean;
}

// Compose from a typed project path so this does not depend on regenerating OpenAPI.
// follower_id must be in the URL: react-use-websocket share:true keys by URL.
export const runtimeSocketUrl = (project_id: string, follower_id: string): string =>
    `${fetchClient.PATH('/api/projects/{project_id}', {
        params: { path: { project_id } },
    })}/runtime/ws?follower_id=${encodeURIComponent(follower_id)}`;

export const useJointState = (
    project_id: string,
    follower_id: string,
    leader_id?: string,
    camera_ids: string[] = EMPTY_CAMERA_IDS,
    pose_camera_id?: string
) => {
    const [joints, setJoints] = useState<JointsState>([]);
    const [state, setState] = useState<RobotControlState>({
        connected: false,
        follower_source: 'hold',
    });
    const [poseOverlay, setPoseOverlay] = useState<PoseOverlay | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [errorCode, setErrorCode] = useState<string | null>(null);
    const [warning, setWarning] = useState<string | null>(null);
    const [socketEnabled, setSocketEnabled] = useState(true);
    const hasFatalError = useRef(false);
    const restartRequested = useRef(false);
    const socketEnabledRef = useRef(true);

    const handleMessage = useCallback((event: WebSocketEventMap['message']) => {
        try {
            const payload = JSON.parse(event.data);

            if (payload['event'] === 'observation') {
                const newJoints = getNewJointState(payload['data']);
                setJoints(newJoints);
            } else if (payload['event'] === 'pose') {
                setPoseOverlay({ cameraId: payload['camera_id'], landmarks: payload['landmarks'] });
            } else if (payload['event'] === 'state') {
                setState(payload['data']);
                setError(null);
                setErrorCode(null);
                // Deliberately not clearing `warning` here: a recoverable error
                // (e.g. pose_init_failed) is reported once via an 'error' event,
                // often around the same time as the initial 'state' event on
                // connect, and an unrelated later state change (model loaded,
                // dataset loaded, ...) must not silently hide a condition that
                // is still true. Only a fresh 'error' event or a reconnect
                // (onOpen) replaces it.
                hasFatalError.current = false;
            } else if (payload['event'] === 'error') {
                if (isRecoverableRobotControlError(payload.error_code)) {
                    setWarning(
                        typeof payload.message === 'string'
                            ? payload.message
                            : 'The robot session entered a safe state.'
                    );
                    return;
                }
                hasFatalError.current = true;
                setError(typeof payload.message === 'string' ? payload.message : 'Failed to connect to the robot.');
                setErrorCode(typeof payload.error_code === 'string' ? payload.error_code : 'robot_connection_failed');
            }
        } catch (parseError) {
            console.error('Failed to parse WebSocket message:', parseError);
        }
    }, []);

    const socket = useWebSocket(
        runtimeSocketUrl(project_id, follower_id),
        {
            share: true,
            shouldReconnect: () => !hasFatalError.current,
            reconnectAttempts: 5,
            reconnectInterval: 3000,
            onOpen: () => {
                socketEnabledRef.current = true;
                if (hasFatalError.current) {
                    return;
                }
                setError(null);
                setErrorCode(null);
                setWarning(null);
                const shouldRestart = restartRequested.current;
                restartRequested.current = false;
                socket.sendJsonMessage({
                    follower_id,
                    leader_id,
                    pose_camera_id,
                    camera_ids,
                    ...(shouldRestart ? { restart: true } : {}),
                });
            },
            onMessage: handleMessage,
            onError: (wsError) => console.error('WebSocket error:', wsError),
            onClose: (event) => {
                if (!socketEnabledRef.current || restartRequested.current) {
                    return;
                }
                if (!hasFatalError.current && event.code !== 1000) {
                    hasFatalError.current = true;
                    setError(`Connection closed unexpectedly (code ${event.code})`);
                    setErrorCode('connection_closed');
                }
            },
        },
        socketEnabled
    );

    useEffect(() => {
        if (!socketEnabled) {
            setSocketEnabled(true);
        }
    }, [socketEnabled]);

    const setFollowerSourceRequest = (value: FollowerSource) => {
        socket.sendJsonMessage({
            event: 'set_follower_source',
            data: { follower_source: value },
        });
    };

    const disconnect = () => {
        socket.sendJsonMessage({ event: 'disconnect' });
    };

    const restart = useCallback(() => {
        hasFatalError.current = false;
        restartRequested.current = true;
        socketEnabledRef.current = false;
        setError(null);
        setErrorCode(null);
        setSocketEnabled(false);
    }, []);

    return {
        joints,
        socket,
        state,
        poseOverlay,
        error,
        errorCode,
        warning,
        setFollowerSource: setFollowerSourceRequest,
        disconnect,
        restart,
    };
};

/**
 * Passive listener for a runtime session's pose overlay, for a component
 * (e.g. a camera panel) that is not the one driving the handshake.
 *
 * Relies on react-use-websocket's `share: true` reusing the same underlying
 * socket the robot panel already opened and handshook for this follower —
 * this hook only listens, it never sends the handshake itself. Pass
 * `follower_id: undefined` when there is no pose teleoperator to watch; the
 * socket is not opened in that case.
 */
export const usePoseOverlay = (project_id: string, follower_id: string | undefined): PoseOverlay | null => {
    const [poseOverlay, setPoseOverlay] = useState<PoseOverlay | null>(null);

    useWebSocket(
        runtimeSocketUrl(project_id, follower_id ?? ''),
        {
            share: true,
            onMessage: (event: WebSocketEventMap['message']) => {
                try {
                    const payload = JSON.parse(event.data);
                    if (payload['event'] === 'pose') {
                        setPoseOverlay({ cameraId: payload['camera_id'], landmarks: payload['landmarks'] });
                    }
                } catch (parseError) {
                    console.error('Failed to parse WebSocket message:', parseError);
                }
            },
        },
        follower_id !== undefined
    );

    return poseOverlay;
};
