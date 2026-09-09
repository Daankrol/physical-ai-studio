import { useCallback, useState } from 'react';

import { PoseLandmark } from '../use-joint-state';

interface UsePoseTeleopArgs {
    onEnabledChange: (enabled: boolean) => void;
    sendPoseLandmarks: (landmarks: PoseLandmark[]) => void;
}

/** Wires the pose camera capture on/off state to the runtime session's `pose` follower source. */
export const usePoseTeleop = ({ onEnabledChange, sendPoseLandmarks }: UsePoseTeleopArgs) => {
    const [enabled, setEnabled] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const toggle = useCallback(
        (next: boolean) => {
            setError(null);
            setEnabled(next);
            onEnabledChange(next);
        },
        [onEnabledChange]
    );

    const handleLandmarks = useCallback(
        (landmarks: PoseLandmark[]) => {
            sendPoseLandmarks(landmarks);
        },
        [sendPoseLandmarks]
    );

    const handleError = useCallback((message: string) => {
        setError(message);
        setEnabled(false);
    }, []);

    return { enabled, error, toggle, handleLandmarks, handleError };
};
