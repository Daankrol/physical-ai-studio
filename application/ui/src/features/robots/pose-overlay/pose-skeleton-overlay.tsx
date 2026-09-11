import { useEffect, useRef } from 'react';

import { PoseLandmark } from '../use-joint-state';

// A minimal subset of MediaPipe's 33-point BlazePose topology: shoulders,
// elbows, wrists, hips. Good enough to see the arm skeleton driving pose
// teleop; not a general-purpose pose visualizer.
const CONNECTIONS: Array<[number, number]> = [
    [11, 12], // shoulder to shoulder
    [11, 13], // left shoulder to elbow
    [13, 15], // left elbow to wrist
    [12, 14], // right shoulder to elbow
    [14, 16], // right elbow to wrist
    [11, 23], // left shoulder to hip
    [12, 24], // right shoulder to hip
    [23, 24], // hip to hip
];

const MIN_VISIBILITY = 0.5;

interface PoseSkeletonOverlayProps {
    landmarks: PoseLandmark[];
    width: number;
    height: number;
}

/** Draws the pose-estimation skeleton on top of a camera feed, from normalized image-space landmarks. */
export const PoseSkeletonOverlay = ({ landmarks, width, height }: PoseSkeletonOverlayProps) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (!canvas || !ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#00e5ff';
        ctx.fillStyle = '#00e5ff';
        ctx.lineWidth = 3;

        const point = (index: number) => {
            const landmark = landmarks[index];
            if (landmark === undefined || landmark.visibility < MIN_VISIBILITY) return undefined;
            return { x: landmark.x * canvas.width, y: landmark.y * canvas.height };
        };

        for (const [a, b] of CONNECTIONS) {
            const start = point(a);
            const end = point(b);
            if (start === undefined || end === undefined) continue;
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.stroke();
        }

        landmarks.forEach((landmark, index) => {
            if (landmark.visibility < MIN_VISIBILITY) return;
            const isTracked = CONNECTIONS.some(([a, b]) => a === index || b === index);
            if (!isTracked) return;
            ctx.beginPath();
            ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);
            ctx.fill();
        });
    }, [landmarks]);

    return (
        <canvas
            ref={canvasRef}
            width={width}
            height={height}
            style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
            aria-hidden
        />
    );
};
