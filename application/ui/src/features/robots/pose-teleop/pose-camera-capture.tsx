import { useEffect, useRef, useState } from 'react';

import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';

import { PoseLandmark } from '../use-joint-state';

// Same CDN MediaPipe's own quick-start samples use for the wasm + model
// assets; there is no npm-published copy of either. PoC only — a follow-up
// should vendor these for offline/air-gapped installs.
const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm';
const MODEL_URL =
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/' +
    'pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

// Studio's runtime loop runs at 30 fps; sending landmarks faster than that
// just wastes bandwidth on frames the control loop cannot use anyway.
const SEND_INTERVAL_MS = 1000 / 30;

interface PoseCameraCaptureProps {
    onLandmarks: (landmarks: PoseLandmark[]) => void;
    onError: (message: string) => void;
}

/** Captures the browser webcam and streams MediaPipe pose landmarks to `onLandmarks`. */
export const PoseCameraCapture = ({ onLandmarks, onError }: PoseCameraCaptureProps) => {
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const landmarkerRef = useRef<PoseLandmarker | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const rafRef = useRef<number | null>(null);
    const lastSendRef = useRef(0);
    const [ready, setReady] = useState(false);
    const onErrorRef = useRef(onError);
    onErrorRef.current = onError;

    useEffect(() => {
        let cancelled = false;

        const setup = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 },
                    audio: false,
                });
                if (cancelled) {
                    stream.getTracks().forEach((track) => track.stop());
                    return;
                }
                streamRef.current = stream;
                const video = videoRef.current;
                if (video) {
                    video.srcObject = stream;
                    await video.play();
                }

                const fileset = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
                const landmarker = await PoseLandmarker.createFromOptions(fileset, {
                    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
                    runningMode: 'VIDEO',
                    numPoses: 1,
                });
                if (cancelled) {
                    landmarker.close();
                    return;
                }
                landmarkerRef.current = landmarker;
                setReady(true);
            } catch (cause) {
                onErrorRef.current(cause instanceof Error ? cause.message : 'Failed to start the pose camera.');
            }
        };

        void setup();

        return () => {
            cancelled = true;
            if (rafRef.current !== null) {
                cancelAnimationFrame(rafRef.current);
            }
            landmarkerRef.current?.close();
            landmarkerRef.current = null;
            streamRef.current?.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        };
    }, []);

    useEffect(() => {
        if (!ready) return;

        const tick = () => {
            const video = videoRef.current;
            const landmarker = landmarkerRef.current;
            const now = performance.now();
            if (video && landmarker && now - lastSendRef.current >= SEND_INTERVAL_MS) {
                lastSendRef.current = now;
                const result = landmarker.detectForVideo(video, now);
                const world = result.worldLandmarks[0];
                if (world) {
                    onLandmarks(
                        world.map((landmark) => ({
                            x: landmark.x,
                            y: landmark.y,
                            z: landmark.z,
                            visibility: landmark.visibility ?? 1,
                        }))
                    );
                }
            }
            rafRef.current = requestAnimationFrame(tick);
        };
        rafRef.current = requestAnimationFrame(tick);

        return () => {
            if (rafRef.current !== null) {
                cancelAnimationFrame(rafRef.current);
            }
        };
    }, [ready, onLandmarks]);

    // Hidden: this only exists to feed the landmarker. See PoseTeleopPreview for a visible feed.
    return <video ref={videoRef} muted playsInline style={{ display: 'none' }} />;
};
