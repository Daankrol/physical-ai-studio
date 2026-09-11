import { $api } from '../../../../api/client';
import { WebsocketCamera } from '../../../cameras/websocket-camera';
import { useProjectId } from '../../../projects/use-project';
import { usePoseOverlay } from '../../use-joint-state';

export const CameraCell = ({ camera_id, pose_follower_id }: { camera_id: string; pose_follower_id?: string }) => {
    const { project_id } = useProjectId();
    const cameraQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras/{camera_id}', {
        params: { path: { project_id, camera_id } },
    });
    // Reuses the runtime session's already-open WebSocket (see
    // runtime-session-provider / use-joint-state) purely to listen for the
    // pose overlay; this component never sends the handshake.
    const poseOverlay = usePoseOverlay(project_id, pose_follower_id);
    const overlayLandmarks =
        pose_follower_id !== undefined && poseOverlay?.cameraId === camera_id ? poseOverlay.landmarks : undefined;

    return <WebsocketCamera camera={cameraQuery.data} poseOverlayLandmarks={overlayLandmarks} />;
};
