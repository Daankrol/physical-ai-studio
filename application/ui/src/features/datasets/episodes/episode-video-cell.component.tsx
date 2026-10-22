import { useEffect, useId, useRef } from 'react';

import { fetchClient } from '../../../api/client';
import { SchemaEpisodeVideo } from '../../../api/openapi-spec';
import { useFittedMediaSize } from '../../cameras/use-fitted-media-size';
import { useEpisodeViewer } from './episode-viewer-provider.component';
import { Player } from './use-player';

type EpisodeVideoProps = {
    url: string | undefined;
    player: Player;
    episodeVideo: SchemaEpisodeVideo | undefined;
    isDepth: boolean;
};

const EpisodeVideo = ({ url, player, episodeVideo, isDepth }: EpisodeVideoProps) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const colorMapId = useId();

    useEffect(() => {
        const video = videoRef.current;
        const start = episodeVideo?.start;
        if (!video || start === undefined || !Number.isFinite(start)) return;

        video.currentTime = player.timeRef.current + start;
        if (player.isPlaying) {
            video.play();
        } else {
            video.pause();
        }
    }, [player, episodeVideo?.start, videoRef]);

    useEffect(() => {
        const video = videoRef.current;
        const start = episodeVideo?.start;

        if (video && player.isSeeking && start !== undefined && Number.isFinite(start)) {
            const interval = setInterval(() => {
                video.currentTime = player.timeRef.current + start;
            }, 1000 / 60);
            return () => clearInterval(interval);
        }
    }, [player, episodeVideo?.start]);

    const { containerRef, width, height } = useFittedMediaSize(
        videoRef.current?.videoWidth,
        videoRef.current?.videoHeight
    );

    /* eslint-disable jsx-a11y/media-has-caption */
    return (
        <div ref={containerRef} style={{ height: '100%', width: '100%' }}>
            {isDepth && (
                <svg aria-hidden='true' focusable='false' width={0} height={0} style={{ position: 'absolute' }}>
                    <defs>
                        <filter id={colorMapId} colorInterpolationFilters='sRGB'>
                            {/* Cached previews are inverse depth: black is invalid, brighter is nearer. */}
                            <feComponentTransfer>
                                <feFuncR type='table' tableValues='0 0 0 1 1' />
                                <feFuncG type='table' tableValues='0 0.6 1 1 0' />
                                <feFuncB type='table' tableValues='0 1 0.4 0 0' />
                            </feComponentTransfer>
                        </filter>
                    </defs>
                </svg>
            )}
            {url !== undefined && (
                <video
                    ref={videoRef}
                    src={url}
                    style={isDepth ? { filter: `url(#${colorMapId})` } : undefined}
                    preload='metadata'
                    width={width}
                    height={height}
                    onLoadedMetadata={() => {
                        if (videoRef.current) {
                            videoRef.current.currentTime = player.timeRef.current + (episodeVideo?.start ?? 0);
                            if (player.isPlaying) void videoRef.current.play();
                        }
                    }}
                />
            )}
        </div>
    );
};

export const EpisodeVideoCell = ({ videoId, datasetId }: { videoId: string; datasetId: string | undefined }) => {
    const { player, episode, modality } = useEpisodeViewer();

    const episodeVideo: SchemaEpisodeVideo | undefined =
        modality === 'depth' ? episode.depth_videos?.[videoId] : episode.videos[videoId];

    const url =
        episodeVideo === undefined || datasetId === undefined
            ? undefined
            : fetchClient.PATH('/api/dataset/{dataset_id}/video/{video_path}', {
                  params: {
                      path: {
                          dataset_id: datasetId,
                          video_path: episodeVideo.path,
                      },
                  },
              });

    if (modality === 'depth' && episodeVideo === undefined) {
        return <div>No depth available for this camera.</div>;
    }

    return (
        <EpisodeVideo
            key={`${episode.episode_index}-${modality}`}
            url={url}
            player={player}
            episodeVideo={episodeVideo}
            isDepth={modality === 'depth'}
        />
    );
};
