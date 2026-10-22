import { createContext, ReactNode, useContext, useState } from 'react';

import { SchemaEnvironmentWithRelations, SchemaEpisode } from '../../../api/openapi-spec';
import { Player, usePlayer } from './use-player';

type EpisodeViewerContextValue = null | {
    player: Player;
    episode: SchemaEpisode;
    environment: SchemaEnvironmentWithRelations;
    modality: 'rgb' | 'depth';
    setModality: (modality: 'rgb' | 'depth') => void;
};

const EpisodeViewerContext = createContext<EpisodeViewerContextValue>(null);

interface EpisodeViewerProviderProps {
    children: ReactNode;
    episode: SchemaEpisode;
    environment: SchemaEnvironmentWithRelations;
}

export const EpisodeViewerProvider = ({ children, episode, environment }: EpisodeViewerProviderProps) => {
    const player = usePlayer(episode);
    const [modality, setModality] = useState<'rgb' | 'depth'>('rgb');

    return (
        <EpisodeViewerContext.Provider
            value={{
                player,
                episode,
                environment,
                modality: Object.keys(episode.depth_videos ?? {}).length > 0 ? modality : 'rgb',
                setModality,
            }}
        >
            {children}
        </EpisodeViewerContext.Provider>
    );
};

export const useEpisodeViewer = () => {
    const ctx = useContext(EpisodeViewerContext);
    if (!ctx) throw new Error('useEpisodeViewer must be used within EpisodeViewerProvider');
    return ctx;
};
