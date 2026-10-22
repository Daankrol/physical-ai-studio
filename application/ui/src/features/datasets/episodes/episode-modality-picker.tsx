// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti-ui/ui';

import { useEpisodeViewer } from './episode-viewer-provider.component';

export const EpisodeModalityPicker = () => {
    const { episode, modality, setModality } = useEpisodeViewer();

    if (Object.keys(episode.depth_videos ?? {}).length === 0) {
        return null;
    }

    return (
        <Picker
            label='Camera view'
            labelPosition='side'
            selectedKey={modality}
            onSelectionChange={(key) => setModality(key === 'depth' ? 'depth' : 'rgb')}
        >
            <Item key='rgb'>RGB</Item>
            <Item key='depth'>Depth</Item>
        </Picker>
    );
};
