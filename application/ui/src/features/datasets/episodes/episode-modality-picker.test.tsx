// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { rs } from '@rstest/core';
import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { getMockedEnvironment } from '../../../test-utils/mocks/mock-environment';
import { getMockedEpisodeDetails } from '../../../test-utils/mocks/mock-episode';
import { render } from '../../../test-utils/render';
import { EpisodeModalityPicker } from './episode-modality-picker';
import { EpisodeVideoCell } from './episode-video-cell.component';
import { EpisodeViewerProvider, useEpisodeViewer } from './episode-viewer-provider.component';

const RGB = 'observation.images.front';

const SeekControl = () => {
    const { player } = useEpisodeViewer();
    return <button onClick={() => player.seek(0.1)}>Seek</button>;
};

it('switches the same camera between RGB and depth without losing the playhead', async () => {
    rs.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    const episode = getMockedEpisodeDetails({
        depth_videos: { [RGB]: { start: 0, end: 1, path: 'depth.mp4' } },
    });
    const user = userEvent.setup();
    const { container } = render(
        <EpisodeViewerProvider episode={episode} environment={getMockedEnvironment()}>
            <EpisodeModalityPicker />
            <EpisodeVideoCell videoId={RGB} datasetId='dataset-1' />
            <SeekControl />
        </EpisodeViewerProvider>
    );
    expect(container.querySelector('video')).toHaveAttribute('src', expect.stringContaining('rgb.mp4'));
    await user.click(screen.getByRole('button', { name: 'Seek' }));
    await user.click(screen.getByRole('button', { name: /Camera view/ }));
    await user.click(screen.getByRole('option', { name: 'Depth' }));
    const depth = container.querySelector('video');
    if (depth === null) throw new Error('Depth video is missing');
    expect(depth).toHaveAttribute('src', expect.stringContaining('depth.mp4'));
    expect(depth.style.filter).toMatch(/^url\(/);
    expect(container.querySelector('filter feComponentTransfer')).not.toBeNull();
    fireEvent.loadedMetadata(depth);
    expect(depth.currentTime).toBeCloseTo(0.1);
    await user.click(screen.getByRole('button', { name: /Camera view/ }));
    await user.click(screen.getByRole('option', { name: 'RGB' }));
    const rgb = container.querySelector('video');
    if (rgb === null) throw new Error('RGB video is missing');
    expect(rgb).toHaveAttribute('src', expect.stringContaining('rgb.mp4'));
    expect(rgb.style.filter).toBe('');
    expect(container.querySelector('filter')).toBeNull();
    fireEvent.loadedMetadata(rgb);
    expect(rgb.currentTime).toBeCloseTo(2.1);
});

it('does not offer Depth when the episode has no depth previews', () => {
    render(
        <EpisodeViewerProvider episode={getMockedEpisodeDetails()} environment={getMockedEnvironment()}>
            <EpisodeModalityPicker />
        </EpisodeViewerProvider>
    );
    expect(screen.queryByRole('button', { name: /Camera view/ })).not.toBeInTheDocument();
});
