// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaEpisode, SchemaEpisodeInfo } from '../../api/openapi-spec';

export const getMockedEpisode = (overrides: Partial<SchemaEpisodeInfo> = {}): SchemaEpisodeInfo => ({
    episode_index: 0,
    tasks: [],
    length: 10,
    fps: 30,
    ...overrides,
});

export const getMockedEpisodeDetails = (overrides: Partial<SchemaEpisode> = {}): SchemaEpisode => ({
    ...getMockedEpisode(),
    actions: [],
    action_keys: [],
    videos: { 'observation.images.front': { start: 2, end: 3, path: 'rgb.mp4' } },
    depth_videos: {},
    ...overrides,
});

export const getMockedEpisodes = (
    overridesList: Partial<SchemaEpisodeInfo>[] = [{ episode_index: 0 }, { episode_index: 1 }]
): SchemaEpisodeInfo[] => overridesList.map((overrides) => getMockedEpisode(overrides));
