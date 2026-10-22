// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaDatasetPreprocessingInfo } from '../../api/openapi-spec';

export const getMockedDatasetPreprocessingInfo = (
    overrides: Partial<SchemaDatasetPreprocessingInfo> = {}
): SchemaDatasetPreprocessingInfo => ({
    frames: 4,
    has_teacher_cache: false,
    features: [
        { key: 'observation.images.front', shape: [32, 32, 3], is_depth: false },
        { key: 'observation.images.wrist', shape: [32, 32, 3], is_depth: false },
        { key: 'observation.images.sensor_depth', shape: [32, 32, 1], is_depth: true, unit: 'mm' },
    ],
    ...overrides,
});
