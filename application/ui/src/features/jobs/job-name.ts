// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaDatasetPreprocessingJob, SchemaTrainJob } from '../../api/openapi-spec';

export type VisibleJob = SchemaTrainJob | SchemaDatasetPreprocessingJob;

export const getJobName = (job: VisibleJob): string =>
    job.type === 'training'
        ? job.payload.model_name
        : job.payload.operation === 'depth'
          ? 'Compute depth maps'
          : 'Cache teacher targets';
