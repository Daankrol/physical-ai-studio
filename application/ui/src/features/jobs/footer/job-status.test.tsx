// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { rs } from '@rstest/core';
import { act, screen } from '@testing-library/react';

import { getMockedDatasetPreprocessingJob } from '../../../test-utils/mocks/mock-dataset-preprocessing-job';
import { render } from '../../../test-utils/render';
import { JobStatus } from './job-status';

let onMessage: (event: { data: string }) => void;
rs.mock('react-use-websocket', () => ({
    default: (_url: string, options: { onMessage: typeof onMessage }) => {
        onMessage = options.onMessage;
        return {};
    },
}));

it('shows preprocessing progress in the footer without opening its popup', () => {
    render(<JobStatus />);
    act(() =>
        onMessage({
            data: JSON.stringify({
                event: 'JOB_UPDATE',
                data: getMockedDatasetPreprocessingJob({ status: 'running', progress: 40, message: 'Computing depth' }),
            }),
        })
    );
    expect(screen.getByText('Computing depth (Compute depth maps)')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '40');
});
