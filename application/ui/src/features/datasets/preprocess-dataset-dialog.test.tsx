import { rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { getMockedDataset } from '../../test-utils/mocks/mock-dataset';
import { getMockedDatasetPreprocessingInfo } from '../../test-utils/mocks/mock-dataset-preprocessing-info';
import { getMockedDatasetPreprocessingJob } from '../../test-utils/mocks/mock-dataset-preprocessing-job';
import { render } from '../../test-utils/render';
import { PreprocessDatasetDialog } from './preprocess-dataset-dialog';

const dataset = getMockedDataset({ id: 'dataset-1' });
const DEPTH = 'observation.images.sensor_depth';

const setup = () => {
    server.use(
        http.get('/api/dataset/{dataset_id}/preprocessing', () =>
            HttpResponse.json(getMockedDatasetPreprocessingInfo())
        ),
        http.get('/api/jobs', () => HttpResponse.json([]))
    );
};

it('queues depth for all cameras in the same dataset without asking for names', async () => {
    setup();
    const submitted = rs.fn();
    server.use(
        http.post('/api/dataset/{dataset_id}/preprocessing', async ({ request }) => {
            submitted(await request.json());
            return HttpResponse.json(getMockedDatasetPreprocessingJob(), { status: 202 });
        })
    );
    const user = userEvent.setup();
    render(<PreprocessDatasetDialog dataset={dataset} close={rs.fn()} />);
    await screen.findByText(/Computes depth for all 2 RGB cameras/);
    expect(screen.queryByRole('textbox', { name: 'New dataset name' })).not.toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'New depth feature' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /RGB camera/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Compute depth maps' }));
    await waitFor(() => expect(submitted).toHaveBeenCalled());
    expect(submitted.mock.calls[0][0]).toEqual({
        operation: 'depth',
        rgb_key: null,
        depth_key: null,
        depth_aligned: false,
        chunk_size: 50,
        batch_size: 1,
        device_index: 0,
    });
});

it.each([
    {
        error: {
            error_code: 'bad_request',
            message: { output_name: ['Field required'], rgb_key: ['Input should be a valid string'] },
            http_status: 400,
        },
        expected: 'output_name: Field required; rgb_key: Input should be a valid string',
    },
    {
        error: { message: 'The dataset is already being processed.' },
        expected: 'The dataset is already being processed.',
    },
    { error: {}, expected: 'Could not queue preprocessing. Try again.' },
])('shows the queue error: $expected', async ({ error, expected }) => {
    setup();
    server.use(
        http.post('/api/dataset/{dataset_id}/preprocessing', ({ response }) =>
            response.untyped(HttpResponse.json(error, { status: 400 }))
        )
    );
    const user = userEvent.setup();
    render(<PreprocessDatasetDialog dataset={dataset} close={rs.fn()} />);
    await screen.findByText(/Computes depth for all 2 RGB cameras/);
    await user.click(screen.getByRole('button', { name: 'Compute depth maps' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(expected);
});

it('uses the largest camera for its VRAM budget and submits the selected depth batch size', async () => {
    setup();
    const submitted = rs.fn();
    server.use(
        http.get('/api/dataset/{dataset_id}/preprocessing', () =>
            HttpResponse.json(
                getMockedDatasetPreprocessingInfo({
                    features: [
                        { key: 'observation.images.front', shape: [480, 640, 3], is_depth: false },
                        { key: 'observation.images.wrist', shape: [1080, 1920, 3], is_depth: false },
                    ],
                })
            )
        ),
        http.post('/api/dataset/{dataset_id}/preprocessing', async ({ request }) => {
            submitted(await request.json());
            return HttpResponse.json(getMockedDatasetPreprocessingJob(), { status: 202 });
        })
    );
    const user = userEvent.setup();
    render(<PreprocessDatasetDialog dataset={dataset} close={rs.fn()} />);
    expect(await screen.findByText(/Estimated VRAM budget: ~2.2 GiB/)).toBeInTheDocument();
    const batch = screen.getByRole('textbox', { name: 'Batch size' });
    await user.clear(batch);
    await user.type(batch, '8');
    await user.tab();
    expect(screen.getByText(/Estimated VRAM budget: ~4.0 GiB/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Compute depth maps' }));
    await waitFor(() => expect(submitted).toHaveBeenCalledWith(expect.objectContaining({ batch_size: 8 })));
});

it('requires explicit alignment when caching targets from recorded depth', async () => {
    setup();
    const user = userEvent.setup();
    render(<PreprocessDatasetDialog dataset={dataset} close={rs.fn()} />);
    await screen.findByText(/Computes depth for all 2 RGB cameras/);
    await user.click(screen.getByRole('button', { name: /Operation/ }));
    await user.click(screen.getByRole('option', { name: 'Cache LingBot teacher targets' }));
    expect(screen.queryByRole('textbox', { name: 'Batch size' })).not.toBeInTheDocument();
    expect(screen.queryByText(/Estimated VRAM budget/)).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /Depth source/ }));
    await user.click(screen.getByRole('option', { name: `${DEPTH} (mm)` }));
    expect(screen.getByRole('button', { name: 'Cache teacher targets' })).toBeDisabled();
    await user.click(screen.getByRole('checkbox', { name: /calibrated and aligned/ }));
    expect(screen.getByRole('button', { name: 'Cache teacher targets' })).toBeEnabled();
});

it('shows background task progress and allows cancellation', async () => {
    setup();
    const job = getMockedDatasetPreprocessingJob({ status: 'running', progress: 40 });
    server.use(http.get('/api/jobs', () => HttpResponse.json([job])));
    const canceled = rs.fn();
    server.use(
        http.post('/api/jobs/{job_id}:interrupt', () => {
            canceled();
            return new HttpResponse(null, { status: 200 });
        })
    );
    const user = userEvent.setup();
    render(<PreprocessDatasetDialog dataset={dataset} close={rs.fn()} />);
    const cancelButton = await screen.findByRole('button', { name: 'Cancel task' });
    expect(screen.getByRole('progressbar', { name: 'Preprocessing progress' })).toHaveAttribute('aria-valuenow', '40');
    await user.click(cancelButton);
    await waitFor(() => expect(canceled).toHaveBeenCalled());
});
