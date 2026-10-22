// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { HttpResponse } from 'msw';

import { getMockedDataset } from '../src/test-utils/mocks/mock-dataset';
import { getMockedDatasetPreprocessingInfo } from '../src/test-utils/mocks/mock-dataset-preprocessing-info';
import { getMockedDatasetPreprocessingJob } from '../src/test-utils/mocks/mock-dataset-preprocessing-job';
import { getMockedEnvironment } from '../src/test-utils/mocks/mock-environment';
import { getMockedProject } from '../src/test-utils/mocks/mock-project';
import { expect, http, test } from './fixtures';

test('depth preprocessing accepts a batch size and stays visible in Jobs after closing the popup', async ({
    page,
    network,
}) => {
    const dataset = getMockedDataset();
    const project = getMockedProject({ datasets: [dataset] });
    const job = getMockedDatasetPreprocessingJob({ status: 'running', progress: 40, message: 'Computing depth' });
    let submitted: unknown;
    let interrupted: string | undefined;
    network.use(
        http.get('/api/projects', () => HttpResponse.json([project])),
        http.get('/api/projects/{project_id}', () => HttpResponse.json(project)),
        http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(dataset)),
        http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json([])),
        http.get('/api/projects/{project_id}/environments/{environment_id}', () =>
            HttpResponse.json(getMockedEnvironment())
        ),
        http.get('/api/dataset/{dataset_id}/preprocessing', () =>
            HttpResponse.json(getMockedDatasetPreprocessingInfo())
        ),
        http.get('/api/jobs', () => HttpResponse.json(submitted === undefined ? [] : [job])),
        http.post('/api/jobs/{job_id}:interrupt', ({ params }) => {
            interrupted = params.job_id;
            job.status = 'canceled';
            return new HttpResponse(null, { status: 200 });
        }),
        http.post('/api/dataset/{dataset_id}/preprocessing', async ({ request }) => {
            submitted = await request.json();
            return HttpResponse.json(job, { status: 202 });
        })
    );

    await page.goto(`/projects/${project.id}/datasets/${dataset.id}`);
    await page.getByRole('tab', { selected: true }).locator('[aria-haspopup="true"]').click();
    await page.getByRole('menuitem', { name: 'Preprocess dataset' }).click();

    const dialog = page.getByRole('dialog', { name: 'Preprocess dataset' });
    const start = dialog.getByRole('button', { name: 'Compute depth maps', exact: true });
    await expect(dialog.getByRole('button', { name: 'Close', exact: true })).toBeVisible();
    await expect(start).toBeVisible();
    await expect(start).toBeEnabled();
    await dialog.getByRole('textbox', { name: 'Batch size' }).fill('4');
    await dialog.getByRole('textbox', { name: 'Batch size' }).press('Tab');
    await expect(dialog.getByText(/Estimated VRAM budget: ~2.5 GiB/)).toBeVisible();
    await start.click();
    await expect
        .poll(() => submitted)
        .toMatchObject({ operation: 'depth', rgb_key: null, depth_key: null, batch_size: 4 });
    await dialog.getByRole('button', { name: 'Close', exact: true }).click();
    await expect(dialog).toBeHidden();

    await page.getByRole('button', { name: 'Jobs', exact: true }).click();
    const jobs = page.getByRole('dialog', { name: 'Current project jobs' });
    await expect(jobs.getByText('Compute depth maps', { exact: true })).toBeVisible();
    await expect(jobs.getByRole('progressbar', { name: /Preprocessing progress/ })).toHaveAttribute(
        'aria-valuenow',
        '40'
    );
    await jobs.getByRole('button', { name: 'Cancel task' }).click();
    await expect.poll(() => interrupted).toBe(job.id);
    await expect(jobs.getByText('canceled', { exact: true })).toBeVisible();
});
