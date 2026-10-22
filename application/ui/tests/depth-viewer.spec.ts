// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { HttpResponse } from 'msw';

import { getMockedDataset } from '../src/test-utils/mocks/mock-dataset';
import { getMockedEnvironment } from '../src/test-utils/mocks/mock-environment';
import { getMockedEpisode, getMockedEpisodeDetails } from '../src/test-utils/mocks/mock-episode';
import { getMockedProject } from '../src/test-utils/mocks/mock-project';
import { expect, http, test } from './fixtures';

test('colorizes cached grayscale depth previews without altering RGB or invalid depth', async ({ page, network }) => {
    const dataset = getMockedDataset();
    const project = getMockedProject({ datasets: [dataset] });
    const episode = getMockedEpisodeDetails({
        depth_videos: { 'observation.images.front': { start: 0, end: 1, path: 'depth.mp4' } },
    });
    network.use(
        http.get('/api/projects', () => HttpResponse.json([project])),
        http.get('/api/projects/{project_id}', () => HttpResponse.json(project)),
        http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(dataset)),
        http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json([getMockedEpisode()])),
        http.get('/api/dataset/{dataset_id}/episodes/{episode_index}', () => HttpResponse.json(episode)),
        http.get('/api/projects/{project_id}/environments/{environment_id}', () =>
            HttpResponse.json(getMockedEnvironment())
        ),
        http.get('/api/jobs', () => HttpResponse.json([]))
    );
    await page.goto(`/projects/${project.id}/datasets/${dataset.id}`);
    const video = page.locator('video').first();

    const renderedPixel = async (gray: number) => {
        await video.evaluate(async (element, value) => {
            if (!(element instanceof HTMLVideoElement)) throw new Error('Video is unavailable');
            const image = new Image();
            image.src = `data:image/svg+xml,${encodeURIComponent(
                `<svg xmlns="http://www.w3.org/2000/svg" width="160" height="120"><rect width="160" height="120" fill="rgb(${value},${value},${value})"/></svg>`
            )}`;
            await image.decode();
            element.removeAttribute('src');
            element.load();
            element.poster = image.src;
            element.style.width = '160px';
            element.style.height = '120px';
        }, gray);
        const screenshot = await video.screenshot();
        return page.evaluate(async (base64) => {
            const image = new Image();
            image.src = `data:image/png;base64,${base64}`;
            await image.decode();
            const canvas = document.createElement('canvas');
            canvas.width = image.width;
            canvas.height = image.height;
            const context = canvas.getContext('2d');
            if (context === null) throw new Error('Canvas is unavailable');
            context.drawImage(image, 0, 0);
            return Array.from(context.getImageData(canvas.width / 2, canvas.height / 2, 1, 1).data).slice(0, 3);
        }, screenshot.toString('base64'));
    };

    await expect(page.getByRole('button', { name: /Camera view/ })).toBeVisible();
    expect(await renderedPixel(128)).toEqual([128, 128, 128]);
    await page.getByRole('button', { name: /Camera view/ }).click();
    await page.getByRole('option', { name: 'Depth', exact: true }).click();
    const far = await renderedPixel(64);
    const near = await renderedPixel(224);
    expect(far[2]).toBeGreaterThan(far[0] + 100);
    expect(near[0]).toBeGreaterThan(near[2] + 100);
    expect(await renderedPixel(0)).toEqual([0, 0, 0]);
    await page.getByRole('button', { name: /Camera view/ }).click();
    await page.getByRole('option', { name: 'RGB', exact: true }).click();
    expect(await renderedPixel(128)).toEqual([128, 128, 128]);
});
