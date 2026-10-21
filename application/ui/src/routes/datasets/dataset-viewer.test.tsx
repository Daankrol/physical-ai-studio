import { ReactNode } from 'react';

import { rs } from '@rstest/core';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaEpisode } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { getMockedDataset } from '../../test-utils/mocks/mock-dataset';
import { getMockedEnvironment } from '../../test-utils/mocks/mock-environment';
import { getMockedEpisodes } from '../../test-utils/mocks/mock-episode';
import { render } from '../../test-utils/render';
import { DatasetProvider, EpisodeSummary } from './dataset-provider';
import { DatasetViewer } from './dataset-viewer';

rs.mock('../../components/virtualized-grid-layout/virtualized-grid-layout', () => ({
    VirtualizedGridLayout: ({
        items,
        renderItem,
    }: {
        items: EpisodeSummary[];
        renderItem: (episode: EpisodeSummary) => ReactNode;
    }) => (
        <div aria-label='Episode list'>
            {items.map((episode) => (
                <div key={episode.episode_index}>{renderItem(episode)}</div>
            ))}
        </div>
    ),
}));

rs.mock('./episode-viewer', () => ({
    EpisodeViewer: ({ episode }: { episode: SchemaEpisode }) => <div>Viewing episode {episode.episode_index + 1}</div>,
}));

const episodes = getMockedEpisodes([
    { episode_index: 0 },
    { episode_index: 2, intervention: true },
    { episode_index: 5 },
]);

const renderViewer = (episodeSummaries = episodes) => {
    server.use(
        http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(getMockedDataset())),
        http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json(episodeSummaries)),
        http.get('/api/dataset/{dataset_id}/episodes/{episode_index}', ({ params }) => {
            const episode = episodes.find((item) => item.episode_index === Number(params.episode_index));
            if (!episode) return new HttpResponse(null, { status: 404 });
            return HttpResponse.json({ ...episode, actions: [], action_keys: [], videos: {} });
        }),
        http.get('/api/projects/{project_id}/environments/{environment_id}', () =>
            HttpResponse.json(getMockedEnvironment())
        )
    );

    return render(
        <DatasetProvider dataset_id='dataset-1'>
            <DatasetViewer />
        </DatasetProvider>,
        {
            route: '/projects/project-1/datasets/dataset-1/episodes/5',
            path: '/projects/:project_id/datasets/:dataset_id/episodes/:episode_index',
        }
    );
};

const chooseFilter = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
    await user.click(screen.getByRole('button', { name: /Episode type/ }));
    await user.click(await screen.findByRole('option', { name }));
};

describe('DatasetViewer episode types', () => {
    it('identifies corrections and filters both episode types without changing episode numbers', async () => {
        const user = userEvent.setup();
        renderViewer();

        expect(await screen.findByText('Viewing episode 6')).toBeInTheDocument();
        const list = within(screen.getByLabelText('Episode list'));
        expect(list.getByText('Human correction')).toBeInTheDocument();
        expect(list.getAllByRole('img').map((image) => image.getAttribute('alt'))).toEqual([
            'Camera frame of 5',
            'Camera frame of 2',
            'Camera frame of 0',
        ]);

        await chooseFilter(user, 'Human corrections');

        expect(list.getAllByRole('img')).toHaveLength(1);
        expect(list.getByText('E3')).toBeInTheDocument();
        expect(await screen.findByText('Viewing episode 3')).toBeInTheDocument();

        await chooseFilter(user, 'Regular recordings');

        expect(list.getAllByRole('img')).toHaveLength(2);
        expect(list.queryByText('Human correction')).not.toBeInTheDocument();
        expect(await screen.findByText('Viewing episode 6')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Sort by episode number: newest first' }));
        expect(list.getAllByRole('img').map((image) => image.getAttribute('alt'))).toEqual([
            'Camera frame of 0',
            'Camera frame of 5',
        ]);

        await chooseFilter(user, 'All episodes');
        expect(list.getAllByRole('img')).toHaveLength(3);
        expect(screen.getByText('Viewing episode 6')).toBeInTheDocument();
    });

    it('shows an empty filter state and recovers when another type is chosen', async () => {
        const user = userEvent.setup();
        renderViewer(episodes.filter((episode) => !episode.intervention));
        await screen.findByText('Viewing episode 6');

        await chooseFilter(user, 'Human corrections');

        expect(screen.getByText('No matching episodes')).toBeInTheDocument();
        expect(within(screen.getByLabelText('Episode list')).queryByRole('img')).not.toBeInTheDocument();
        expect(screen.queryByText('Viewing episode 6')).not.toBeInTheDocument();

        await chooseFilter(user, 'All episodes');

        expect(await screen.findByText('Viewing episode 6')).toBeInTheDocument();
        expect(screen.queryByText('No matching episodes')).not.toBeInTheDocument();
    });

    it('clears bulk selection when changing the filter so hidden episodes cannot be deleted', async () => {
        const user = userEvent.setup();
        renderViewer();
        await screen.findByText('Viewing episode 6');

        await user.click(screen.getAllByRole('checkbox')[0]);
        expect(await screen.findByRole('button', { name: 'Clear selection' })).toBeInTheDocument();

        await chooseFilter(user, 'Human corrections');

        await waitFor(() => expect(screen.queryByRole('button', { name: 'Clear selection' })).not.toBeInTheDocument());
        expect(screen.getAllByRole('checkbox')[0]).not.toBeChecked();
    });
});
