import { screen } from '@testing-library/react';

import { EpisodeSummary } from '../../../routes/datasets/dataset-provider';
import { render } from '../../../test-utils/render';
import { EpisodeTag } from './episode-tag';

const episode: EpisodeSummary = { episode_index: 1, tasks: [], length: 30, fps: 30 };

describe('EpisodeTag', () => {
    it.each(['small', 'medium'] as const)('shows a human correction badge in the %s variant', (variant) => {
        render(<EpisodeTag episode={{ ...episode, intervention: true }} variant={variant} />);

        expect(screen.getByText('Human correction')).toBeInTheDocument();
    });

    it.each([false, undefined])('does not show a badge when intervention is %s', (intervention) => {
        render(<EpisodeTag episode={{ ...episode, intervention }} variant='small' />);

        expect(screen.queryByText('Human correction')).not.toBeInTheDocument();
    });
});
