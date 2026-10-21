import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { useLocation } from 'react-router';

import { SchemaModel } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { StartInferenceDialog } from './start-inference-dialog';

const model = {
    id: 'model-1',
    name: 'ACT',
    path: '/models/act',
    policy: 'act',
    properties: {},
    project_id: 'project-1',
    dataset_id: 'dataset-1',
    snapshot_id: null,
    train_job_id: 'job-1',
    parent_model_id: null,
    version: 1,
    created_at: '2026-07-14T12:00:00Z',
    available_backends: ['openvino'],
    lora_enabled: false,
    lora_use_dora: false,
    snapflow_enabled: false,
} as SchemaModel;

const Location = () => {
    const location = useLocation();
    return <output aria-label='Current route'>{location.pathname + location.search}</output>;
};

describe('StartInferenceDialog', () => {
    it('adds collect=1 to the inference route when correction collection is selected', async () => {
        const user = userEvent.setup();
        server.use(
            http.get('/api/system/devices/inference', () =>
                HttpResponse.json([{ backend: 'openvino', device: 'CPU', type: 'cpu', name: 'CPU' }])
            ),
            http.get('/api/policies/backends', () => HttpResponse.json({ act: ['openvino'] }))
        );
        render(
            <>
                <StartInferenceDialog close={() => {}} model={model} />
                <Location />
            </>,
            {
                route: '/projects/project-1/models/model-1/inference/openvino',
                path: '/projects/:project_id/models/:model_id/inference/:backend',
            }
        );

        await user.click(
            await screen.findByRole('checkbox', { name: 'Collect corrections into the training dataset' })
        );
        await user.click(await screen.findByRole('button', { name: 'Start' }));

        expect(await screen.findByLabelText('Current route')).toHaveTextContent(
            '/projects/project-1/models/model-1/inference/openvino?device=CPU&collect=1'
        );
    });
});
