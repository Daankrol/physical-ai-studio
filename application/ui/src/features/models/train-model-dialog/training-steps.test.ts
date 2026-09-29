import { describe, expect, it } from 'vitest';

import { getTrainingStepRange } from './training-steps';

describe('getTrainingStepRange', () => {
    it('returns an exact count when episode lengths produce the same train size', () => {
        expect(getTrainingStepRange([100, 100], 8, 5)).toEqual({ min: 60, max: 60 });
    });

    it('returns bounds for the random validation episode split', () => {
        expect(getTrainingStepRange([10, 100], 8, 5)).toEqual({ min: 5, max: 60 });
    });
});
