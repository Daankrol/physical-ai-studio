export const TRAINING_VAL_SPLIT = 0.1;

export interface TrainingStepRange {
    min: number;
    max: number;
}

export const getTrainingStepRange = (
    episodeLengths: readonly number[],
    batchSize: number,
    epochs: number
): TrainingStepRange | undefined => {
    if (episodeLengths.length === 0 || batchSize < 1 || epochs < 1) {
        return undefined;
    }

    const validationEpisodes = Math.max(1, Math.floor(episodeLengths.length * TRAINING_VAL_SPLIT));
    const sortedLengths = [...episodeLengths].sort((a, b) => a - b);
    const totalSamples = sortedLengths.reduce((sum, length) => sum + length, 0);
    const validationMinSamples = sortedLengths.slice(0, validationEpisodes).reduce((sum, length) => sum + length, 0);
    const validationMaxSamples = sortedLengths.slice(-validationEpisodes).reduce((sum, length) => sum + length, 0);

    return {
        min: Math.floor((totalSamples - validationMaxSamples) / batchSize) * epochs,
        max: Math.floor((totalSamples - validationMinSamples) / batchSize) * epochs,
    };
};
