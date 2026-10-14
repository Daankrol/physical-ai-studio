import { useCallback, useState } from 'react';

import {
    Button,
    ButtonGroup,
    ComboBox,
    Flex,
    Heading,
    Item,
    Link,
    ProgressCircle,
    StatusLight,
    Switch,
    Text,
} from '@geti-ui/ui';
import { Back, DownloadIcon, Pause, Play } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { paths } from '../../../router';
import { formatKeyCombo, getEffectiveBindings } from '../../hotkeys/key-combo';
import { useHotkey } from '../../hotkeys/use-hotkey';
import { useProjectId } from '../../projects/use-project';
import { RobotControlView } from '../../robots/robot-control/robot-control-view';
import { RobotModelsProvider } from '../../robots/robot-models-context';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { runtimeExportUrl } from '../runtime-export';

interface InferenceViewerProps {
    tasks: string[];
    collect?: boolean;
    defaultTask?: string;
}

const errorMessage = (error: unknown) => (error instanceof Error ? error.message : 'An unexpected error occurred.');

export const InferenceViewer = ({ tasks, collect = false, defaultTask }: InferenceViewerProps) => {
    const { project_id } = useProjectId();
    const [task, setTask] = useState<string>(defaultTask ?? tasks[0] ?? '');
    const [actionError, setActionError] = useState<string>();

    const {
        model,
        readyForInference,
        readyForRecording,
        state,
        startTask,
        stopTask,
        setFollowerSource,
        startEpisode,
        saveEpisode,
        discardEpisode,
        environment,
        observation,
        inferenceDevice,
    } = useRuntimeSession();

    const { data: settings } = $api.useQuery('get', '/api/settings');
    const bindings = getEffectiveBindings(settings?.hotkeys.bindings);
    const canTeleoperate = state.has_leader;
    const isTeleoperating = state.follower_source === 'teleop';
    const handoverInProgress = state.handover_state !== 'idle';
    const ready = readyForInference && (!collect || readyForRecording);

    const requestTeleop = useCallback(
        async (confirmManualAlignment = false) => {
            setActionError(undefined);
            try {
                return await setFollowerSource.mutateAsync(
                    confirmManualAlignment ? { follower_source: 'teleop', confirm_manual_alignment: true } : 'teleop'
                );
            } catch (error) {
                setActionError(errorMessage(error));
                return undefined;
            }
        },
        [setFollowerSource]
    );

    const startCorrection = useCallback(
        async (confirmManualAlignment = false) => {
            setActionError(undefined);
            if (state.follower_source !== 'teleop' || state.handover_state !== 'idle') {
                const result = await requestTeleop(confirmManualAlignment);
                if (result?.data?.follower_source !== 'teleop' || result.data.handover_state !== 'idle') {
                    return;
                }
            }
            try {
                await startEpisode.mutateAsync(task);
            } catch (error) {
                setActionError(errorMessage(error));
            }
        },
        [requestTeleop, startEpisode, state.follower_source, state.handover_state, task]
    );

    const endCorrection = useCallback(
        async (save: boolean) => {
            setActionError(undefined);
            try {
                if (
                    state.follower_source !== 'hold' ||
                    state.handover_state === 'aligning' ||
                    state.handover_state === undefined
                ) {
                    await setFollowerSource.mutateAsync('hold');
                }
                if (save) {
                    await saveEpisode.mutateAsync();
                } else {
                    await discardEpisode.mutateAsync();
                }
            } catch (error) {
                setActionError(errorMessage(error));
            }
        },
        [discardEpisode, saveEpisode, setFollowerSource, state.follower_source, state.handover_state]
    );

    const changeSource = useCallback(
        (enabled: boolean) => {
            if (enabled) {
                void requestTeleop();
                return;
            }
            setActionError(undefined);
            setFollowerSource.mutateAsync('hold').catch((error: unknown) => setActionError(errorMessage(error)));
        },
        [requestTeleop, setFollowerSource]
    );

    useHotkey(
        bindings['recording.start_episode'],
        () => void startCorrection(),
        collect &&
            !state.is_recording &&
            canTeleoperate &&
            task !== '' &&
            !handoverInProgress &&
            !startEpisode.isPending
    );
    useHotkey(
        bindings['recording.accept_episode'],
        () => void endCorrection(true),
        collect &&
            state.is_recording &&
            !setFollowerSource.isPending &&
            !saveEpisode.isPending &&
            !discardEpisode.isPending
    );
    useHotkey(
        bindings['recording.discard_episode'],
        () => void endCorrection(false),
        collect &&
            state.is_recording &&
            !setFollowerSource.isPending &&
            !saveEpisode.isPending &&
            !discardEpisode.isPending
    );

    const exportUrl =
        model?.id !== undefined && inferenceDevice !== undefined
            ? runtimeExportUrl({
                  modelId: model.id,
                  environmentId: environment.id,
                  backend: inferenceDevice.backend,
                  device: inferenceDevice.device,
                  task,
              })
            : undefined;

    if (!ready) {
        return (
            <Flex width='100%' height='100%' alignItems='center' justifyContent='center' direction='column'>
                <Heading level={2}>
                    <Text>Initializing</Text>
                    <ProgressCircle marginStart='size-200' size='S' isIndeterminate alignSelf='center' />
                </Heading>
                <Flex direction='column' margin='size-200'>
                    <StatusLight variant={state.model_loaded ? 'positive' : 'yellow'}>Model</StatusLight>
                    {collect && (
                        <StatusLight variant={state.dataset_loaded ? 'positive' : 'yellow'}>Dataset</StatusLight>
                    )}
                    <StatusLight variant={state.connected ? 'positive' : 'yellow'}>Environment</StatusLight>
                </Flex>
                <Button variant='secondary' href={paths.project.models.index({ project_id })}>
                    Cancel
                </Button>
            </Flex>
        );
    }

    const busy =
        setFollowerSource.isPending ||
        startEpisode.isPending ||
        saveEpisode.isPending ||
        discardEpisode.isPending ||
        startTask.isPending ||
        stopTask.isPending;
    const controlsDisabled = collect && !canTeleoperate;
    const canResumePolicy = !state.is_recording && !handoverInProgress && !busy && !controlsDisabled;

    return (
        <RobotModelsProvider>
            <Flex flex direction='column' height='100%' position='relative'>
                <Flex alignItems='center' gap='size-100' height='size-400' margin='size-200'>
                    <Link aria-label='Rewind' href={paths.project.models.index({ project_id })}>
                        <Back fill='white' />
                    </Link>
                    <Heading>Model Run {model?.name}</Heading>
                    <ComboBox flex isRequired allowsCustomValue={false} inputValue={task} onInputChange={setTask}>
                        {tasks.map((taskText, index) => (
                            <Item key={index}>{taskText}</Item>
                        ))}
                    </ComboBox>
                    {collect ? (
                        state.is_recording ? (
                            <ButtonGroup>
                                <Button variant='negative' isDisabled={busy} onPress={() => void endCorrection(false)}>
                                    <Text>Discard</Text>
                                    <Text aria-label='Discard hotkey'>
                                        {formatKeyCombo(bindings['recording.discard_episode'])}
                                    </Text>
                                </Button>
                                <Button variant='primary' isDisabled={busy} onPress={() => void endCorrection(true)}>
                                    <Text>Save correction</Text>
                                    <Text aria-label='Save hotkey'>
                                        {formatKeyCombo(bindings['recording.accept_episode'])}
                                    </Text>
                                </Button>
                            </ButtonGroup>
                        ) : (
                            <Button
                                variant='accent'
                                isDisabled={controlsDisabled || task === '' || busy || handoverInProgress}
                                isPending={setFollowerSource.isPending || startEpisode.isPending}
                                onPress={() => void startCorrection()}
                            >
                                Take over
                                <Text aria-label='Take over hotkey'>
                                    {formatKeyCombo(bindings['recording.start_episode'])}
                                </Text>
                            </Button>
                        )
                    ) : (
                        <Switch
                            isEmphasized
                            isSelected={isTeleoperating}
                            isDisabled={
                                !canTeleoperate ||
                                setFollowerSource.isPending ||
                                startTask.isPending ||
                                stopTask.isPending
                            }
                            onChange={changeSource}
                        >
                            Teleoperate
                        </Switch>
                    )}
                    <ButtonGroup>
                        {exportUrl !== undefined && (
                            <Button
                                href={exportUrl}
                                aria-label='Download runtime export'
                                variant='secondary'
                                target='_blank'
                                rel='noopener noreferrer'
                            >
                                <DownloadIcon />
                                Runtime export
                            </Button>
                        )}
                        {state.follower_source === 'policy' ? (
                            <Button variant='primary' isPending={stopTask.isPending} onPress={() => stopTask.mutate()}>
                                <Pause fill='white' />
                                Stop
                            </Button>
                        ) : (
                            <Button
                                variant='primary'
                                isDisabled={!canResumePolicy}
                                isPending={startTask.isPending}
                                onPress={() => startTask.mutate(task)}
                            >
                                <Play fill='white' />
                                Play
                            </Button>
                        )}
                    </ButtonGroup>
                </Flex>
                {collect && (
                    <Flex direction='column' gap='size-50' marginX='size-200'>
                        <Text>{state.episodes_recorded} corrections recorded this session</Text>
                        {!canTeleoperate && <div role='alert'>Collecting corrections needs a leader arm.</div>}
                    </Flex>
                )}
                {state.handover_state === 'aligning' && <div role='status'>Aligning leader; follower is holding.</div>}
                {state.handover_state === 'manual_required' && (
                    <Flex alignItems='center' gap='size-100' marginX='size-200'>
                        <Text>
                            Manual alignment required: move the leader to match the held follower, then confirm.
                        </Text>
                        <Button
                            isDisabled={!canTeleoperate || busy}
                            isPending={setFollowerSource.isPending}
                            onPress={() => (collect ? void startCorrection(true) : void requestTeleop(true))}
                        >
                            Confirm manual alignment
                        </Button>
                    </Flex>
                )}
                {actionError && <div role='alert'>{actionError}</div>}
                <RobotControlView environment={environment} isReady={state.connected} joints={observation} />
            </Flex>
        </RobotModelsProvider>
    );
};
