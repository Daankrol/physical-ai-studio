import { useCallback, useState } from 'react';

import {
    AlertDialog,
    Button,
    ButtonGroup,
    ComboBox,
    DialogContainer,
    Flex,
    Heading,
    Icon,
    Item,
    Keyboard,
    Link,
    ProgressCircle,
    StatusLight,
    Switch,
    Text,
    View,
} from '@geti-ui/ui';
import { Accept, Back, Delete, DownloadIcon, Human, Pause, Play } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { paths } from '../../../paths';
import { formatKeyCombo, getEffectiveBindings } from '../../hotkeys/key-combo';
import { useHotkey } from '../../hotkeys/use-hotkey';
import { useProjectId } from '../../projects/use-project';
import { RobotControlView } from '../../robots/robot-control/robot-control-view';
import { RobotModelsProvider } from '../../robots/robot-models-context';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { runtimeExportUrl } from '../runtime-export';

import classes from './inference-viewer.module.css';

interface InferenceViewerProps {
    tasks: string[];
    collect?: boolean;
    defaultTask?: string;
}

const errorMessage = (error: unknown) => (error instanceof Error ? error.message : 'An unexpected error occurred.');

export const InferenceViewer = ({ tasks, collect = false, defaultTask }: InferenceViewerProps) => {
    const { project_id } = useProjectId();
    // Keep the typed prompt when the field loses focus; dataset tasks are suggestions.
    const [task, setTask] = useState<string>(defaultTask ?? tasks[0] ?? '');
    const [actionError, setActionError] = useState<string>();
    const [isEmptyPromptDialogOpen, setIsEmptyPromptDialogOpen] = useState(false);

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
    const awaitingRelease =
        state.handover_state === 'awaiting_release' ||
        state.handover_state === 'alignment_failed' ||
        state.handover_state === 'manual_required';
    const ready = readyForInference && (!collect || readyForRecording);

    const requestTeleop = useCallback(
        async (action: 'align' | 'release' = 'align') => {
            setActionError(undefined);
            try {
                return await setFollowerSource.mutateAsync({
                    follower_source: 'teleop',
                    auto_align_leader: action === 'align',
                    ...(action === 'release' ? { release_alignment: true } : {}),
                });
            } catch (error) {
                setActionError(errorMessage(error));
                return undefined;
            }
        },
        [setFollowerSource]
    );

    const startCorrection = useCallback(
        async (action: 'align' | 'release' = 'align') => {
            setActionError(undefined);
            if (state.follower_source !== 'teleop' || state.handover_state !== 'idle') {
                const result = await requestTeleop(action);
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
                <View
                    backgroundColor='gray-100'
                    borderColor='gray-300'
                    borderWidth='thin'
                    borderRadius='regular'
                    margin='size-200'
                    padding='size-150'
                    flexShrink={0}
                >
                    <Flex alignItems='center' gap='size-150' wrap='wrap'>
                        <Link aria-label='Rewind' href={paths.project.models.index({ project_id })}>
                            <Back fill='white' />
                        </Link>
                        <Heading margin={0}>Model Run {model?.name}</Heading>
                        <ComboBox
                            flex='1 1 240px'
                            minWidth='size-2400'
                            aria-label='Task prompt'
                            allowsCustomValue
                            inputValue={task}
                            onInputChange={setTask}
                        >
                            {tasks.map((taskText) => (
                                <Item key={taskText}>{taskText}</Item>
                            ))}
                        </ComboBox>
                        {collect ? (
                            state.is_recording ? (
                                <ButtonGroup>
                                    <Button
                                        variant='negative'
                                        style='outline'
                                        isDisabled={busy}
                                        isPending={discardEpisode.isPending}
                                        onPress={() => void endCorrection(false)}
                                    >
                                        <Icon size='S'>
                                            <Delete />
                                        </Icon>
                                        <Text>Discard</Text>
                                        <Keyboard aria-label='Discard hotkey' UNSAFE_className={classes.hotkey}>
                                            {formatKeyCombo(bindings['recording.discard_episode'])}
                                        </Keyboard>
                                    </Button>
                                    <Button
                                        variant='accent'
                                        style='fill'
                                        isDisabled={busy}
                                        isPending={saveEpisode.isPending}
                                        onPress={() => void endCorrection(true)}
                                    >
                                        <Icon size='S'>
                                            <Accept />
                                        </Icon>
                                        <Text>Save correction</Text>
                                        <Keyboard aria-label='Save hotkey' UNSAFE_className={classes.hotkey}>
                                            {formatKeyCombo(bindings['recording.accept_episode'])}
                                        </Keyboard>
                                    </Button>
                                </ButtonGroup>
                            ) : (
                                <Button
                                    variant='accent'
                                    style='fill'
                                    isDisabled={controlsDisabled || task === '' || busy || handoverInProgress}
                                    isPending={setFollowerSource.isPending || startEpisode.isPending}
                                    onPress={() => void startCorrection()}
                                >
                                    <Icon size='S'>
                                        <Human />
                                    </Icon>
                                    <Text>Take over</Text>
                                    <Keyboard aria-label='Take over hotkey' UNSAFE_className={classes.hotkey}>
                                        {formatKeyCombo(bindings['recording.start_episode'])}
                                    </Keyboard>
                                </Button>
                            )
                        ) : (
                            <Switch
                                isEmphasized
                                isSelected={isTeleoperating}
                                isDisabled={
                                    !canTeleoperate ||
                                    handoverInProgress ||
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
                                    <Icon size='S'>
                                        <DownloadIcon />
                                    </Icon>
                                    <Text>Runtime export</Text>
                                </Button>
                            )}
                            {state.follower_source === 'policy' ? (
                                <Button
                                    variant='secondary'
                                    style='outline'
                                    isPending={stopTask.isPending}
                                    onPress={() => stopTask.mutate()}
                                >
                                    <Icon size='S'>
                                        <Pause />
                                    </Icon>
                                    <Text>Stop</Text>
                                </Button>
                            ) : (
                                <Button
                                    variant='primary'
                                    style='fill'
                                    isDisabled={!canResumePolicy}
                                    isPending={startTask.isPending}
                                    onPress={() =>
                                        task.trim() === '' ? setIsEmptyPromptDialogOpen(true) : startTask.mutate(task)
                                    }
                                >
                                    <Icon size='S'>
                                        <Play />
                                    </Icon>
                                    <Text>Play</Text>
                                </Button>
                            )}
                        </ButtonGroup>
                    </Flex>
                </View>
                {collect && (
                    <Flex direction='column' gap='size-50' marginX='size-200'>
                        <Text UNSAFE_className={classes.sessionSummary}>
                            {state.episodes_recorded} corrections recorded this session
                        </Text>
                        {!canTeleoperate && <div role='alert'>Collecting corrections needs a leader arm.</div>}
                    </Flex>
                )}
                {(state.handover_state === 'aligning' || awaitingRelease) && (
                    <View
                        backgroundColor='yellow-400'
                        padding='size-200'
                        marginX='size-200'
                        borderRadius='regular'
                        flexShrink={0}
                        UNSAFE_style={{ color: 'black' }}
                    >
                        <Flex alignItems='center' justifyContent='space-between' gap='size-200' wrap='wrap'>
                            <div role='status'>
                                {state.handover_state === 'aligning' ? (
                                    <>
                                        Aligning leader; follower is holding. Leader torque stays on until you release
                                        alignment.
                                    </>
                                ) : state.handover_state === 'awaiting_release' ? (
                                    <>
                                        Leader reached alignment tolerance. Take hold of the leader, then release
                                        alignment to take over.
                                    </>
                                ) : state.handover_state === 'manual_required' ? (
                                    <>
                                        Automatic alignment is unavailable for this leader. Take hold of the leader,
                                        then release alignment to take over.
                                    </>
                                ) : (
                                    <>
                                        Leader did not reach alignment tolerance. Take hold of the leader, then release
                                        alignment to take over. No further alignment confirmation is needed.
                                    </>
                                )}
                            </div>
                            {awaitingRelease && (
                                <Button
                                    variant='primary'
                                    style='fill'
                                    isDisabled={!canTeleoperate || busy}
                                    isPending={setFollowerSource.isPending}
                                    onPress={() =>
                                        collect ? void startCorrection('release') : void requestTeleop('release')
                                    }
                                >
                                    <Text>Release alignment</Text>
                                </Button>
                            )}
                        </Flex>
                    </View>
                )}
                {actionError && <div role='alert'>{actionError}</div>}
                <RobotControlView environment={environment} isReady={state.connected} joints={observation} />
            </Flex>
            <DialogContainer onDismiss={() => setIsEmptyPromptDialogOpen(false)}>
                {isEmptyPromptDialogOpen && (
                    <AlertDialog
                        title='Start without a task prompt?'
                        variant='warning'
                        primaryActionLabel='Start anyway'
                        secondaryActionLabel='Cancel'
                        onPrimaryAction={() => {
                            setIsEmptyPromptDialogOpen(false);
                            startTask.mutate(task);
                        }}
                        onSecondaryAction={() => setIsEmptyPromptDialogOpen(false)}
                    >
                        <Text>
                            The task prompt is empty. Policies that follow language instructions, such as Pi0.5, will
                            run without an instruction.
                        </Text>
                    </AlertDialog>
                )}
            </DialogContainer>
        </RobotModelsProvider>
    );
};
