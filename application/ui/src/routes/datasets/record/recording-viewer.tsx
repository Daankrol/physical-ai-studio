import { FormEvent, useCallback, useEffect, useState } from 'react';

import {
    Button,
    ButtonGroup,
    ComboBox,
    Flex,
    Form,
    Heading,
    Icon,
    Item,
    Keyboard,
    ProgressCircle,
    StatusLight,
    Text,
    View,
} from '@geti-ui/ui';
import { Accept, Delete, Video } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { formatKeyCombo, getEffectiveBindings } from '../../../features/hotkeys/key-combo';
import { useHotkey } from '../../../features/hotkeys/use-hotkey';
import { RobotControlView } from '../../../features/robots/robot-control/robot-control-view';
import { RobotModelsProvider } from '../../../features/robots/robot-models-context';
import { useRuntimeSession } from '../../../features/robots/runtime-session-provider';
import { paths } from '../../../paths';

import classes from './recording-viewer.module.css';

const RecordingTimer = () => {
    const [seconds, setSeconds] = useState(0);

    useEffect(() => {
        const timer = window.setInterval(() => setSeconds((value) => value + 1), 1000);
        return () => window.clearInterval(timer);
    }, []);

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60)
        .toString()
        .padStart(2, '0');
    const remainingSeconds = (seconds % 60).toString().padStart(2, '0');

    return (
        <span className={classes.timer} aria-hidden='true'>
            {hours > 0
                ? hours.toString().padStart(2, '0') + ':' + minutes + ':' + remainingSeconds
                : minutes + ':' + remainingSeconds}
        </span>
    );
};

const RecordingDot = () => <span className={classes.recordingDot} aria-hidden='true' />;

const ToolbarRecordingStatus = () => (
    <div className={classes.toolbarStatus} role='status' aria-live='polite'>
        <RecordingDot />
        <div className={classes.statusCopy}>
            <strong>Recording episode</strong>
        </div>
        <RecordingTimer />
    </div>
);

export const RecordingViewer = () => {
    const {
        dataset,
        state,
        startEpisode,
        setFollowerSource,
        discardEpisode,
        saveEpisode,
        readyForRecording,
        environment,
        observation,
        actions,
    } = useRuntimeSession();

    if (dataset === undefined) {
        throw 'Cannot load recording viewer without dataset.';
    }
    const [task, setTask] = useState<string>(dataset.default_task);

    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');
    const bindings = getEffectiveBindings(settings.hotkeys.bindings);

    const teleopReady = state.follower_source === 'teleop' && state.handover_state === 'idle';
    const canStartEpisode =
        state.has_leader &&
        state.handover_state === 'idle' &&
        !state.is_recording &&
        !startEpisode.isPending &&
        !setFollowerSource.isPending &&
        !saveEpisode.isPending &&
        !discardEpisode.isPending;
    const startRecording = useCallback(async () => {
        if (!canStartEpisode || task === '') return;
        try {
            if (!teleopReady) {
                const result = await setFollowerSource.mutateAsync({
                    follower_source: 'teleop',
                    auto_align_leader: false,
                });
                if (result.data?.follower_source !== 'teleop' || result.data.handover_state !== 'idle') return;
            }
            startEpisode.mutate(task);
        } catch {
            // The provider reports runtime failures to its configured toast handler.
        }
    }, [canStartEpisode, setFollowerSource, startEpisode, task, teleopReady]);
    const finishEpisode = useCallback(
        async (save: boolean) => {
            try {
                if (
                    state.follower_source !== 'hold' ||
                    state.handover_state === 'aligning' ||
                    state.handover_state === undefined
                ) {
                    await setFollowerSource.mutateAsync('hold');
                }
                await (save ? saveEpisode : discardEpisode).mutateAsync();
            } catch {
                // The provider reports runtime failures to its configured toast handler.
            }
        },
        [discardEpisode, saveEpisode, setFollowerSource, state.follower_source, state.handover_state]
    );

    useHotkey(bindings['recording.start_episode'], () => void startRecording(), canStartEpisode && task !== '');
    useHotkey(
        bindings['recording.accept_episode'],
        () => void finishEpisode(true),
        state.is_recording && !saveEpisode.isPending && !discardEpisode.isPending && !setFollowerSource.isPending
    );
    useHotkey(
        bindings['recording.discard_episode'],
        () => void finishEpisode(false),
        state.is_recording && !discardEpisode.isPending && !saveEpisode.isPending && !setFollowerSource.isPending
    );

    const onStart = (e: FormEvent) => {
        e.preventDefault();
        void startRecording();
    };

    if (!readyForRecording) {
        return (
            <Flex width='100%' height={'100%'} alignItems={'center'} justifyContent={'center'} direction={'column'}>
                <Heading level={2}>
                    <Text>Initializing</Text>
                    <ProgressCircle marginStart='size-200' size='S' isIndeterminate alignSelf={'center'} />
                </Heading>
                <Flex direction='column' margin='size-200'>
                    <StatusLight variant={state.dataset_loaded ? 'positive' : 'yellow'}>Dataset</StatusLight>
                    <StatusLight variant={state.connected ? 'positive' : 'yellow'}>Environment</StatusLight>
                </Flex>
                <Button
                    variant={'secondary'}
                    href={paths.project.datasets.show({
                        dataset_id: dataset.id!,
                        project_id: dataset.project_id,
                    })}
                >
                    Cancel
                </Button>
            </Flex>
        );
    }

    return (
        <RobotModelsProvider>
            <Flex direction={'column'} height={'100%'} position={'relative'}>
                <Form validationBehavior='native' onSubmit={onStart}>
                    <View
                        backgroundColor='gray-100'
                        borderColor='gray-300'
                        borderWidth='thin'
                        borderRadius='regular'
                        padding='size-150'
                        marginBottom='size-150'
                    >
                        <Flex justifyContent='start' alignItems='center' gap='size-150' wrap='wrap'>
                            <ComboBox
                                isReadOnly={state.is_recording}
                                errorMessage={'A task is required in order to record.'}
                                name='Task'
                                aria-label='Task'
                                flex='1 1 240px'
                                minWidth='size-2400'
                                isRequired
                                allowsCustomValue
                                inputValue={task}
                                onInputChange={setTask}
                            >
                                <Item key={dataset.default_task}>{dataset.default_task}</Item>
                            </ComboBox>
                            {state.is_recording && <ToolbarRecordingStatus />}
                            {state.is_recording ? (
                                <ButtonGroup>
                                    <Button
                                        isDisabled={
                                            saveEpisode.isPending ||
                                            discardEpisode.isPending ||
                                            setFollowerSource.isPending
                                        }
                                        variant={'negative'}
                                        style='outline'
                                        isPending={discardEpisode.isPending}
                                        onPress={() => void finishEpisode(false)}
                                    >
                                        <Icon size='S'>
                                            <Delete />
                                        </Icon>
                                        <Text>Discard</Text>
                                        <Keyboard UNSAFE_className={classes.hotkey}>
                                            {formatKeyCombo(bindings['recording.discard_episode'])}
                                        </Keyboard>
                                    </Button>
                                    <Button
                                        variant='accent'
                                        style='fill'
                                        isDisabled={discardEpisode.isPending || setFollowerSource.isPending}
                                        isPending={saveEpisode.isPending}
                                        onPress={() => void finishEpisode(true)}
                                    >
                                        <Icon size='S'>
                                            <Accept />
                                        </Icon>
                                        <Text>Accept</Text>
                                        <Keyboard UNSAFE_className={classes.hotkey}>
                                            {formatKeyCombo(bindings['recording.accept_episode'])}
                                        </Keyboard>
                                    </Button>
                                </ButtonGroup>
                            ) : (
                                <Button
                                    variant='accent'
                                    style='fill'
                                    type={'submit'}
                                    isDisabled={!canStartEpisode}
                                    isPending={startEpisode.isPending}
                                >
                                    <Icon size='S'>
                                        <Video />
                                    </Icon>
                                    <Text>Start episode</Text>
                                    <Keyboard UNSAFE_className={classes.hotkey}>
                                        {formatKeyCombo(bindings['recording.start_episode'])}
                                    </Keyboard>
                                </Button>
                            )}
                        </Flex>
                    </View>
                </Form>
                {!state.has_leader && <div role='alert'>Recording needs a leader arm.</div>}
                <RobotControlView
                    environment={environment}
                    isReady={state.connected}
                    joints={{
                        get current() {
                            return actions.current ?? observation.current;
                        },
                    }}
                />
            </Flex>
        </RobotModelsProvider>
    );
};
