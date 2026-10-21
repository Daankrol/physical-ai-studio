# Deploying Model Policies

| **Run model policy**               | **Model inference screen**                  |
|------------------------------------|---------------------------------------------|
| ![Run model policy][inference-run] | ![Model inference screen][inference-screen] |

[inference-run]: ./assets/07-inference-run.png
[inference-screen]: ./assets/07-inference-screen.png

Models can run inside Physical AI Studio or be deployed with [OpenVINO PhysicalAI](https://github.com/openvinotoolkit/physicalai).
On the Models screen, click Run model for a trained policy, then select the inference backend and device. Studio currently supports PyTorch and OpenVINO on GPU or CPU.

When you start the model, Studio loads the environment used to record the dataset the model was trained on.
After the environment and model load, you will see a screen similar to the recording view. Pick the task the model should perform and click Play.

## Collect human corrections

To collect recovery demonstrations while running a policy, choose **Collect corrections into the training dataset** when launching the model. Studio loads the model's original training dataset into the same robot session. The screen needs a leader arm as well as the follower.

1. Click **Play** to run the policy.
2. When the policy needs correction, click **Take over** or use the configured Start episode hotkey. The follower holds while Studio aligns the leader.
3. For powered alignment, the yellow banner reports whether the leader reached tolerance. Take hold of the leader, then click **Release alignment** to turn its torque off and take over, even if alignment missed tolerance. Studio waits for this button without a release timer and does not ask for another alignment confirmation. The follower starts following the leader with bounded initial steps, so check the remaining pose difference before releasing.
4. If automatic alignment is unavailable, the banner offers the same **Release alignment** button. Take hold of the leader and release to take over. Neither this fallback nor a missed automatic alignment requires matching poses or a second confirmation.
5. Demonstrate the recovery. Only fresh human teleoperation frames are recorded. The policy's actions, alignment movement and hold ticks are not correction labels.
6. Click **Save correction** or **Discard**. Saved episodes show a **Human correction** badge in the dataset. Click **Play** to return control to the policy and repeat without resetting the scene.

The original demonstrations stay in the dataset. Retrain from the model on that dataset so training sees both the original task demonstrations and the new recovery examples. Studio currently uses the human-gated variant of DAgger: it records corrections when a person takes over, not expert labels for every state the policy visits. The intervention flag identifies frames in the dataset; it is not a reward or training-loss function.

**Hardware note:** the pinned Runtime and SO101 plugin do not support automatic SO101 alignment. See [Test Studio DAgger handover](./testing-auto-alignment.md) for the dependency requirements and hardware checks. With the pinned dependencies, take hold of the leader and use **Release alignment** without powered movement. Keep a hand on the stop control and test takeover in a clear workspace; the full handover has not been validated on hardware.

## Next

- For training, see [Training Policies](./06-training-policies.md).
