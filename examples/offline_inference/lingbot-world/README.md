# LingBot-World

Offline LingBot-World examples.

## Download

```bash
python download_lingbot_world.py \
  --model-id robbyant/lingbot-world-base-cam \
  --output-dir ./lingbot-world-base-cam
```

The prepared model directory looks like this:

```text
lingbot-world-base-cam/
├── configuration.json
├── google/
├── high_noise_model/
├── low_noise_model/
├── models_t5_umt5-xxl-enc-bf16.pth
├── model_index.json
└── Wan2.1_VAE.pth
```

## Run With Control Signals

```bash
PROMPT="$(cat /tmp/vllm-omni-dependency/lingbot-world/examples/00/prompt.txt)"

python image_to_video.py \
  --model ./lingbot-world-base-cam \
  --image /tmp/vllm-omni-dependency/lingbot-world/examples/00/image.jpg \
  --action-path /tmp/vllm-omni-dependency/lingbot-world/examples/00 \
  --prompt "$PROMPT" \
  --output lingbot_world_base_cam_examples00.mp4
```

## Run Without Control Signals

```bash
PROMPT="$(cat /tmp/vllm-omni-dependency/lingbot-world/examples/00/prompt.txt)"

python image_to_video.py \
  --model ./lingbot-world-base-cam \
  --image /tmp/vllm-omni-dependency/lingbot-world/examples/00/image.jpg \
  --prompt "$PROMPT" \
  --output lingbot_world_base_cam_no_control.mp4
```

## Notes

- `--action-path` is optional.
- For `LingBot-World-Base (Cam)`, control signals should contain `poses.npy` and `intrinsics.npy`.
- For `LingBot-World-Base (Act)`, `action.npy` is also required.
- `--enable-cpu-offload` is supported for offline inference.
