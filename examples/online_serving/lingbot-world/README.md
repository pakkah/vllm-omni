# LingBot-World

Online LingBot-World examples using the async `/v1/videos` API.

## Download

```bash
python ../../offline_inference/lingbot-world/download_lingbot_world.py \
  --model-id robbyant/lingbot-world-base-cam \
  --output-dir ./lingbot-world-base-cam
```

## Start Server

```bash
MODEL=./lingbot-world-base-cam \
bash run_server.sh
```

Or:

```bash
vllm serve ./lingbot-world-base-cam --omni --port 8099
```

## Submit a Job

```bash
PROMPT="$(cat /tmp/vllm-omni-dependency/lingbot-world/examples/00/prompt.txt)"
INPUT_IMAGE=/tmp/vllm-omni-dependency/lingbot-world/examples/00/image.jpg \
PROMPT="$PROMPT" \
bash run_curl_image_to_video.sh
```

The script follows the generic `examples/online_serving/image_to_video` flow and uses the same 480P defaults as the upstream LingBot-World example.

- `POST /v1/videos`
- `GET /v1/videos/{video_id}`
- `GET /v1/videos/{video_id}/content`

## Current Limitation

`action_path` is not exposed by the current online `/v1/videos` API. Control signals are offline-only for now.
