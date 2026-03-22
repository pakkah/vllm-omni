# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video with LingBot-World.")
    parser.add_argument("--model", default="./lingbot-world-base-cam", help="Model path or Hugging Face ID.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=480, help="Video height.")
    parser.add_argument("--width", type=int, default=832, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=161, help="Number of frames.")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Sampling steps.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Low-noise CFG scale.")
    parser.add_argument("--guidance-scale-high", type=float, default=5.0, help="High-noise CFG scale.")
    parser.add_argument("--boundary-ratio", type=float, default=0.875, help="Boundary split ratio.")
    parser.add_argument("--flow-shift", type=float, default=10.0, help="Scheduler flow shift.")
    parser.add_argument("--action-path", default=None, help="Optional path to control signals.")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the output video.")
    parser.add_argument("--output", default="lingbot_world_output.mp4", help="Output video path.")
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offload.",
    )
    return parser.parse_args()


def extract_frames(output: object) -> tuple[object, object | None]:
    frames = output
    audio = None

    if isinstance(frames, list):
        frames = frames[0] if frames else None

    if isinstance(frames, OmniRequestOutput):
        if frames.final_output_type != "image":
            raise ValueError(f"Unexpected output type '{frames.final_output_type}', expected 'image'.")
        if frames.multimodal_output and "audio" in frames.multimodal_output:
            audio = frames.multimodal_output["audio"]
        if frames.is_pipeline_output and frames.request_output is not None:
            inner_output = frames.request_output
            if isinstance(inner_output, OmniRequestOutput):
                if inner_output.multimodal_output and "audio" in inner_output.multimodal_output:
                    audio = inner_output.multimodal_output["audio"]
                frames = inner_output
        if isinstance(frames, OmniRequestOutput):
            if frames.images:
                if len(frames.images) == 1 and isinstance(frames.images[0], tuple) and len(frames.images[0]) == 2:
                    frames, audio = frames.images[0]
                elif len(frames.images) == 1 and isinstance(frames.images[0], dict):
                    audio = frames.images[0].get("audio")
                    frames = frames.images[0].get("frames") or frames.images[0].get("video")
                else:
                    frames = frames.images
            else:
                raise ValueError("No video frames found in OmniRequestOutput.")

    if isinstance(frames, list) and frames:
        first_item = frames[0]
        if isinstance(first_item, tuple) and len(first_item) == 2:
            frames, audio = first_item
        elif isinstance(first_item, dict):
            audio = first_item.get("audio")
            frames = first_item.get("frames") or first_item.get("video")
        elif isinstance(first_item, list):
            frames = first_item

    if isinstance(frames, tuple) and len(frames) == 2:
        frames, audio = frames
    elif isinstance(frames, dict):
        audio = frames.get("audio")
        frames = frames.get("frames") or frames.get("video")

    if frames is None:
        raise ValueError("No video frames found in output.")

    return frames, audio


def normalize_frame(frame: object) -> object:
    if isinstance(frame, torch.Tensor):
        frame_tensor = frame.detach().cpu()
        if frame_tensor.dim() == 4 and frame_tensor.shape[0] == 1:
            frame_tensor = frame_tensor[0]
        if frame_tensor.dim() == 3 and frame_tensor.shape[0] in (3, 4):
            frame_tensor = frame_tensor.permute(1, 2, 0)
        if frame_tensor.is_floating_point():
            frame_tensor = frame_tensor.clamp(-1, 1) * 0.5 + 0.5
        return frame_tensor.float().numpy()
    if isinstance(frame, np.ndarray):
        frame_array = frame
        if frame_array.ndim == 4 and frame_array.shape[0] == 1:
            frame_array = frame_array[0]
        if np.issubdtype(frame_array.dtype, np.integer):
            frame_array = frame_array.astype(np.float32) / 255.0
        return frame_array
    if isinstance(frame, PIL.Image.Image):
        return np.asarray(frame).astype(np.float32) / 255.0
    return frame


def ensure_frame_list(video_array: object) -> object:
    if isinstance(video_array, list):
        if len(video_array) == 0:
            return video_array
        first_item = video_array[0]
        if isinstance(first_item, np.ndarray):
            if first_item.ndim == 5:
                return list(first_item[0])
            if first_item.ndim == 4:
                return list(first_item)
            if first_item.ndim == 3:
                return video_array
        return video_array
    if isinstance(video_array, np.ndarray):
        if video_array.ndim == 5:
            return list(video_array[0])
        if video_array.ndim == 4:
            return list(video_array)
        if video_array.ndim == 3:
            return [video_array]
    return video_array


def export_video(frames: object, output_path: Path, fps: int) -> None:
    try:
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise ImportError("diffusers is required for export_to_video.") from exc

    if isinstance(frames, torch.Tensor):
        video_tensor = frames.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        frames = video_tensor.numpy()

    frames = ensure_frame_list(frames)
    if not isinstance(frames, list):
        raise ValueError("Expected frames to be a list after normalization.")

    normalized_frames = [normalize_frame(frame) for frame in frames]
    export_to_video(normalized_frames, output_video_path=str(output_path), fps=fps)


def main() -> None:
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    image = PIL.Image.open(args.image).convert("RGB")
    image = image.resize((args.width, args.height), PIL.Image.Resampling.LANCZOS)

    omni = Omni(
        model=args.model,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        enable_cpu_offload=args.enable_cpu_offload,
    )

    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Image: {args.image}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  Frames: {args.num_frames}")
    print(f"  Steps: {args.num_inference_steps}")
    if args.action_path is not None:
        print(f"  Action path: {args.action_path}")

    start = time.perf_counter()
    output = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "multi_modal_data": {"image": image},
        },
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_high,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            frame_rate=float(args.fps),
            extra_args={"action_path": args.action_path} if args.action_path is not None else {},
        ),
    )
    elapsed = time.perf_counter() - start
    print(f"Generation time: {elapsed:.2f}s")

    frames, _ = extract_frames(output)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_video(frames, output_path, args.fps)
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
