# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import cast

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.platforms import current_omni_platform

try:
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.utils.cam_utils import (
        compute_relative_poses,
        get_Ks_transformed,
        get_plucker_embeddings,
        interpolate_camera_poses,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
except ImportError as exc:
    raise ImportError(
        "Failed to import from dependency 'lingbot-world'. "
        "Run examples/offline_inference/image_to_video/download_lingbot_world.py "
        "or install the LingBot-World repository first."
    ) from exc

from .wan_model import WanModel


@dataclass(frozen=True)
class LingbotWorldConfig:
    text_len: int = 512
    t5_dtype: torch.dtype = torch.bfloat16
    param_dtype: torch.dtype = torch.bfloat16
    num_train_timesteps: int = 1000
    frame_num: int = 81
    sample_steps: int = 70
    sample_shift: float = 10.0
    sample_guide_scale: tuple[float, float] = (5.0, 5.0)
    boundary_ratio: float = 0.947
    sample_neg_prompt: str = (
        "画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
        "三条腿，背景人很多，倒着走，镜头晃动，画面闪烁，模糊，噪点，水印，签名，文字，变形，扭曲，"
        "液化，不合逻辑的结构，卡顿，PPT幻灯片感，过暗，欠曝，低对比度，霓虹灯光感，过度锐化，"
        "3D渲染感，人物，行人，游客，身体，皮肤，肢体，面部特征，汽车，电线"
    )
    t5_checkpoint: str = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer: str = "google/umt5-xxl"
    vae_checkpoint: str = "Wan2.1_VAE.pth"
    vae_stride: tuple[int, int, int] = (4, 8, 8)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    low_noise_checkpoint: str = "low_noise_model"
    high_noise_checkpoint: str = "high_noise_model"
    max_area: int = 480 * 832


DEFAULT_CONFIG = LingbotWorldConfig()


def _resolve_model_path(model: str) -> str:
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download remote LingBot-World checkpoints. "
            "Install it or pass a prepared local model directory."
        ) from exc
    return snapshot_download(repo_id=model)


def _infer_control_type(model_ref: str) -> str:
    model_ref = model_ref.lower()
    if "act" in model_ref:
        return "act"
    return "cam"


class LingbotWorldPipeline(nn.Module, SupportImageInput):
    support_image_input = True
    color_format = "RGB"

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = torch.device(current_omni_platform.get_torch_device())
        self.config = DEFAULT_CONFIG
        self.model_path = _resolve_model_path(cast(str, od_config.model))
        self.control_type = _infer_control_type(self.model_path)
        self.enable_cpu_offload = bool(self.od_config.enable_cpu_offload)

        self.text_model = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=self.device,
            checkpoint_path=os.path.join(self.model_path, self.config.t5_checkpoint),
            tokenizer_path=os.path.join(self.model_path, self.config.t5_tokenizer),
        )
        self.text_encoder = self.text_model.model
        self.vae_model = Wan2_1_VAE(
            vae_pth=os.path.join(self.model_path, self.config.vae_checkpoint),
            device=self.device,
        )
        self.vae = self.vae_model.model
        self.low_noise_model = (
            WanModel.from_pretrained(
                self.model_path,
                subfolder=self.config.low_noise_checkpoint,
                torch_dtype=self.config.param_dtype,
                control_type=self.control_type,
            )
            .eval()
            .requires_grad_(False)
        )
        self.transformer = self.low_noise_model
        self.high_noise_model = (
            WanModel.from_pretrained(
                self.model_path,
                subfolder=self.config.high_noise_checkpoint,
                torch_dtype=self.config.param_dtype,
                control_type=self.control_type,
            )
            .eval()
            .requires_grad_(False)
        )
        self.transformer_2 = self.high_noise_model

        if self.enable_cpu_offload:
            self.text_encoder.to("cpu")
            self.transformer.to("cpu")
            self.transformer_2.to("cpu")
            self.vae.to("cpu")
        else:
            self.text_encoder.to(self.device)
            self.transformer.to(self.device)
            self.transformer_2.to(self.device)
            self.vae.to(self.device)

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            shift=1.0,
            use_dynamic_shifting=False,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=8)

    def load_weights(self, weights):
        pass

    def _prepare_model_for_timestep(self, timestep: torch.Tensor, boundary: float) -> nn.Module:
        if timestep.item() >= boundary:
            required_model = self.high_noise_model
            other_model = self.low_noise_model
        else:
            required_model = self.low_noise_model
            other_model = self.high_noise_model

        if self.enable_cpu_offload:
            if next(other_model.parameters()).device.type == self.device.type:
                other_model.to("cpu")
            if next(required_model.parameters()).device.type == "cpu":
                required_model.to(self.device)

        return required_model

    def _encode_prompt(self, prompt: str, negative_prompt: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        context = self.text_model([prompt], self.device)
        context_null = self.text_model([negative_prompt], self.device)
        return [context[0]], [context_null[0]]

    def _get_default_size(self, image: PIL.Image.Image) -> tuple[int, int]:
        aspect_ratio = image.height / image.width
        mod_value = 16
        height = round(np.sqrt(self.config.max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(self.config.max_area / aspect_ratio)) // mod_value * mod_value
        return height, width

    def _load_control_inputs(self, action_path: str | None, num_frames: int) -> tuple[dict | None, int]:
        if not action_path:
            return None, num_frames

        poses_path = os.path.join(action_path, "poses.npy")
        intrinsics_path = os.path.join(action_path, "intrinsics.npy")
        if not os.path.exists(poses_path):
            raise FileNotFoundError(f"LingBot-World control bundle is missing poses.npy: {poses_path}")
        if not os.path.exists(intrinsics_path):
            raise FileNotFoundError(f"LingBot-World control bundle is missing intrinsics.npy: {intrinsics_path}")

        c2ws = np.load(poses_path)
        len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
        num_frames = min(num_frames, len_c2ws)
        c2ws = c2ws[:num_frames]

        wasd_action = None
        if self.control_type == "act":
            action_npy = os.path.join(action_path, "action.npy")
            if not os.path.exists(action_npy):
                raise FileNotFoundError(f"LingBot-World act control bundle is missing action.npy: {action_npy}")
            wasd_action = np.load(action_npy)[:num_frames]

        return {"path": action_path, "c2ws": c2ws, "wasd_action": wasd_action}, num_frames

    def _build_dit_cond_dict(
        self,
        control_inputs: dict | None,
        *,
        lat_f: int,
        lat_h: int,
        lat_w: int,
        resized_height: int,
        resized_width: int,
    ) -> dict | None:
        if control_inputs is None:
            return None

        action_path = cast(str, control_inputs["path"])
        c2ws = cast(np.ndarray, control_inputs["c2ws"])
        wasd_action = cast(np.ndarray | None, control_inputs["wasd_action"])

        Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
        Ks = get_Ks_transformed(
            Ks,
            height_org=480,
            width_org=832,
            height_resize=resized_height,
            width_resize=resized_width,
            height_final=resized_height,
            width_final=resized_width,
        )
        Ks = Ks[0]

        len_c2ws = len(c2ws)
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1)

        c2ws_infer = c2ws_infer.to(self.device)
        Ks = Ks.to(self.device)
        wasd_action_tensor = None
        if wasd_action is not None:
            wasd_action_tensor = torch.from_numpy(wasd_action[::4]).float().to(self.device)

        only_rays_d = wasd_action_tensor is not None
        c2ws_plucker_emb = get_plucker_embeddings(
            c2ws_infer,
            Ks,
            resized_height,
            resized_width,
            only_rays_d=only_rays_d,
        )
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(resized_height // lat_h),
            c2=int(resized_width // lat_w),
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            "b (f h w) c -> b c f h w",
            f=lat_f,
            h=lat_h,
            w=lat_w,
        ).to(self.config.param_dtype)

        if wasd_action_tensor is not None:
            wasd_action_tensor = wasd_action_tensor[:, None, None, :].repeat(1, resized_height, resized_width, 1)
            wasd_action_tensor = rearrange(
                wasd_action_tensor,
                "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                c1=int(resized_height // lat_h),
                c2=int(resized_width // lat_w),
            )
            wasd_action_tensor = wasd_action_tensor[None, ...]
            wasd_action_tensor = rearrange(
                wasd_action_tensor,
                "b (f h w) c -> b c f h w",
                f=lat_f,
                h=lat_h,
                w=lat_w,
            ).to(self.config.param_dtype)
            c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_action_tensor], dim=1)

        return {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("LingBot-World currently supports exactly one prompt per request.")
        if req.sampling_params.num_outputs_per_prompt != 1:
            raise ValueError("LingBot-World currently supports num_outputs_per_prompt=1 only.")

        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            prompt = prompt_data
            negative_prompt = self.config.sample_neg_prompt
            multi_modal_data = {}
        else:
            prompt = cast(str, prompt_data.get("prompt"))
            negative_prompt = cast(str | None, prompt_data.get("negative_prompt")) or self.config.sample_neg_prompt
            multi_modal_data = cast(dict, prompt_data.get("multi_modal_data") or {})

        raw_image = multi_modal_data.get("image")
        if raw_image is None:
            raise ValueError("LingBot-World requires an input image.")
        if isinstance(raw_image, list):
            if len(raw_image) == 0:
                raise ValueError("LingBot-World received an empty image list.")
            raw_image = raw_image[0]
        if isinstance(raw_image, str):
            image = PIL.Image.open(raw_image).convert("RGB")
        else:
            image = cast(PIL.Image.Image, raw_image)

        height = req.sampling_params.height
        width = req.sampling_params.width
        if height is None or width is None:
            default_height, default_width = self._get_default_size(image)
            height = default_height if height is None else height
            width = default_width if width is None else width
        image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)

        if req.sampling_params.guidance_scale_provided:
            guidance_low = req.sampling_params.guidance_scale
        else:
            guidance_low = self.config.sample_guide_scale[0]
        guidance_high = req.sampling_params.guidance_scale_2
        if guidance_high is None:
            guidance_high = self.config.sample_guide_scale[1]

        num_frames = req.sampling_params.num_frames or self.config.frame_num
        action_path = cast(str | None, req.sampling_params.extra_args.get("action_path"))
        control_inputs, num_frames = self._load_control_inputs(action_path, num_frames)
        num_steps = req.sampling_params.num_inference_steps or self.config.sample_steps
        boundary_ratio = req.sampling_params.boundary_ratio
        if boundary_ratio is None:
            boundary_ratio = self.config.boundary_ratio
        flow_shift = cast(float | None, req.sampling_params.extra_args.get("flow_shift"))
        if flow_shift is None:
            flow_shift = self.config.sample_shift

        img = TF.to_tensor(image).sub_(0.5).div_(0.5).to(self.device)

        aspect_ratio = img.shape[1] / img.shape[2]
        lat_h = round(
            np.sqrt(self.config.max_area * aspect_ratio)
            // self.config.vae_stride[1]
            // self.config.patch_size[1]
            * self.config.patch_size[1]
        )
        lat_w = round(
            np.sqrt(self.config.max_area / aspect_ratio)
            // self.config.vae_stride[2]
            // self.config.patch_size[2]
            * self.config.patch_size[2]
        )
        resized_height = lat_h * self.config.vae_stride[1]
        resized_width = lat_w * self.config.vae_stride[2]

        if num_frames % self.config.vae_stride[0] != 1:
            num_frames = (num_frames - 1) // self.config.vae_stride[0] * self.config.vae_stride[0] + 1
        lat_f = (num_frames - 1) // self.config.vae_stride[0] + 1
        max_seq_len = lat_f * lat_h * lat_w // (self.config.patch_size[1] * self.config.patch_size[2])

        generator = req.sampling_params.generator
        if generator is None:
            seed = req.sampling_params.seed if req.sampling_params.seed is not None else random.randint(0, 2**31 - 1)
            generator = torch.Generator(device=self.device).manual_seed(seed)

        noise = torch.randn(
            16,
            lat_f,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=generator,
            device=self.device,
        )

        msk = torch.ones(1, num_frames, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        context, context_null = self._encode_prompt(prompt, negative_prompt)

        vae_input = torch.concat(
            [
                torch.nn.functional.interpolate(
                    img[None].cpu(),
                    size=(resized_height, resized_width),
                    mode="bicubic",
                ).transpose(0, 1),
                torch.zeros(3, num_frames - 1, resized_height, resized_width),
            ],
            dim=1,
        ).to(self.device)
        y = self.vae_model.encode([vae_input])[0]
        y = torch.concat([msk, y])

        dit_cond_dict = self._build_dit_cond_dict(
            control_inputs,
            lat_f=lat_f,
            lat_h=lat_h,
            lat_w=lat_w,
            resized_height=resized_height,
            resized_width=resized_width,
        )

        self.scheduler.set_timesteps(num_steps, device=self.device, shift=flow_shift)
        timesteps = self.scheduler.timesteps
        boundary = boundary_ratio * self.config.num_train_timesteps

        arg_c = {"context": context, "seq_len": max_seq_len, "y": [y], "dit_cond_dict": dit_cond_dict}
        arg_null = {"context": context_null, "seq_len": max_seq_len, "y": [y], "dit_cond_dict": dit_cond_dict}

        autocast_enabled = not self.od_config.disable_autocast and self.device.type == "cuda"
        autocast_context = (
            torch.amp.autocast(self.device.type, dtype=self.config.param_dtype)
            if autocast_enabled
            else torch.autocast("cpu", enabled=False)
        )

        latent = noise
        with autocast_context:
            with torch.no_grad():
                for timestep in timesteps:
                    latent_model_input = [latent.to(self.device)]
                    timestep_tensor = torch.stack([timestep]).to(self.device)
                    model = self._prepare_model_for_timestep(timestep, boundary)
                    current_guidance = guidance_high if timestep.item() >= boundary else guidance_low

                    noise_pred_cond = model(latent_model_input, t=timestep_tensor, **arg_c)[0]
                    noise_pred_uncond = model(latent_model_input, t=timestep_tensor, **arg_null)[0]
                    noise_pred = noise_pred_uncond + current_guidance * (noise_pred_cond - noise_pred_uncond)

                    latent = self.scheduler.step(
                        noise_pred.unsqueeze(0),
                        timestep,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=generator,
                    )[0].squeeze(0)

        videos = self.vae_model.decode([latent])
        output = videos[0].unsqueeze(0) if videos[0].ndim == 4 else videos[0]
        output_type = req.sampling_params.output_type or "np"
        if output_type != "latent":
            output = self.video_processor.postprocess_video(output, output_type=output_type)
        return DiffusionOutput(output=output)
