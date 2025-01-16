import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageFilter
import uuid
from scipy.interpolate import interp1d, PchipInterpolator
import torchvision
from flowgen.models.controlnet import ControlNetModel
from scripts.utils import *

import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from flowgen.models.unet3d import UNet3DConditionModel as UNet3DConditionModelFlow
from animation.models.forward_unet import UNet3DConditionModel

from flowgen.pipelines.pipeline_flow_gen import FlowGenPipeline
from animation.pipelines.pipeline_animation import AnimationPipeline


from animation.utils.util import save_videos_grid
from animation.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
import math
from pathlib import Path

from PIL import Image

import numpy as np

import torch.nn as nn

output_dir = "outputs"
ensure_dirname(output_dir)

def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    # fx = interp1d(t, x, kind='cubic')
    # fy = interp1d(t, y, kind='cubic')
    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points


def visualize_drag_v2(background_image_path, brush_mask, splited_tracks, width, height):
    trajectory_maps = []

    background_image = Image.open(background_image_path).convert("RGBA")
    background_image = background_image.resize((width, height))
    w, h = background_image.size

    # Create a half-transparent background
    transparent_background = np.array(background_image)
    transparent_background[:, :, -1] = 128

    # Create a purple overlay layer
    purple_layer = np.zeros((h, w, 4), dtype=np.uint8)
    purple_layer[:, :, :3] = [128, 0, 128]  # Purple color
    purple_alpha = np.where(brush_mask < 0.5, 64, 0)  # Alpha values based on brush_mask
    purple_layer[:, :, 3] = purple_alpha

    # Convert to PIL image for alpha_composite
    purple_layer = Image.fromarray(purple_layer)
    transparent_background = Image.fromarray(transparent_background)

    # Blend the purple layer with the background
    transparent_background = Image.alpha_composite(transparent_background, purple_layer)

    # Create a transparent layer with the same size as the background image
    transparent_layer = np.zeros((h, w, 4))
    print(f"len(splited_tracks): {len(splited_tracks)}")
    for splited_track in splited_tracks:
        print(f"len(splited_track): {len(splited_track)}")
        if len(splited_track) > 1:
            splited_track = interpolate_trajectory(splited_track, 16)
            splited_track = splited_track[:16]
            for i in range(len(splited_track) - 1):
                start_point = (int(splited_track[i][0]), int(splited_track[i][1]))
                end_point = (int(splited_track[i + 1][0]), int(splited_track[i + 1][1]))
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                
                # 终止位置画箭头
                if i == len(splited_track) - 2:
                    cv2.arrowedLine(
                        transparent_layer,
                        start_point,
                        end_point,
                        (255, 0, 0, 192),
                        2,
                        tipLength=8 / arrow_length,
                    )
                # 其它位置画红线
                else:
                    cv2.line(
                        transparent_layer, start_point, end_point, (255, 0, 0, 192), 2
                    )
        else:
            cv2.circle(
                transparent_layer,
                (int(splited_track[0][0]), int(splited_track[0][1])),
                5,
                (255, 0, 0, 192),
                -1,
            )

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    trajectory_maps.append(trajectory_map)
    return trajectory_maps, transparent_layer


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
# 改成单例模式，保证不会重复调用
class Drag(metaclass=Singleton):
    def __init__(
        self,
        device,
        pretrained_model_path,
        inference_config,
        height,
        width,
        model_length,
    ):
        self.device = device

        inference_config = OmegaConf.load(inference_config)
        ### >>> create validation pipeline >>> ###
        print("start loading")
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        )
        # unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
        unet = UNet3DConditionModelFlow.from_config_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                inference_config.unet_additional_kwargs
            ),
        )
        vae_img = AutoencoderKL.from_pretrained(
            "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/Motion-I2V/models/stage2/StableDiffusion", subfolder="vae"
        )
        import json

        with open("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/Motion-I2V/models/stage1/StableDiffusion-FlowGen/vae/config.json", "r") as f:
            vae_config = json.load(f)
        vae = AutoencoderKL.from_config(vae_config)
        vae_pretrained_path = (
            "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/Motion-I2V/models/stage1/StableDiffusion-FlowGen/vae_flow/diffusion_pytorch_model.bin"
        )
        print("[Load vae weights from {}]".format(vae_pretrained_path))
        processed_ckpt = {}
        weight = torch.load(vae_pretrained_path, map_location="cpu")
        vae.load_state_dict(weight, strict=True)
        controlnet = ControlNetModel.from_unet(unet)
        unet.controlnet = controlnet
        unet.control_scale = 1.0

        unet_pretrained_path = (
            "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/Motion-I2V/models/stage1/StableDiffusion-FlowGen/unet/diffusion_pytorch_model.bin"
        )
        print("[Load unet weights from {}]".format(unet_pretrained_path))
        weight = torch.load(unet_pretrained_path, map_location="cpu")
        m, u = unet.load_state_dict(weight, strict=False)

        controlnet_pretrained_path = (
            "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/Motion-I2V/models/stage1/StableDiffusion-FlowGen/controlnet/controlnet.bin"
        )
        print("[Load controlnet weights from {}]".format(controlnet_pretrained_path))
        weight = torch.load(controlnet_pretrained_path, map_location="cpu")
        m, u = unet.load_state_dict(weight, strict=False)

        print("finish loading")
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            assert False
        pipeline = FlowGenPipeline(
            vae_img=vae_img,
            vae_flow=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DDIMScheduler(
                **OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            ),
        )  # .to("cuda")
        pipeline = pipeline.to("cuda")

        self.pipeline = pipeline
        self.height = height
        self.width = width
        self.ouput_prefix = f"flow_debug"
        self.model_length = model_length

        ### >>> create validation pipeline >>> ###
        pretrained_model_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/Motion-I2V/models/stage2/StableDiffusion"
        print("start loading")
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                inference_config.unet_additional_kwargs
            ),
        )
        # 3. text_model
        motion_module_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/Motion-I2V/models/stage2/Motion_Module/motion_block.bin"
        print("[Loading motion module ckpt from {}]".format(motion_module_path))
        weight = torch.load(motion_module_path, map_location="cpu")
        unet.load_state_dict(weight, strict=False)

        from safetensors import safe_open

        dreambooth_state_dict = {}
        with safe_open(
            "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/Motion-I2V/models/stage2/DreamBooth_LoRA/realisticVisionV51_v20Novae.safetensors",
            framework="pt",
            device="cpu",
        ) as f:
            for key in f.keys():
                dreambooth_state_dict[key] = f.get_tensor(key)

        from animation.utils.convert_from_ckpt import (
            convert_ldm_unet_checkpoint,
            convert_ldm_clip_checkpoint,
            convert_ldm_vae_checkpoint,
        )

        converted_vae_checkpoint = convert_ldm_vae_checkpoint(
            dreambooth_state_dict, vae.config
        )
        vae.load_state_dict(converted_vae_checkpoint)
        personalized_unet_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/Motion-I2V/models/stage2/DreamBooth_LoRA/realistic_unet.ckpt"
        print("[Loading personalized unet ckpt from {}]".format(personalized_unet_path))
        unet.load_state_dict(torch.load(personalized_unet_path), strict=False)

        print("finish loading")
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            assert False
        pipeline = AnimationPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DDIMScheduler(
                **OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            ),
        )  # .to("cuda")
        pipeline = pipeline.to("cuda")

        self.animate_pipeline = pipeline

    @torch.no_grad()
    def forward_sample(
        self,
        input_drag,
        mask_drag,
        brush_mask,
        input_first_frame,
        prompt,
        n_prompt,
        inference_steps,
        guidance_scale,
        outputs=dict(),
    ):
        device = self.device

        b, l, h, w, c = input_drag.size()
        # drag = torch.cat([torch.zeros_like(drag[:, 0]).unsqueeze(1), drag], dim=1)  # pad the first frame with zero flow
        drag = rearrange(input_drag, "b l h w c -> b c l h w")
        mask = rearrange(mask_drag, "b l h w c -> b c l h w")
        brush_mask = rearrange(brush_mask, "b l h w c -> b c l h w")

        sparse_flow = drag # input_drag就是输入tracking point处理得到的光流
        sparse_mask = mask # mask_drag 总之也跟输入mask有关
        print(f"sparse_flow.shape: {sparse_flow.shape}")
        print(f"sparse_mask.shape: {sparse_mask.shape}")

        sparse_flow = (sparse_flow - 1 / 2) * sparse_mask + 1 / 2

        flow_mask_latent = rearrange(
            self.pipeline.vae_flow.encode(
                rearrange(sparse_flow, "b c f h w -> (b f) c h w")
            ).latent_dist.sample(),
            "(b f) c h w -> b c f h w",
            f=l,
        )
        # flow_mask_latent = vae.encode(sparse_flow).latent_dist.sample()*0.18215
        sparse_mask = F.interpolate(sparse_mask, scale_factor=(1, 1 / 8, 1 / 8))
        control = torch.cat([flow_mask_latent, sparse_mask], dim=1) # flow_mask_latent就是
        # print(drag)
        stride = list(range(8, 121, 8))

        # （1）FlowGenPipeline
        # 有意思，它是先生成一个flow_pre，生成运动场或流场数据，这些数据描述了图像中像素的运动信息
        sample = self.pipeline(
            prompt,
            first_frame=input_first_frame.squeeze(0),
            control=control,
            stride=torch.tensor([stride]).cuda(),
            negative_prompt=n_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            width=w,
            height=h,
            video_length=len(stride),
        ).videos
        sample = (sample * 2 - 1).clamp(-1, 1)
        sample = sample * (1 - brush_mask.to(sample.device))
        sample[:, 0:1, ...] = sample[:, 0:1, ...] * w
        sample[:, 1:2, ...] = sample[:, 1:2, ...] * h

        flow_pre = sample.squeeze(0)
        flow_pre = rearrange(flow_pre, "c f h w -> f c h w")
        flow_pre = torch.cat(
            [torch.zeros(1, 2, h, w).to(flow_pre.device), flow_pre], dim=0
        )
        
        # （2）然后再生成AnimationPipeline
        # 利用生成的流场数据和初始帧，结合文本提示，生成最终的动画视频。
        sample = self.animate_pipeline(
            prompt,
            first_frame=input_first_frame.squeeze(0) * 2 - 1,
            flow_pre=flow_pre,
            brush_mask=brush_mask,
            negative_prompt=n_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            width=w,
            height=h,
            video_length=self.model_length,
        ).videos

        return sample

    def run(
        self,
        first_frame_path,
        image_brush,
        tracking_points,
        inference_batch_size,
        flow_unit_id,
        prompt,
    ):
        original_width, original_height = self.width, self.height
        

        brush_mask = image_brush["mask"]

        brush_mask = (
            cv2.resize(
                brush_mask[:, :, 0],
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            / 255.0
        )

        brush_mask_bool = brush_mask > 0.5
        brush_mask[brush_mask_bool], brush_mask[~brush_mask_bool] = 0, 1

        brush_mask = torch.from_numpy(brush_mask)

        brush_mask = (
            torch.zeros_like(brush_mask) if (brush_mask == 1).all() else brush_mask
        )

        brush_mask = brush_mask.unsqueeze(0).unsqueeze(3)

        input_all_points = tracking_points
        resized_all_points = [
            tuple(
                [
                    tuple(
                        [
                            int(e1[0] * self.width / original_width),
                            int(e1[1] * self.height / original_height),
                        ]
                    )
                    for e1 in e
                ]
            )
            for e in input_all_points
        ]

        input_drag = torch.zeros(self.model_length - 1, self.height, self.width, 2)
        mask_drag = torch.zeros(self.model_length - 1, self.height, self.width, 1)
        for splited_track in resized_all_points:
            if len(splited_track) == 1:  # stationary point
                displacement_point = tuple(
                    [splited_track[0][0] + 1, splited_track[0][1] + 1]
                )
                splited_track = tuple([splited_track[0], displacement_point])
            # interpolate the track
            splited_track = interpolate_trajectory(splited_track, self.model_length)
            splited_track = splited_track[: self.model_length]
            if len(splited_track) < self.model_length:
                splited_track = splited_track + [splited_track[-1]] * (
                    self.model_length - len(splited_track)
                )
            for i in range(self.model_length - 1):
                start_point = splited_track[0]
                end_point = splited_track[i + 1]
                input_drag[
                    i,
                    max(int(start_point[1]) - flow_unit_id, 0) : int(start_point[1])
                    + flow_unit_id,
                    max(int(start_point[0]) - flow_unit_id, 0) : int(
                        start_point[0] + flow_unit_id
                    ),
                    0,
                ] = (
                    end_point[0] - start_point[0]
                )
                input_drag[
                    i,
                    max(int(start_point[1]) - flow_unit_id, 0) : int(start_point[1])
                    + flow_unit_id,
                    max(int(start_point[0]) - flow_unit_id, 0) : int(
                        start_point[0] + flow_unit_id
                    ),
                    1,
                ] = (
                    end_point[1] - start_point[1]
                )
                mask_drag[
                    i,
                    max(int(start_point[1]) - flow_unit_id, 0) : int(start_point[1])
                    + flow_unit_id,
                    max(int(start_point[0]) - flow_unit_id, 0) : int(
                        start_point[0] + flow_unit_id
                    ),
                ] = 1

        print(f"input_drag.shape: {input_drag.shape}")
        print(f"mask_drag.shape: {mask_drag.shape}")
        
        input_drag[..., 0] /= self.width
        input_drag[..., 1] /= self.height

        input_drag = input_drag * (1 - brush_mask)
        mask_drag = torch.where(brush_mask.expand_as(mask_drag) > 0, 1, mask_drag)

        input_drag = (input_drag + 1) / 2
        dir, base, ext = split_filename(first_frame_path)
        id = base.split("_")[-1]
        
        id = first_frame_path.split("/")[-3] # 获取video_Id

        image_pil = image2pil(first_frame_path)
        image_pil = image_pil.resize((self.width, self.height), Image.BILINEAR).convert(
            "RGB"
        )

        visualized_drag, _ = visualize_drag_v2(
            first_frame_path,
            brush_mask.squeeze(3).squeeze(0).cpu().numpy(),
            resized_all_points,
            self.width,
            self.height,
        )
        outputs_path = f"visualized_drag/{id}.png"
        output_dir = os.path.dirname(outputs_path)
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        visualized_drag[0].save(outputs_path)
        

        first_frames_transform = transforms.Compose(
            [
                lambda x: Image.fromarray(x),
                transforms.Resize((320, 512)), # 直接裁剪到 320x512
                # transforms.Resize(320, interpolation=transforms.InterpolationMode.BILINEAR),  # 按最短边缩放到 320
                # transforms.CenterCrop((320, 512)),  # 中心裁剪到 320x512
                transforms.ToTensor(),
            ]
        )

        outputs = None
        ouput_video_list = []
        num_inference = 1
        for i in tqdm(range(num_inference)):
            if not outputs:

                first_frames = image2arr(first_frame_path)
                Image.fromarray(first_frames).save("./temp.png")
                first_frames = repeat(
                    first_frames_transform(first_frames),
                    "c h w -> b c h w",
                    b=inference_batch_size,
                ).to(self.device)
            else:
                first_frames = outputs[:, -1]

            # 输入模型
            outputs = self.forward_sample(
                repeat(
                    # 输入tracking points
                    input_drag[
                        i * (self.model_length - 1) : (i + 1) * (self.model_length - 1)
                    ],
                    "l h w c -> b l h w c",
                    b=inference_batch_size,
                ).to(self.device),
                repeat(
                    # 输入tracking point 和 mask的合并
                    mask_drag[
                        i * (self.model_length - 1) : (i + 1) * (self.model_length - 1)
                    ],
                    "l h w c -> b l h w c",
                    b=inference_batch_size,
                ).to(self.device),
                repeat(
                    # 输入mask
                    brush_mask[
                        i * (self.model_length - 1) : (i + 1) * (self.model_length - 1)
                    ],
                    "l h w c -> b l h w c",
                    b=inference_batch_size,
                ).to(self.device),
                first_frames,
                prompt,
                "(blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
                25,
                7,
            )
            ouput_video_list.append(outputs)

        outputs_path = f"outputs/{id}.mp4"
        save_videos_grid(outputs, outputs_path)

        return visualized_drag[0], outputs_path

def run_motioni2v_inference(image_path,height,width,model_length,traj,caption):

    # 1. 初始化 Drag
    device = "cuda"
    pretrained_model_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/Motion-I2V/models/stage1/StableDiffusion-FlowGen"
    inference_config = "configs/configs_flowgen/inference/inference.yaml"
    height, width = height,width
    # height, width = 1920, 1080
    model_length = model_length
    DragNUWA_net = Drag(device, pretrained_model_path, inference_config, height, width, model_length) # 单例模式 

    # 2. 输入
    first_frame_path = image_path
    tracking_points = traj.permute(1,0,2) # 把point_num放到第一才能对上
    prompt = caption
    brush_mask_array = np.zeros((height, width, 3), dtype=np.uint8)  #   如果完全不需要mask，可以给一张纯黑图( shape=[H,W,3] )即可
    image_brush = {
        "mask": brush_mask_array 
        # cv2.imread("your_brush_mask.png", cv2.IMREAD_UNCHANGED)
    }
  
    inference_batch_size = 1
    flow_unit_id = 64
    

    # 3. 开始推理
    visualized_drag_img, output_video_path = DragNUWA_net.run(
        first_frame_path,
        image_brush,
        tracking_points,
        inference_batch_size,
        flow_unit_id,
        prompt
    )
    print("Done! The result video is saved at:", output_video_path)    


if __name__ == "__main__":
    first_frame_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/project/Motion-I2V_evaluate/test_imgs/bear/00000.jpg"
    height, width = 320, 512
    model_length = 16
    tracking_points =  [
        [(100,100), (150,150)],   # 第一条 drag
        [(200,120), (220,150)],  # 第二条 drag
    ]    
    prompt = "A beer"
    run_motioni2v_inference(
        image_path=first_frame_path,
        height=height,
        width=width,
        model_length=model_length,
        traj=tracking_points,
        caption=prompt)