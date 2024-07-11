from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward

from einops import rearrange, repeat
import math


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(in_channels, motion_module_type: str, motion_module_kwargs: dict):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(
            in_channels=in_channels,
            **motion_module_kwargs,
        )
    else:
        raise ValueError


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels
            // num_attention_heads
            // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(
                self.temporal_transformer.proj_out
            )

    def forward(
        self,
        input_tensor,
        temb,
        encoder_hidden_states,
        attention_mask=None,
        anchor_frame_idx=None,
        flow_pre=None,
    ):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(
            hidden_states, encoder_hidden_states, attention_mask, flow_pre=flow_pre
        )

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        flow_pre=None,
    ):
        assert (
            hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * weight, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                flow_pre=flow_pre,
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, weight, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=(
                        cross_attention_dim if block_name.endswith("_Cross") else None
                    ),
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    _query_dim=dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        flow_pre=None,
    ):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=(
                        encoder_hidden_states
                        if attention_block.is_cross_attention
                        else None
                    ),
                    video_length=video_length,
                    flow_pre=flow_pre,
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class VersatileAttention(CrossAttention):
    def __init__(
        self,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        _query_dim=-1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert (attention_mode == "Temporal") or (attention_mode == "DeformableBiased")

        if attention_mode == "DeformableBiased":
            self.offset_head = nn.Linear(_query_dim + 2, 2 * 5)
            self.init_offset()

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None

        self.pos_encoder = (
            PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.0,
                max_len=temporal_position_encoding_max_len,
            )
            if (
                temporal_position_encoding
                and (
                    attention_mode == "Temporal" or attention_mode == "DeformableBiased"
                )
            )
            else None
        )

    def init_offset(self):
        self.offset_head.weight.data.zero_()
        self.offset_head.bias.data.zero_()

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def coords_grid(self, batch, ht, wd, device):
        coords = torch.meshgrid(
            torch.arange(ht, device=device), torch.arange(wd, device=device)
        )
        coords = torch.stack(coords[::-1], dim=-1).float()
        return coords[None].repeat(batch, 1, 1, 1)

    def bilinear_sampler(self, img, coords, mode="bilinear"):
        """Wrapper for grid_sample, uses pixel coordinates"""
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        img = F.grid_sample(img, grid, align_corners=True)

        return img

    def sample_k_v(self, H, offsets, key, value, flow_pre):
        # print(offsets.shape, key.shape, value.shape, "!@#!@#")
        # torch.Size([1600, 16, 8]) torch.Size([1600, 16, 320]) torch.Size([1600, 16, 320])
        BHH, video_length, dim = key.shape
        _, _, num_offset = offsets.shape
        B = int(BHH // H // H)

        key = key[:, 0, :].reshape(B, H, H, dim).permute(0, 3, 1, 2)
        value = value[:, 0, :].reshape(B, H, H, dim).permute(0, 3, 1, 2)
        offsets = offsets.reshape(B, H, H, video_length, num_offset)
        offsets = torch.tanh(offsets) * (H / 4)
        offsets = (
            offsets.reshape(B, H, H, video_length, num_offset // 2, 2)
            .permute(0, 3, 4, 1, 2, 5)
            .reshape(B * video_length * num_offset // 2, H, H, 2)
        )  # [64, 40, 40, 2]

        flow_pre = (
            flow_pre.repeat(1, 1, num_offset // 2)
            .reshape(B, H, H, video_length, 2, num_offset // 2)
            .permute(0, 3, 5, 1, 2, 4)
            .reshape(B * video_length * num_offset // 2, H, H, 2)
        )

        coords = self.coords_grid(
            B * video_length * num_offset // 2, H, H, offsets.device
        )  # torch.Size([64, 40, 40, 2])
        coords = coords + offsets + flow_pre

        coords = coords.reshape(
            B, video_length * num_offset // 2, H * H, 2
        )  # [1, 64, 1600, 2]

        sampled_key = self.bilinear_sampler(key, coords)
        sampled_value = self.bilinear_sampler(value, coords)  # [1, 320, 64, 1600]

        sampled_key = sampled_key.reshape(
            B, dim, video_length, num_offset // 2, H * H
        ).permute(0, 2, 3, 4, 1)
        sampled_value = sampled_value.reshape(
            B, dim, video_length, num_offset // 2, H * H
        ).permute(
            0, 2, 3, 4, 1
        )  # [1, 16, 4, 1600, 320]

        return sampled_key, sampled_value

    def biased_attention(self, query, key, value):
        # torch.Size([12800, 16, 40]) torch.Size([1, 16, 4, 1600, 320]) torch.Size([1, 16, 4, 1600, 320])
        B, video_length, num_offset, HW, dim = key.shape

        if self.upcast_attention:
            query = query.float()
            key = key.float()

        query = query.reshape(B * HW * 8 * video_length, 1, dim // 8)  # [204800, 1, 40]

        key = key.reshape(B, video_length, num_offset, HW, 8, dim // 8).permute(
            0, 3, 4, 1, 2, 5
        )  # [1, 1600, 8, 16, 4, 40]
        key = key.reshape(
            B * HW * 8 * video_length, num_offset, dim // 8
        )  # [204800, 4, 40]

        value = value.reshape(B, video_length, num_offset, HW, 8, dim // 8).permute(
            0, 3, 4, 1, 2, 5
        )
        value = value.reshape(B * HW * 8 * video_length, num_offset, dim // 8)

        attention_scores = torch.baddbmm(
            torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        attention_probs = attention_probs.to(value.dtype)

        hidden_states = torch.bmm(attention_probs, value)  # [204800, 1, 40]

        hidden_states = hidden_states.reshape(B, HW, 8, video_length, dim // 8).permute(
            0, 3, 1, 2, 4
        )
        hidden_states = hidden_states.reshape(B * video_length, HW, dim)

        return hidden_states

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        flow_pre=None,
    ):

        if self.attention_mode == "Temporal":
            batch_size, sequence_length, _ = hidden_states.shape

            if self.attention_mode == "Temporal":
                d = hidden_states.shape[1]
                hidden_states = rearrange(
                    hidden_states, "(b f) d c -> (b d) f c", f=video_length
                )

                if self.pos_encoder is not None:
                    hidden_states = self.pos_encoder(hidden_states)

                encoder_hidden_states = (
                    repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
                    if encoder_hidden_states is not None
                    else encoder_hidden_states
                )
            else:
                raise NotImplementedError

            encoder_hidden_states = encoder_hidden_states

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)
            dim = query.shape[-1]
            query = self.reshape_heads_to_batch_dim(query)

            if self.added_kv_proj_dim is not None:
                raise NotImplementedError

            encoder_hidden_states = (
                encoder_hidden_states
                if encoder_hidden_states is not None
                else hidden_states
            )
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(
                        attention_mask, (0, target_length), value=0.0
                    )
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

            # attention, what we cannot get enough of
            if self._use_memory_efficient_attention_xformers:
                hidden_states = self._memory_efficient_attention_xformers(
                    query, key, value, attention_mask
                )
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
                if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                    hidden_states = self._attention(query, key, value, attention_mask)
                else:
                    hidden_states = self._sliced_attention(
                        query, key, value, sequence_length, dim, attention_mask
                    )

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if self.attention_mode == "Temporal":
                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

            return hidden_states

        elif self.attention_mode == "DeformableBiased":

            batch_size, sequence_length, dim = hidden_states.shape  # [16, 1600, 320]
            H = int(sequence_length**0.5)
            assert sequence_length == H**2

            B_flow, L_flow, _, H_flow, W_flow = flow_pre.shape
            flow_pre = F.interpolate(
                flow_pre.reshape(B_flow * L_flow, 2, H_flow, W_flow),
                size=(H, H),
                mode="bilinear",
            ) / (H_flow / H)
            flow_pre = (
                flow_pre.reshape(B_flow, L_flow, 2, H, H)
                .permute(0, 3, 4, 1, 2)
                .reshape(B_flow * H * H, L_flow, 2)
            )

            if self.attention_mode == "DeformableBiased":
                d = hidden_states.shape[1]
                hidden_states = rearrange(
                    hidden_states, "(b f) d c -> (b d) f c", f=video_length
                )

                if self.pos_encoder is not None:
                    hidden_states = self.pos_encoder(hidden_states)

                encoder_hidden_states = (
                    repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
                    if encoder_hidden_states is not None
                    else encoder_hidden_states
                )
            else:
                raise NotImplementedError

            encoder_hidden_states = encoder_hidden_states

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            # [1600, 16, 320]
            query = self.to_q(hidden_states)
            dim = query.shape[-1]

            # offsets = self.offset_head(hidden_states)
            offsets = self.offset_head(torch.cat([hidden_states, flow_pre], dim=-1))
            query = self.reshape_heads_to_batch_dim(query)  # [12800, 16, 40]

            if self.added_kv_proj_dim is not None:
                raise NotImplementedError

            encoder_hidden_states = (
                encoder_hidden_states
                if encoder_hidden_states is not None
                else hidden_states
            )
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)  # [1600, 16, 320]

            sampled_key, sampled_value = self.sample_k_v(
                H, offsets, key, value, flow_pre
            )  # [1, 16, 4, 1600, 320]

            # torch.Size([12800, 16, 40]) torch.Size([1, 16, 4, 1600, 320]) torch.Size([1, 16, 4, 1600, 320])

            hidden_states = self.biased_attention(query, sampled_key, sampled_value)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if self.attention_mode == "Temporal":
                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

            return hidden_states
