# load vae models
import os
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder  # for vae support
from einops import rearrange
from datetime import datetime


class VideoAutoencoderKL(nn.Module):
    def __init__(
            self,
            from_pretrained=None,
            micro_batch_size=None,
            cache_dir=None,
            local_files_only=False,
            subfolder=None,
            scaling_factor=0.18215,
    ):
        super().__init__()

        print("Time: " + datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + " -- Initialize AutoencoderKL from pretrained model: {} in cache dir: {}.".format(
            from_pretrained, cache_dir))
        self.module = AutoencoderKL.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            subfolder=subfolder,
            # force_download=True,
        )

        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size
        self.scaling_factor = scaling_factor

    def encode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.sample().mul_(self.scaling_factor)
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i: i + bs]
                x_bs = self.module.encode(x_bs).latent_dist.sample().mul_(self.scaling_factor)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def decode(self, x, **kwargs):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            x = self.module.decode(x / self.scaling_factor).sample
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i: i + bs]
                x_bs = self.module.decode(x_bs / self.scaling_factor).sample
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            # assert (
            #     input_size[i] is None or input_size[i] % self.patch_size[i] == 0
            # ), "Input size must be divisible by patch size"
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
