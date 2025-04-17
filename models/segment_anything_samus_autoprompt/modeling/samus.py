# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from turtle import shape
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .auto_prompt_encoder import Prompt_Embedding_Generator, make_prompt_from_mask, LayerNorm2d, MaskAttention
from einops import rearrange


class Samus(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.prompt_generator = Prompt_Embedding_Generator()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        for param in self.prompt_encoder.parameters():
          param.requires_grad = False
        for param in self.mask_decoder.parameters():
          param.requires_grad = False
        for param in self.image_encoder.parameters():
          param.requires_grad = False

        # for n, value in self.image_encoder.named_parameters():
        #   if "cnn_embed" not in n and "post_pos_embed" not in n and "Adapter" not in n and "2.attn.rel_pos" not in n and "5.attn.rel_pos" not in n and "8.attn.rel_pos" not in n and "11.attn.rel_pos" not in n and "upneck" not in n:
        #     value.requires_grad = False

        self.embed_dim = 256
        re = 4
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim//re, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(self.embed_dim//re),
            nn.GELU(),
            nn.Conv2d(self.embed_dim//re, self.embed_dim//re, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(self.embed_dim//re),
            nn.GELU(),
            nn.Conv2d(self.embed_dim//re, self.embed_dim//re, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(self.embed_dim//re),
            nn.GELU(),
            nn.Conv2d(self.embed_dim//re, self.embed_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(self.embed_dim),
            nn.GELU()        
        )


    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self, 
        imgs: torch.Tensor,
        pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
        bbox: torch.Tensor=None
    ) -> torch.Tensor:
        imge= self.image_encoder(imgs)
        bs = imge.shape[0]
        de = self.prompt_encoder.get_dense_embeddings(batchsize=bs)
        output_tokens = self.mask_decoder.get_tokens(batchsize=bs)
        new_imge, object_token, new_output_tokens = self.prompt_generator(img_embedding=imge, output_token=output_tokens)
        
        generate_de = self.feature_adapter(new_imge)
        #generate_de = new_imge

        # generate_de, masks1 = self.feature_adapter(new_imge)
        # masks1 = F.interpolate(masks1, (256, 256), mode="bilinear", align_corners=False)

        low_res_masks, _ = self.mask_decoder(
                    image_embeddings=new_imge,
                    image_pe=self.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=object_token,
                    dense_prompt_embeddings=generate_de, 
                    new_output_tokens = output_tokens,
                    multimask_output=False,
                    )
        masks = F.interpolate(low_res_masks, (256, 256), mode="bilinear", align_corners=False)
        outputs = {"low_res_logits": low_res_masks, "masks": masks, "low_res_logits1": low_res_masks, "masks1": masks}
        return outputs
      
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
