from typing import Optional

import torch
from torch import Tensor, nn


class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            weight, bias = torch.split(
                self.project_layer(embedding),
                split_size_or_sections=self.d_model,
                dim=-1,
            )
            return (weight * self.norm(input) + bias, embedding)

        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(input) + bias


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model: int, *args, **kwargs) -> None:
        super(TransformerEncoderLayer, self).__init__(d_model, *args, **kwargs)
        self.norm1 = AdaptiveLayerNorm(d_model, self.norm1)
        self.norm2 = AdaptiveLayerNorm(d_model, self.norm2)
        # make why_not_sparsity_fast_path != ""
        self.norm1.eps += self.norm2.eps * 1.0001

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        # why_not_sparsity_fast_path = "TODO:"
        x, stage_embedding = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, stage_embedding), src_mask, src_key_padding_mask
            )
            x = x + self._ff_block(self.norm2(x, stage_embedding))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask),
                stage_embedding,
            )
            x = self.norm2(x + self._ff_block(x), stage_embedding)

        return (x, stage_embedding)


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        *args,
        **kwargs,
    ) -> None:
        super(TransformerDecoderLayer, self).__init__(d_model, *args, **kwargs)
        self.norm1 = AdaptiveLayerNorm(d_model, self.norm1)
        self.norm2 = AdaptiveLayerNorm(d_model, self.norm2)
        self.norm3 = AdaptiveLayerNorm(d_model, self.norm3)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x, stage_embedding = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, stage_embedding), tgt_mask, tgt_key_padding_mask
            )
            x = x + self._mha_block(
                self.norm2(x, stage_embedding),
                memory,
                memory_mask,
                memory_key_padding_mask,
            )
            x = x + self._ff_block(self.norm3(x, stage_embedding))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask),
                stage_embedding,
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask
                ),
                stage_embedding,
            )
            x = self.norm3(x + self._ff_block(x), stage_embedding)

        return (x, stage_embedding)
