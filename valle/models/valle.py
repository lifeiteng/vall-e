# Copyright    2023                             (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from icefall.utils import make_pad_mask
from torchmetrics.classification import MulticlassAccuracy

from valle.modules.embedding import SinePositionalEmbedding, TokenEmbedding
from valle.modules.transformer import (
    AdaptiveLayerNorm,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

NUM_TEXT_TOKENS = 128
NUM_AUDIO_TOKENS = 1024  # EnCodec RVQ bins
NUM_MEL_BINS = 100  # BigVGAN bigvgan_24khz_100band


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)


# NOTE: There are two ways to implement the model
#       1) [VALL-F] standard TransformerDecoder, use x as memory
#       2) [VALL-E] modified TransformerDecoder like GPT-x(e.g. causal TransformerEncoder),
#          use x as the prefix of decoder inputs
class VALLF(nn.Module):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        decoder_cls: Union[
            nn.TransformerDecoder, nn.TransformerEncoder
        ] = nn.TransformerDecoder,
        decoder_layer_cls: Union[
            TransformerDecoderLayer, TransformerEncoderLayer
        ] = TransformerDecoderLayer,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super().__init__()
        self.text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)  # W_x
        self.audio_embeddings = nn.ModuleList(
            [TokenEmbedding(d_model, NUM_AUDIO_TOKENS + 1)]
            + [TokenEmbedding(d_model, NUM_AUDIO_TOKENS) for i in range(6)]
        )  # W_a

        # PreNet
        if add_prenet:
            self.text_prenet = nn.Sequential(
                Transpose(),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                Transpose(),
                nn.Linear(d_model, d_model),
            )

            self.audio_prenet = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, d_model),
            )
        else:
            self.text_prenet = nn.Identity()
            self.audio_prenet = nn.Identity()

        self.text_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
        )
        self.audio_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
        )

        self.stage_embeddings = nn.ModuleList(
            [TokenEmbedding(d_model, 1) for i in range(8)]
        )

        self.ar_decoder = decoder_cls(
            decoder_layer_cls(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=AdaptiveLayerNorm(d_model, norm=nn.LayerNorm(d_model))
            if norm_first
            else None,
        )
        self.nar_decoder = decoder_cls(
            decoder_layer_cls(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=AdaptiveLayerNorm(d_model, norm=nn.LayerNorm(d_model))
            if norm_first
            else None,
        )

        self.predict_layers = nn.ModuleList(
            [nn.Linear(d_model, NUM_AUDIO_TOKENS + 1)]
            + [nn.Linear(d_model, NUM_AUDIO_TOKENS) for i in range(7)]
        )

        # We share the parameters of the output projection layer with the parameters of the acoustic embedding Wa
        self.predict_layers[0].weight = self.audio_embeddings[0].weight
        # We also share the parameters of the acoustic embedding layer and the output prediction layer,
        # which means the weights of the j-th prediction layer are the same as the (j + 1)-th acoustic embedding layer.
        for j in range(1, 6):
            self.predict_layers[j].weight = self.audio_embeddings[j + 1].weight

        self.rng = random.Random(0)

        self.ar_accuracy_metric = MulticlassAccuracy(
            NUM_AUDIO_TOKENS + 1,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=NUM_AUDIO_TOKENS,
        )

        self.nar_accuracy_metric = MulticlassAccuracy(
            NUM_AUDIO_TOKENS + 1,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=NUM_AUDIO_TOKENS,
        )

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear)):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=1.0)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
        reduction: str = "sum",
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)

        x = self.text_embedding(x)
        x = self.text_prenet(x)
        x = self.text_position(x)

        total_loss, metrics = 0.0, {}

        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)

        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))

        # Training
        # AR Decoder
        def pad_y_eos(y, eos_id):
            y = F.pad(y, (0, 1), value=0) + eos_id * F.pad(
                y_mask_int, (0, 1), value=1
            )
            # inputs, targets
            return y[:, :-1], y[:, 1:]

        y, targets = pad_y_eos(codes[..., 0], eos_id=NUM_AUDIO_TOKENS)
        y_emb = self.audio_embeddings[0](y)
        y_emb = self.audio_prenet(y_emb)
        y_pos = self.audio_position(y_emb)

        y_len = y_lens.max()
        tgt_mask = torch.triu(
            torch.ones(y_len, y_len, device=y.device, dtype=torch.bool),
            diagonal=1,
        )
        y_dec, _ = self.ar_decoder(
            (y_pos, self.stage_embeddings[0].weight),
            x,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=y_mask,
            memory_mask=None,
            memory_key_padding_mask=x_mask,
        )
        logits = self.predict_layers[0](y_dec).permute(0, 2, 1)
        # loss
        total_loss = F.cross_entropy(logits, targets, reduction=reduction)
        metrics["ArTop10Accuracy"] = self.ar_accuracy_metric(
            logits.detach(), targets
        ).item() * y_lens.sum().type(torch.float32)

        # Non-AR Decoders
        train_stage = self.rng.choices(
            (1, 2, 3, 4, 5, 6, 7), weights=[1.0 / 7] * 7, k=1
        )[0]
        for i in range(0, train_stage - 1):
            # Formula (4) (5)
            y_pos = y_pos + self.audio_embeddings[i + 1](codes[..., i + 1])
        targets = codes[..., train_stage] + NUM_AUDIO_TOKENS * y_mask_int

        y_dec, _ = self.nar_decoder(
            (y_pos, self.stage_embeddings[train_stage].weight),
            x,
            tgt_mask=None,
            tgt_key_padding_mask=y_mask,
            memory_mask=None,
            memory_key_padding_mask=x_mask,
        )
        logits = self.predict_layers[train_stage](y_dec).permute(0, 2, 1)
        # loss
        total_loss += F.cross_entropy(
            logits,
            targets,
            ignore_index=NUM_AUDIO_TOKENS,
            reduction=reduction,
        )
        metrics["NarTop10Accuracy"] = (
            self.nar_accuracy_metric(
                F.pad(
                    logits.detach(),
                    (0, 0, 1, 0, 0, 0),
                    value=logits.min().cpu().item(),
                ),
                targets,
            ).item()
            * y_lens.sum().type(torch.float32)
        )

        return ((x, codes), total_loss / 2.0, metrics)

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        top_k: int = -100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
          Return the predicted audio code matrix and cross-entropy loss.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        x = self.text_embedding(x)
        x = self.text_prenet(x)
        x = self.text_position(x)
        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)

        prompts = y
        prompts_len = y.shape[1]

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        y = prompts[..., 0]
        while True:
            y_emb = self.audio_embeddings[0](y)
            y_emb = self.audio_prenet(y_emb)
            y_pos = self.audio_position(y_emb)

            tgt_mask = torch.triu(
                torch.ones(
                    y.shape[1], y.shape[1], device=y.device, dtype=torch.bool
                ),
                diagonal=1,
            )

            y_dec, _ = self.ar_decoder(
                (y_pos, self.stage_embeddings[0].weight),
                x,
                tgt_mask=tgt_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            logits = self.predict_layers[0](y_dec[:, -1])
            if top_k > 0:
                samples = topk_sampling(
                    logits, top_k=top_k, top_p=1.0, temperature=temperature
                )
            else:
                samples = torch.multinomial(
                    F.softmax(logits, dim=-1),
                    num_samples=1,
                )

            if (
                samples[0, 0] == NUM_AUDIO_TOKENS
                or (y.shape[1] - prompts_len) > x_lens.max() * 16
            ):
                print(f"VALL-F EOS [{prompts_len} -> {y.shape[1]}]")
                break

            y = torch.concat([y, samples], dim=1)

        codes = [y[:, prompts_len:]]
        # Non-AR Decoders
        for i, (predict_layer, embedding_layer) in enumerate(
            zip(
                self.predict_layers[1:],
                self.audio_embeddings[1:] + [None],
            )
        ):
            y_dec, _ = self.nar_decoder(
                (y_pos, self.stage_embeddings[i + 1].weight),
                x,
                tgt_mask=None,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            logits = predict_layer(y_dec[:, prompts_len:])

            samples = torch.argmax(logits, dim=-1)
            codes.append(samples)
            # Formula (4) (5)
            if i < 6:
                y_pos[:, :prompts_len] += embedding_layer(prompts[..., i + 1])
                y_pos[:, prompts_len:] += embedding_layer(samples)

        assert len(codes) == 8
        return torch.stack(codes, dim=-1)


class VALLE(VALLF):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super(VALLE, self).__init__(
            d_model,
            nhead,
            num_layers,
            norm_first=norm_first,
            add_prenet=add_prenet,
            decoder_cls=nn.TransformerEncoder,
            decoder_layer_cls=TransformerEncoderLayer,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
        reduction: str = "sum",
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)

        x = self.text_embedding(x)
        x = self.text_prenet(x)
        x = self.text_position(x)

        total_loss, metrics = 0.0, {}

        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)

        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))

        x_len = x_lens.max()
        y_len = y_lens.max()

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        # xy_padding_mask = F.pad(y_mask, (x_len, 0), value=False)

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)

        # Training
        # AR Decoder
        def pad_y_eos(y, eos_id):
            y = F.pad(y, (0, 1)) + eos_id * F.pad(y_mask_int, (0, 1), value=1)
            # inputs, targets
            return y[:, :-1], y[:, 1:]

        y, targets = pad_y_eos(codes[..., 0], eos_id=NUM_AUDIO_TOKENS)
        y_emb = self.audio_embeddings[0](y)
        y_emb = self.audio_prenet(y_emb)
        y_pos = self.audio_position(y_emb)

        xy_pos = torch.concat([x, y_pos], dim=1)

        xy_dec, _ = self.ar_decoder(
            (xy_pos, self.stage_embeddings[0].weight),
            mask=xy_attn_mask,
            src_key_padding_mask=xy_padding_mask,
            # is_causal=True,
        )
        logits = self.predict_layers[0](xy_dec[:, x_len:]).permute(0, 2, 1)
        # loss
        # total_loss = F.cross_entropy(logits, targets, reduction="none")
        # loss_mask = 1.0 - y_mask.type(torch.float32)
        # total_loss = torch.sum(total_loss * loss_mask)
        total_loss = F.cross_entropy(logits, targets, reduction=reduction)

        metrics["ArTop10Accuracy"] = self.ar_accuracy_metric(
            logits.detach(), targets
        ).item() * y_lens.sum().type(torch.float32)

        # Non-AR Decoders
        train_stage = self.rng.choices(
            (1, 2, 3, 4, 5, 6, 7), weights=[1.0 / 7] * 7, k=1
        )[0]
        for i in range(0, train_stage - 1):
            # Formula (4) (5)
            # xy_pos[:, x_len:] = xy_pos[:, x_len:] + embedding_layer(codes[..., i + 1])
            # xy_pos[:, x_len:] += embedding_layer(codes[..., i + 1])
            y_pos = y_pos + self.audio_embeddings[i + 1](codes[..., i + 1])
        xy_pos = torch.concat([x, y_pos], dim=1)
        targets = codes[..., train_stage] + NUM_AUDIO_TOKENS * y_mask_int

        xy_dec, _ = self.nar_decoder(
            (xy_pos, self.stage_embeddings[train_stage].weight),
            src_key_padding_mask=xy_padding_mask,
            # is_causal=False,
        )
        logits = self.predict_layers[train_stage](xy_dec[:, x_len:]).permute(
            0, 2, 1
        )

        # loss
        total_loss += F.cross_entropy(
            logits,
            targets,
            ignore_index=NUM_AUDIO_TOKENS,
            reduction=reduction,
        )
        metrics["NarTop10Accuracy"] = (
            self.nar_accuracy_metric(
                F.pad(
                    logits.detach(),
                    (0, 0, 1, 0, 0, 0),
                    value=logits.min().cpu().item(),
                ),
                targets,
            ).item()
            * y_lens.sum().type(torch.float32)
        )

        return ((x, codes), total_loss / 2.0, metrics)

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        top_k: int = -100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
          Return the predicted audio code matrix and cross-entropy loss.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        x = self.text_embedding(x)
        x = self.text_prenet(x)
        x = self.text_position(x)

        prompts = y
        prompts_len = x.shape[1] + y.shape[1]

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        y = prompts[..., 0]

        x_len = x_lens.max()
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        while True:
            y_emb = self.audio_embeddings[0](y)
            y_emb = self.audio_prenet(y_emb)
            y_pos = self.audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)

            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1
                ),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat(
                [x_attn_mask_pad, y_attn_mask], dim=0
            ).to(y.device)

            xy_dec, _ = self.ar_decoder(
                (xy_pos, self.stage_embeddings[0].weight),
                mask=xy_attn_mask,
            )
            logits = self.predict_layers[0](xy_dec[:, -1])
            if top_k > 0:
                samples = topk_sampling(
                    logits, top_k=top_k, top_p=1.0, temperature=temperature
                )
            else:
                samples = torch.multinomial(
                    F.softmax(logits, dim=-1),
                    num_samples=1,
                )

            if (
                samples[0, 0] == NUM_AUDIO_TOKENS
                or (y.shape[1] - prompts.shape[1]) > x_lens.max() * 16
            ):
                if prompts.shape[1] == y.shape[1]:
                    y = torch.concat([y, samples], dim=1)
                    y_emb = self.audio_embeddings[0](y)
                    y_pos = self.audio_position(y_emb)
                    xy_pos = torch.concat([x, y_pos], dim=1)

                print(f"VALL-E EOS [{prompts.shape[1]} -> {y.shape[1]}]")
                break

            y = torch.concat([y, samples], dim=1)

        # for k in range(1, 7):
        #     xy_pos[:, x_lens.max() : prompts_len] += self.audio_embeddings[k](
        #         prompts[..., k]
        #     )

        codes = [y[:, prompts.shape[1] :]]
        # Non-AR Decoders
        for i, (predict_layer, embedding_layer) in enumerate(
            zip(
                self.predict_layers[1:],
                self.audio_embeddings[1:] + [None],
            )
        ):
            xy_dec, _ = self.nar_decoder(
                (xy_pos, self.stage_embeddings[i + 1].weight)
            )
            logits = predict_layer(xy_dec[:, prompts_len:])

            samples = torch.argmax(logits, dim=-1)
            codes.append(samples)
            # Formula (4) (5)
            if i < 6:
                xy_pos[:, x_lens.max() : prompts_len] += embedding_layer(
                    prompts[..., i + 1]
                )
                xy_pos[:, prompts_len:] += embedding_layer(samples)

        assert len(codes) == 8
        return torch.stack(codes, dim=-1)


# https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token
