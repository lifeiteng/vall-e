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

import argparse
import random
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from icefall.utils import AttributeDict, make_pad_mask

from valle.modules.embedding import SinePositionalEmbedding, TokenEmbedding

NUM_TEXT_TOKENS = 128
NUM_AUDIO_TOKENS = 1024  # EnCodec RVQ bins


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
        decoder_cls: Union[
            nn.TransformerDecoder, nn.TransformerEncoder
        ] = nn.TransformerDecoder,
        decoder_layer_cls: Union[
            nn.TransformerDecoderLayer, nn.TransformerEncoderLayer
        ] = nn.TransformerDecoderLayer,
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
        self.text_position = SinePositionalEmbedding(d_model)

        self.audio_embeddings = nn.ModuleList(
            [TokenEmbedding(d_model, NUM_AUDIO_TOKENS + 1)]
            + [TokenEmbedding(d_model, NUM_AUDIO_TOKENS) for i in range(6)]
        )  # W_a
        self.audio_position = SinePositionalEmbedding(d_model)

        self.stage_embeddings = nn.ModuleList(
            [TokenEmbedding(d_model, 8) for k in range(8)]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                decoder_cls(
                    decoder_layer_cls(
                        d_model,
                        nhead,
                        dim_feedforward=d_model * 4,
                        dropout=0.1,
                        batch_first=True,
                        norm_first=True,
                    ),
                    num_layers=num_layers,
                )
                for i in range(8)
            ]
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

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: Union[torch.Tensor, None] = None,
        y_lens: Union[torch.Tensor, None] = None,
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
          Return the predicted audio code matrix and cross-entropy loss.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)

        x = self.text_embedding(x)
        x = self.text_position(x)

        total_loss = 0.0

        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)

        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))

        # Training
        # AR Decoder

        def pad_y_eos(y, y_lens, eos_id):
            y = F.pad(y, (0, 1)) + eos_id * F.pad(y_mask_int, (0, 1), value=1)
            # inputs, targets
            return y[:, :-1], y[:, 1:]

        y, targets = pad_y_eos(codes[..., 0], y_lens, eos_id=NUM_AUDIO_TOKENS)
        y_emb = self.audio_embeddings[0](y)
        y_pos = self.audio_position(y_emb)

        y_len = y_lens.max()
        tgt_mask = torch.triu(
            torch.ones(y_len, y_len, device=y.device, dtype=torch.bool),
            diagonal=1,
        )
        y_dec = self.decoder_blocks[0](
            y_pos,
            x,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=y_mask,
            memory_mask=None,
            memory_key_padding_mask=x_mask,
        )
        logits = self.predict_layers[0](y_dec)
        logits = logits.reshape([-1, NUM_AUDIO_TOKENS + 1])
        # loss
        total_loss = F.cross_entropy(
            logits, targets.reshape([-1]), reduction="sum"
        )
        # samples = [
        #     torch.multinomial(F.softmax(logits, dim=1), num_samples=1)
        # ]

        stop_idx = self.rng.choices(
            (1, 2, 3, 4, 5, 6, 7), weights=[1.0 / 7] * 7, k=1
        )[0]
        # Non-AR Decoders
        # TODO: Adaptive Layer Normalization
        for i, (decoder_block, predict_layer, embedding_layer) in enumerate(
            zip(
                self.decoder_blocks[1:],
                self.predict_layers[1:],
                self.audio_embeddings[1:] + [None],
            )
        ):
            y_dec = decoder_block(
                y_pos,
                x,
                tgt_mask=None,
                tgt_key_padding_mask=y_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            logits = predict_layer(y_dec)

            # loss
            targets = codes[..., i + 1] + NUM_AUDIO_TOKENS * y_mask_int
            total_loss += F.cross_entropy(
                logits.permute(0, 2, 1),
                targets,
                ignore_index=NUM_AUDIO_TOKENS,
                reduction="sum",
            )

            if i + 1 == stop_idx or i == 6:
                break

            # samples.append(
            #     torch.multinomial(F.softmax(logits.reshape([-1, NUM_AUDIO_TOKENS]), dim=1), num_samples=1)
            # )
            # Formula (4) (5)
            y_pos = y_pos + embedding_layer(codes[..., i + 1])

        return (codes, total_loss / (stop_idx + 1.0))

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
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
        Returns:
          Return the predicted audio code matrix and cross-entropy loss.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        x = self.text_embedding(x)
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
            y_pos = self.audio_position(y_emb)

            tgt_mask = torch.triu(
                torch.ones(y.shape[1], y.shape[1], device=y.device, dtype=torch.bool),
                diagonal=1,
            )

            y_dec = self.decoder_blocks[0](
                y_pos,
                x,
                tgt_mask=tgt_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            logits = self.predict_layers[0](y_dec[:, -1:])
            samples = torch.multinomial(
                F.softmax(logits.reshape([-1, NUM_AUDIO_TOKENS + 1]), dim=-1),
                num_samples=1,
            )
            if (
                samples[0, 0] == NUM_AUDIO_TOKENS
                or (y.shape[1] - prompts_len) > x_lens.max() * 20
            ):
                print(f"EOS [{prompts_len} -> {y.shape[1]}]")
                break

            y = torch.concat([y, samples], dim=1)

        codes = [y[:, prompts_len:]]
        # Non-AR Decoders
        # TODO: Adaptive Layer Normalization
        for i, (decoder_block, predict_layer, embedding_layer) in enumerate(
            zip(
                self.decoder_blocks[1:],
                self.predict_layers[1:],
                self.audio_embeddings[1:] + [None],
            )
        ):
            y_dec = decoder_block(
                y_pos,
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
            decoder_cls=nn.TransformerEncoder,
            decoder_layer_cls=nn.TransformerEncoderLayer,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: Union[torch.Tensor, None] = None,
        y_lens: Union[torch.Tensor, None] = None,
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
          Return the predicted audio code matrix and cross-entropy loss.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)

        x = self.text_embedding(x)
        x = self.text_position(x)

        total_loss = 0.0

        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)

        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)

        x_len = x_lens.max()
        y_len = y_lens.max()
        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool),
            (0, y_len),
            value=False,
        )
        y_attn_mask = F.pad(
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
            (x_len, 0),
            value=True,
        )
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0).to(
            y.device
        )

        # Training
        # AR Decoder
        def pad_y_eos(y, y_lens, eos_id):
            y = F.pad(y, (0, 1)) + eos_id * F.pad(y_mask_int, (0, 1), value=1)
            # inputs, targets
            return y[:, :-1], y[:, 1:]

        y, targets = pad_y_eos(codes[..., 0], y_lens, eos_id=NUM_AUDIO_TOKENS)
        y_emb = self.audio_embeddings[0](y)
        y_pos = self.audio_position(y_emb)

        xy_pos = torch.concat([x, y_pos], dim=1)

        xy_dec = self.decoder_blocks[0](
            xy_pos,
            mask=xy_attn_mask,
            src_key_padding_mask=xy_padding_mask,
            # is_causal=True,
        )
        logits = self.predict_layers[0](xy_dec[:, x_len:])
        logits = logits.reshape([-1, NUM_AUDIO_TOKENS + 1])
        # loss
        total_loss = F.cross_entropy(
            logits, targets.reshape([-1]), reduction="sum"
        )
        # samples = [
        #     torch.multinomial(F.softmax(logits, dim=1), num_samples=1)
        # ]

        stop_idx = self.rng.choices(
            (1, 2, 3, 4, 5, 6, 7), weights=[1.0 / 7] * 7, k=1
        )[0]
        # Non-AR Decoders
        # TODO: Adaptive Layer Normalization
        for i, (decoder_block, predict_layer, embedding_layer) in enumerate(
            zip(
                self.decoder_blocks[1:],
                self.predict_layers[1:],
                self.audio_embeddings[1:] + [None],
            )
        ):
            xy_dec = decoder_block(
                xy_pos,
                src_key_padding_mask=xy_padding_mask,
                # is_causal=False,
            )
            logits = predict_layer(xy_dec[:, x_len:])

            # loss
            targets = codes[..., i + 1] + NUM_AUDIO_TOKENS * y_mask_int
            total_loss += F.cross_entropy(
                logits.permute(0, 2, 1),
                targets,
                ignore_index=NUM_AUDIO_TOKENS,
                reduction="sum",
            )

            if i + 1 == stop_idx or i == 6:
                break

            # samples.append(
            #     torch.multinomial(F.softmax(logits.reshape([-1, NUM_AUDIO_TOKENS]), dim=1), num_samples=1)
            # )
            # Formula (4) (5)
            # xy_pos[:, x_len:] = xy_pos[:, x_len:] + embedding_layer(codes[..., i + 1])
            # xy_pos[:, x_len:] += embedding_layer(codes[..., i + 1])
            y_pos = y_pos + embedding_layer(codes[..., i + 1])
            xy_pos = torch.concat([x, y_pos], dim=1)

        return (codes, total_loss / (stop_idx + 1.0))

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
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
            y_pos = self.audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)

            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=False,
            )
            y_attn_mask = F.pad(
                torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
                (x_len, 0),
                value=True,
            )
            xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).to(
                y.device
            )

            xy_dec = self.decoder_blocks[0](
                xy_pos,
                mask=xy_attn_mask,
            )
            logits = self.predict_layers[0](xy_dec[:, -1:])
            samples = torch.multinomial(
                F.softmax(logits.reshape([-1, NUM_AUDIO_TOKENS + 1]), dim=-1),
                num_samples=1,
            )
            if (
                samples[0, 0] == NUM_AUDIO_TOKENS
                or (y.shape[1] - prompts.shape[1]) > x_lens.max() * 20
            ):
                print(f"EOS [{prompts.shape[1]} -> {y.shape[1]}]")
                break

            y = torch.concat([y, samples], dim=1)

        # for k in range(1, 7):
        #     xy_pos[:, x_lens.max() : prompts_len] += self.audio_embeddings[k](
        #         prompts[..., k]
        #     )

        codes = [y[:, prompts.shape[1] :]]
        # Non-AR Decoders
        # TODO: Adaptive Layer Normalization
        for i, (decoder_block, predict_layer, embedding_layer) in enumerate(
            zip(
                self.decoder_blocks[1:],
                self.predict_layers[1:],
                self.audio_embeddings[1:] + [None],
            )
        ):
            xy_dec = decoder_block(xy_pos)
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


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=1024,
        help="Embedding dimension in the decoder model.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=16,
        help="Number of attention heads in the Decoder layers.",
    )
    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=12,
        help="Number of Decoder layers.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="VALL-E",
        help="VALL-E or VALL-F.",
    )


def get_model(params: AttributeDict) -> nn.Module:
    if params.model_name.lower() in ["vall-f", "vallf"]:
        model = VALLF(
            params.decoder_dim, params.nhead, params.num_decoder_layers
        )
    else:
        assert params.model_name.lower() in ["vall-e", "valle"]
        model = VALLE(
            params.decoder_dim, params.nhead, params.num_decoder_layers
        )

    return model


if __name__ == "__main__":
    import numpy as np

    params = AttributeDict()
    params.decoder_dim = 64
    params.nhead = 16
    params.num_decoder_layers = 4

    x = torch.from_numpy(np.random.randint(0, 100, size=[4, 8]))
    x_lens = torch.from_numpy(np.random.randint(4, 8, size=[4]))
    x_lens[-1] = 8

    y = torch.from_numpy(np.random.randint(0, 1000, size=[4, 16, 8]))
    y_lens = torch.from_numpy(np.random.randint(8, 16, size=[4]))
    y_lens[-1] = 16

    # VALL-F
    params.model_name = "VALL-F"
    model = get_model(params)
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of {params.model_name} parameters: {num_param}")

    # Training
    codes, loss = model(x, x_lens, y, y_lens)

    # Inference
    codes = model.inference(x[-1:], x_lens[-1:], y[-1:])

    # VALL-E
    params.model_name = "VALL-E"
    model = get_model(params)
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of {params.model_name} parameters: {num_param}")

    # Training
    codes, loss = model(x, x_lens, y, y_lens)

    # Inference
    codes = model.inference(x[-1:], x_lens[-1:], y[-1:])

    print("model test PASS!")
