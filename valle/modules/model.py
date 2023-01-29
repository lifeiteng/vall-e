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
from icefall.utils import AttributeDict

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
                nn.TransformerDecoder(
                    torch.nn.TransformerDecoderLayer(
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
        prompt_text_tokens: Union[torch.Tensor, None] = None,
        prompt_text_tokens_lens: Union[torch.Tensor, None] = None,
        prompt_audio_tokens: Union[torch.Tensor, None] = None,
        prompt_audio_tokens_lens: Union[torch.Tensor, None] = None,
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
          prompt_*：
            prompts will be used in Inference.
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

        # AR Decoder
        if y is not None:  # Training

            def pad_y_eos(y, y_lens, eos_id):
                y = F.pad(y, (0, 1)) + eos_id * F.pad(
                    y_mask_int, (0, 1), value=1
                )
                # inputs, targets
                return y[:, :-1], y[:, 1:]

            y, targets = pad_y_eos(
                codes[..., 0], y_lens, eos_id=NUM_AUDIO_TOKENS
            )
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
        else:
            pass

        stop_idx = self.rng.choices(
            (1, 2, 3, 4, 5, 6, 7), weights=[1.0 / 7] * 7, k=1
        )[0]
        # Non-AR Decoders
        # TODO: Adaptive Layer Normalization
        for i, (decoder_block, predict_layer, embedding_layer) in enumerate(
            zip(
                self.decoder_blocks[1:],
                self.predict_layers[1:],
                self.audio_embeddings,
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


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)

    expaned_lengths = torch.arange(max_len).expand(n, max_len).to(lengths)

    return expaned_lengths >= lengths.unsqueeze(1)


def get_valle_model(params: AttributeDict) -> nn.Module:
    model = VALLF(params.decoder_dim, params.nhead, params.num_decoder_layers)
    return model


if __name__ == "__main__":
    params = AttributeDict()
    params.decoder_dim = 64
    params.nhead = 16
    params.num_decoder_layers = 4
    model = get_valle_model(params)
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    import numpy as np

    # Training
    x = torch.from_numpy(np.random.randint(0, 100, size=[4, 8]))
    x_lens = torch.from_numpy(np.random.randint(4, 8, size=[4]))
    x_lens[-1] = 8

    y = torch.from_numpy(np.random.randint(0, 1000, size=[4, 16, 8]))
    y_lens = torch.from_numpy(np.random.randint(8, 16, size=[4]))
    y_lens[-1] = 16

    codes, loss = model(x, x_lens, y, y_lens)

    # Inference
    # TODO

    print("model test PASS!")
