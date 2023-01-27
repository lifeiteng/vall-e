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


from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from icefall.utils import AttributeDict

from valle.modules.embedding import SinePositionalEmbedding, TokenEmbedding

NUM_TEXT_TOKENS = 128
NUM_AUDIO_TOKENS = 1024 + 1  # EnCodec RVQ bins + 1


class VALLE(nn.Module):
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
        self.text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)
        self.text_position = SinePositionalEmbedding(d_model)

        self.audio_embedding = TokenEmbedding(d_model, NUM_AUDIO_TOKENS)
        self.audio_position = SinePositionalEmbedding(d_model)

        self.decoder_blocks = nn.TransformerDecoder(
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
        self.predict_layer = nn.Linear(d_model, NUM_AUDIO_TOKENS)

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

        # NOTE: There are two ways to implement the model
        #       1) standard TransformerDecoder, use x as memory
        #       2) modified TransformerDecoder like GPT-x(e.g. causal TransformerEncoder),
        #          use x as the prefix of decoder inputs
        # NOW: we try 1)

        total_loss = 0.0
        codes = y

        # AR Decoder
        if y is not None:  # Training

            def pad_y_eos(y, y_lens, eos_id):
                y_mask = make_pad_mask(y_lens).to(y.device)
                y_mask_int = y_mask.type(torch.int64)
                y = y.type(torch.int64)
                y = F.pad(y * (1 - y_mask_int), (0, 1)) + eos_id * F.pad(
                    y_mask_int, (0, 1), value=1
                )
                del y_mask_int
                # inputs, inputs_mask, targets
                return y[:, :-1], y_mask, y[:, 1:]

            y, y_mask, targets = pad_y_eos(
                y[..., 0], y_lens, eos_id=NUM_AUDIO_TOKENS - 1
            )
            y = self.audio_embedding(y)
            y = self.audio_position(y)

            y_len = y_lens.max()
            tgt_mask = torch.triu(
                torch.ones(y_len, y_len, device=y.device, dtype=torch.bool),
                diagonal=1,
            )
            y = self.decoder_blocks(
                y,
                x,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=y_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            logits = self.predict_layer(y)
            logits = logits.reshape([-1, NUM_AUDIO_TOKENS])
            codes = torch.multinomial(F.softmax(logits, dim=1), num_samples=1)

            # loss
            total_loss = F.cross_entropy(
                logits, targets.reshape([-1]), reduction="sum"
            )
        else:
            pass

        # Non-AR Decoders

        return (codes, total_loss)


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
    model = VALLE(params.decoder_dim, params.nhead, params.num_decoder_layers)
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
