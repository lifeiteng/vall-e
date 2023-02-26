import argparse

import torch.nn as nn
from icefall.utils import AttributeDict, str2bool

from .transformer import Transformer
from .valle import NUM_MEL_BINS, VALLE, VALLF
from .visualizer import visualize


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-name",
        type=str,
        default="VALL-E",
        help="VALL-E, VALL-F or Transformer.",
    )
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
        "--norm-first",
        type=str2bool,
        default=True,
        help="Pre or Post Normalization.",
    )
    parser.add_argument(
        "--add-prenet",
        type=str2bool,
        default=False,
        help="Whether add PreNet after Inputs.",
    )


def get_model(params: AttributeDict) -> nn.Module:
    if params.model_name.lower() in ["vall-f", "vallf"]:
        model = VALLF(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            add_prenet=params.add_prenet,
        )
    elif params.model_name.lower() in ["vall-e", "valle"]:
        model = VALLE(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            add_prenet=params.add_prenet,
        )
    else:
        assert params.model_name in ["Transformer"]
        model = Transformer(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            add_prenet=params.add_prenet,
        )

    return model
