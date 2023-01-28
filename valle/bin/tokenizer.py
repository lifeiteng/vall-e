#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
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
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 bin/tokenizer.py \
        --src_dir ./data/manifests --output_dir ./data/tokenized

"""
import argparse
import logging
import os
from pathlib import Path

import torch
from icefall.utils import get_executor
from lhotse import CutSet, NumpyHdf5Writer
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm.auto import tqdm

from valle.data import (
    AudioTokenConfig,
    AudioTokenExtractor,
    TextTokenizer,
    tokenize_text,
)
from valle.utils import SymbolTable

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Path to the manifest files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Path to the tokenized files",
    )
    parser.add_argument(
        "--dataset-parts",
        type=str,
        default="dev-clean test-clean",
        help="Space separated dataset parts",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="libritts",
        help="prefix of the manifest file",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="jsonl.gz",
        help="suffix of the manifest file",
    )

    return parser.parse_args()


def main():
    args = get_args()
    if args.dataset_parts == "all":
        dataset_parts = (
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        )
    else:
        dataset_parts = args.dataset_parts.replace("-p", "").split(" ")

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=args.src_dir,
        prefix=args.prefix,
        suffix=args.suffix,
    )
    assert len(manifests) == len(dataset_parts)

    text_tokenizer = TextTokenizer()

    # Fix RuntimeError: Cowardly refusing to serialize non-leaf tensor...
    # by remove encodec weight_norm
    num_jobs = min(16, os.cpu_count())

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    unique_symbols = set()
    extractor = AudioTokenExtractor(AudioTokenConfig())
    with get_executor() as ex:
        for partition, m in manifests.items():
            logging.info(f"Processing partition: {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )

            # Tokenize Audio
            storage_path = (
                f"{args.output_dir}/{args.prefix}_encodec_{partition}"
            )
            with torch.no_grad():
                cut_set = cut_set.compute_and_store_features(
                    extractor=extractor,
                    storage_path=storage_path,
                    num_jobs=num_jobs if ex is None else 64,
                    executor=ex,
                    storage_type=NumpyHdf5Writer,
                )

            # Tokenize Text
            for c in tqdm(cut_set):
                phonemes = tokenize_text(
                    text_tokenizer, text=c.supervisions[0].text
                )
                c.supervisions[0].custom["tokens"] = {"text": phonemes}
                unique_symbols.update(list(phonemes))

            cuts_filename = f"{args.prefix}_cuts_{partition}.{args.suffix}"
            cut_set.to_file(f"{args.output_dir}/{cuts_filename}")

    unique_phonemes = SymbolTable()
    for s in unique_symbols:
        unique_phonemes.add(s)
    logging.info(f"unique phonemes: {unique_symbols}")

    unique_phonemes_file = f"{args.output_dir}/unique_text_tokens.k2symbols"
    unique_phonemes.to_file(unique_phonemes_file)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
