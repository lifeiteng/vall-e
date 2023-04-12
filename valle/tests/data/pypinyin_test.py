# Copyright    2023                             (authors: Zhao Ming)
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


import unittest

from valle.data import TextTokenizer


class TestPypinyin(unittest.TestCase):
    def test_pypinyin_g2p(self):
        text_tokenizer = TextTokenizer(backend="pypinyin_g2p")
        input = "你好这是测试"
        output = ["ni3", "hao3", "zhe4", "shi4", "ce4", "shi4"]
        phonemes = text_tokenizer(input)
        assert phonemes == output

    def test_pypinyin_g2p_phone(self):
        text_tokenizer = TextTokenizer(backend="pypinyin_g2p_phone")
        input = "你好这是测试"
        output = [
            "n",
            "i3",
            "h",
            "ao3",
            "zh",
            "e4",
            "sh",
            "i4",
            "c",
            "e4",
            "sh",
            "i4",
        ]
        phonemes = text_tokenizer(input)
        assert phonemes == output


if __name__ == "__main__":
    unittest.main()
