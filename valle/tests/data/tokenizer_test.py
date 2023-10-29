# Copyright    2023                             (authors: Zhao Ming)
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


import unittest

from valle.data import TextTokenizer


class TestTextTokenizer(unittest.TestCase):
    def test_espeak(self):
        text_tokenizer = TextTokenizer(backend="google/mt5-small")

        for (_input, _target) in [
            ("The two parties, the sheep and the wolves, met each other.",
             ['▁', 'The', '_', 'two', '_', 'parties', ',', '_', 'the', '_', 's', 'heep', '_', 'and', '_', 'the', '_', 'wo', 'lves', ',', '_', 'met', '_', 'each', '_', 'other', '.', '</s>']),
            ("Mother! dear father! do you hear me?",
             ['▁', 'Mother', '!', '_', 'de', 'ar', '_', 'father', '!', '_', 'do', '_', 'you', '_', 'hear', '_', 'me', '?', '</s>']),
            ("\"Whoever thou art,\" She exclaimed, suddenly seizing Rodolfo's hand,",
             ['▁', '"', 'Who', 'ever', '_', 'thou', '_', 'art', ',"', '_', 'She', '_', 'ex', 'claim', 'ed', ',', '_', 's', 'udden', 'ly', '_', 'se', 'izing', '_', 'Rod', 'olfo', "'", 's', '_', 'hand', ',', '</s>'])
        ]:
            phonemized = text_tokenizer(_input)
            self.assertEqual(phonemized[0], _target)

    def test_pypinyin(self):
        text_tokenizer = TextTokenizer(backend="pypinyin")

        for (_input, _target) in [
            ("你好这是测试",
              ["ni3", '-', "hao3", '-', "zhe4", '-', "shi4", '-', "ce4", '-', "shi4"]),
            ("\"你好\", 这是测试.",
              ["\"", "ni3", '-', "hao3", "\"", ",", '_', "zhe4", '-', "shi4", '-', "ce4", '-', "shi4", "."]),
            ("此项 工作 还能 怎么 改进",
              ['ci3', '-', 'xiang4', '_', 'gong1', '-', 'zuo4', '_',
               'hai2', '-', 'neng2', '_', 'zen3', '-', 'me5', '_', 'gai3', '-', 'jin4']),  # AISHELL
        ]:
            phonemized = text_tokenizer(_input)
            self.assertEqual(phonemized[0], _target)

    def test_pypinyin_initials_finals(self):
        text_tokenizer = TextTokenizer(backend="pypinyin_initials_finals")

        for (_input, _target) in [
            ("你好这是测试",
              ["n", "i3", "-", "h", "ao3", "-", "zh", "e4", "-", "sh", "i4", "-", "c", "e4", "-", "sh", "i4"],
            ),
            ("\"你好.这是测试.",
              ["\"", "n", "i3", "-", "h", "ao3", ".", "zh", "e4", "-", "sh", "i4", "-", "c", "e4", "-", "sh", "i4", "."],
            ),
            ("\"你好. 这是测试.",
              ["\"", "n", "i3", "-", "h", "ao3", ".", "_", "zh", "e4", "-", "sh", "i4", "-", "c", "e4", "-", "sh", "i4", "."],
            ),
            ("此项 工作 还能 怎么 改进", ['c', 'i3', '-', 'x', 'iang4', '_', 'g', 'ong1', '-', 'z', 'uo4', '_',
                                'h', 'ai2', '-', 'n', 'eng2', '_', 'z', 'en3', '-', 'm', 'e5', '_',
                                'g', 'ai3', '-', 'j', 'in4']),  # AISHELL
        ]:
            phonemized = text_tokenizer(_input)
            self.assertListEqual(phonemized[0], _target)


if __name__ == "__main__":
    unittest.main()
