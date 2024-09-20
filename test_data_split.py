import unittest
import torch
from data_split import ContextWindowAtomizer, token_id_is_lang_id, partition_list_by_lang

class TestContextWindowAtomizer(unittest.TestCase):
    def test_tensorize_shared(self):
        atomizer = ContextWindowAtomizer(3, retain_switch_toks=False)
        toks = [0, 333, 222, 111, 1, 444, 555, 666]
        for lang, toks in partition_list_by_lang(toks):
            atomizer.add_tokenized_item(lang, toks)
        tensor = atomizer.tensorize()
        expected_tensor = torch.tensor([
            [0, 333, 222],
            [0, 111, 444],
            [1, 555, 666]
        ], dtype=torch.short)
        torch.testing.assert_close(tensor, expected_tensor)
    def test_tensorize_multitok(self):
        atomizer = ContextWindowAtomizer(3, retain_switch_toks=True)
        toks = [0, 333, 222, 111, 1, 444, 555, 666]
        for lang, toks in partition_list_by_lang(toks):
            atomizer.add_tokenized_item(lang, toks)
        tensor = atomizer.tensorize()
        expected_tensor = torch.tensor([
            [0, 0, 333],
            [0, 222, 111],
            [1, 1, 444],
            [1, 555, 666]
        ], dtype=torch.short)
        torch.testing.assert_close(tensor, expected_tensor)



if __name__ == '__main__':
    unittest.main()