import unittest
import torch
from data_split import ContextWindowAtomizer, partition_list_by_lang


import torch
import numpy as np

def assert_tensor_equal(actual, expected, rtol=1e-5, atol=1e-8, msg=None):
    """
    Assert that two tensors are equal in both shape and content.
    
    Args:
    actual (torch.Tensor): The tensor to test
    expected (torch.Tensor): The expected tensor
    rtol (float): Relative tolerance
    atol (float): Absolute tolerance
    msg (str): Optional message to use for the AssertionError
    
    Raises:
    AssertionError: If the tensors are not equal in shape or content
    """
    if actual.shape != expected.shape:
        raise AssertionError(f"{msg or ''}Shapes do not match: "
                             f"actual {actual.shape} != expected {expected.shape}\n"
                             f"Actual:\n{actual}\n"
                             f"Expected:\n{expected}")
    
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        # Convert tensors to numpy for easier comparison
        actual_np = actual.detach().cpu().numpy()
        expected_np = expected.detach().cpu().numpy()
        
        # Find indices where values differ
        diff_indices = np.where(~np.isclose(actual_np, expected_np, rtol=rtol, atol=atol))
        
        # Create a readable output of differing values
        diff_values = [
            f"Index {tuple(idx)}: actual {actual_np[idx]:.6f} != expected {expected_np[idx]:.6f}"
            for idx in zip(*diff_indices)
        ]
        
        error_msg = f"{msg or ''}Tensor contents do not match:\n" + "\n".join(diff_values[:10])
        if len(diff_values) > 10:
            error_msg += f"\n... and {len(diff_values) - 10} more differences"
        
        raise AssertionError(error_msg) from e

class TestContextWindowAtomizer(unittest.TestCase):
    def test_tensorize_shared(self):
        atomizer = ContextWindowAtomizer(3, retain_switch_tokens=False)
        toks = [0, 333, 222, 111, 1, 444, 555, 666]
        for lang, toks in partition_list_by_lang(toks):
            atomizer.add_tokenized_item(lang, toks)
        tensor = atomizer.tensorize()
        expected_tensor = torch.tensor([
            [0, 333, 222],
            [0, 111, 444],
            [1, 555, 666]
        ], dtype=torch.short)
        # torch.testing.assert_close(tensor, expected_tensor)
        assert_tensor_equal(tensor, expected_tensor)
    def test_tensorize_multitok(self):
        atomizer = ContextWindowAtomizer(3, retain_switch_tokens=True)
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
        assert_tensor_equal(tensor, expected_tensor)



if __name__ == '__main__':
    unittest.main()