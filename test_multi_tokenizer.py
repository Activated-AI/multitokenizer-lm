import unittest
import torch
from multi_tokenizer import MultiTokenizer
import tokenizers

class TestMultiTokenizer(unittest.TestCase):
    def test_encode(self):
        test_input = (
            """[BEGIN_EN]One day, a little boy saw a cake. It was very big and had many strawberries.
    [BEGIN_ES]Un día, un niño pequeño vio un pastel. Era muy grande y tenía muchas fresas.
    [BEGIN_EN]Slightly more text [END]""")
        
        # Initialize tokenizers for English and Spanish
        english_tokenizer  = tokenizers.Tokenizer.from_file("expt_1/multi_8k/english_tokenizer.json")
        spanish_tokenizer = tokenizers.Tokenizer.from_file("expt_1/multi_8k/spanish_tokenizer.json")
        
        # Create a list of tokenizers
        tokeys = [english_tokenizer, spanish_tokenizer]
        
        # Initialize the MultiTokenizer
        multi_tokenizer = MultiTokenizer(tokeys)
        
        # Encode the test input
        encoded_tokens = multi_tokenizer.encode(test_input)        
        # Decode the tokens back to text
        decoded_text = multi_tokenizer.decode(encoded_tokens)
        assert decoded_text == test_input, "Decoded text does not match the input text"

if __name__ == '__main__':
    unittest.main()