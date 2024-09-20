import re
from global_constants import special_tokens, num_langs
import multiprocessing
import tokenizers
from itertools import groupby
from dataclasses import dataclass
from typing import List




# Regular expression to split text based on language tags
lang_splitter_re = re.compile(r'(\[BEGIN_EN\]|\[BEGIN_ES\])')

def partition_list_by_lang(lst):
    partitioned_sequence = [list(group) for _, group in groupby(lst, token_id_is_lang_id)]
    assert len(partitioned_sequence) % 2 == 0, "partitioned sequence must have an even number of elements"
    for singleton_lang_list, monolingual_tokens in  zip(partitioned_sequence[::2], partitioned_sequence[1::2]):
        assert len(singleton_lang_list) == 1, "singleton lang list must have exactly one element"
        assert token_id_is_lang_id(singleton_lang_list[0]), "singleton lang list must have a langid"
        assert not any(token_id_is_lang_id(t) for t in monolingual_tokens), "monolingual tokens must not be langids"
        yield singleton_lang_list[0], monolingual_tokens

def token_id_is_lang_id(token_id):
    return token_id < num_langs


@dataclass
class FakeTokens:
    ids: List[int]

class MultiTokenizer:
    def __init__(self, tokenizers):
        self.tokenizers = tokenizers

    def encode(self, text):
        # Mapping from language tags to their dense IDs
        assert text.startswith('[BEGIN_EN]') or text.startswith('[BEGIN_ES]')
        # Split the text into language segments
        per_lang_splits = [p for p in lang_splitter_re.split(text) if p.strip()]        
        
        # Extract language IDs and corresponding monolingual texts
        lang_dense_id_per_text = [special_tokens.index(code) for code in per_lang_splits[::2]]
        monolingual_texts = per_lang_splits[1::2]

        token_ids = []
        for lang_dense_id, monolingual_text in zip(lang_dense_id_per_text, monolingual_texts):
            tokenizer = self.tokenizers[lang_dense_id]
            
            # Prepend the language dense ID to the tokenized IDs
            these_tokens = [lang_dense_id] + tokenizer.encode(monolingual_text).ids 
            token_ids.extend(these_tokens)

        return token_ids
    
    def _wrapped_encode(self, args):
        return FakeTokens(self.encode(args))
    
    def encode_batch(self, texts):
        chunk_size = 25000
        for i in range(0, len(texts), chunk_size):
            with multiprocessing.Pool(24) as pool:
                yield from pool.map(self._wrapped_encode, texts[i:i+chunk_size])
        
                
    def decode(self, tokens, langs=None):
        ret = ""
        for lang, token_list in partition_list_by_lang(tokens):
            tokenizer = self.tokenizers[lang]
            ret += tokenizer.decode([lang]+token_list, skip_special_tokens=False)
        return ret

