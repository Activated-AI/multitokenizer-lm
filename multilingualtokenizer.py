# %%
import re
import torch
from global_constants import special_tokens
lang_splitter_re = re.compile(r'(\[BEGIN_EN\]|\[BEGIN_ES\])')
import tokenizers


class MultiTokenizer:
    def __init__(self, tokenizers):
        self.tokenizers = tokenizers
    def encode(self, text):
        lang_code_to_lang_dense_id = {'[BEGIN_EN]': special_tokens.index("[BEGIN_EN]"), '[BEGIN_ES]': special_tokens.index("[BEGIN_ES]")}
        assert text.startswith('[BEGIN_EN]') or text.startswith('[BEGIN_ES]')
        # assert text.endswith(END_TOKEN)

        # This splits text into sequences of [lang_code_1, text_1, lang_code_2, text_2, ...]
        # where lang_code_i is either 'EN' or 'ES'
        per_lang_splits = [p for p in lang_splitter_re.split(text) if p.strip()]        
    
        lang_dense_id_per_text = [lang_code_to_lang_dense_id[code] for code in per_lang_splits[::2]]
        monolingual_texts = per_lang_splits[1::2]

        token_ids, lang_ids = [], []
        for lang_dense_id, monolingual_text in zip(lang_dense_id_per_text, monolingual_texts):
            tokenizer = self.tokenizers[lang_dense_id]
            
            these_tokens = [lang_dense_id] + tokenizer.encode(monolingual_text).ids 
            token_ids.extend(these_tokens)
            lang_ids.extend([lang_dense_id] * len(these_tokens))
        
        return torch.tensor(token_ids, dtype=torch.short), torch.tensor(lang_ids, dtype=torch.short)


test_input = (
    """[BEGIN_EN] [PROMPT] translate [USER] One day, a little boy saw a cake. It was very big and had many strawberries.
[BEGIN_ES] Un día, un niño pequeño vio un pastel. Era muy grande y tenía muchas fresas.
[BEGIN_EN] Slightly more text [END]""")



# %%
english_tokenizer  = tokenizers.ByteLevelBPETokenizer("expt_1/multitok_model_1/tiny-stories-Language.ENGLISH-bpe-vocab.json", "expt_1/multitok_model_1/tiny-stories-Language.ENGLISH-bpe-merges.txt")
spanish_tokenizer = tokenizers.ByteLevelBPETokenizer("expt_1/multitok_model_1/tiny-stories-Language.SPANISH-bpe-vocab.json", "expt_1/multitok_model_1/tiny-stories-Language.SPANISH-bpe-merges.txt")
# %%
tokeys = [english_tokenizer, spanish_tokenizer]

print(MultiTokenizer(tokeys).encode(test_input))

# %%
