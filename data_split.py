EXPT_NAME = 'expt_1'
TRAIN_FRACTION = 0.9

import datasets
import os
import torch
import tokenizers

from global_constants import special_tokens, num_langs
from multi_tokenizer import MultiTokenizer
from tqdm import tqdm
from multi_tokenizer import partition_list_by_lang, token_id_is_lang_id


def write_english_story_tinyprompt_str(text):
    return f'[BEGIN_EN][PROMPT_EN_STORY]{text}[END]'

def write_english_story_tinyprompt(story_dict):
    return write_english_story_tinyprompt_str(story_dict['text'].strip())

def write_spanish_story_tinyprompt_str(text):
    return f'[BEGIN_ES][PROMPT_ES_STORY]{text}[END]'

def write_spanish_story_tinyprompt(story_dict):
    return write_spanish_story_tinyprompt_str(story_dict["story"].strip())

def paragraph_splitter(text):
    return [t.strip() for t in text.split('\n') if t.strip()]

def write_translation_story_tinyprompt_strs(spanish_text, english_text):
    spanish_paragraphs = paragraph_splitter(spanish_text)
    translation_paragraphs = paragraph_splitter(english_text)

    if len(spanish_paragraphs) != len(translation_paragraphs):        
        return ''
    
    alternating_paragraphs = ''.join(
        '[BEGIN_EN]' + a + '\n' + '[BEGIN_ES]' + b + '\n' for a, b in zip(translation_paragraphs, spanish_paragraphs))
    
    alternating_paragraphs = alternating_paragraphs.removeprefix('[BEGIN_EN]')  # Remove the first [BEGIN_EN] (it's redundant bc there wasn't a language switch)

    return f'[BEGIN_EN][PROMPT_EN_ES_TRANS]{alternating_paragraphs.strip()}[END]'

def write_translation_story_tinyprompt(story_dict):
    spanish_story = story_dict['story']
    translation = story_dict['translation']
    return write_translation_story_tinyprompt_strs(spanish_story, translation)


class TinyStoriesDataset:
    def __init__(self, items, formatter):        
        self.items = items
        self.formatted_items = [formatted for item in items if (formatted := formatter(item))]
        self.text = '\n'.join(self.formatted_items)


def build_structured_datasets(smoke_test=False):

    tinystories_ds_es_translated = datasets.load_dataset("robrenaud/multilingual_tinystories", data_files=["stories_00.json"])
    if smoke_test:
        

        tinystories_ds_en = datasets.load_dataset("roneneldan/TinyStories", data_files={"train": "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
                                                                                        "validation": "data/train-00000-of-00004-2d5a1467fff1081b.parquet"})

        tinystories_ds_es = datasets.load_dataset("robrenaud/multilingual_tinystories", data_files=["stories_01.json"])
    else:
        tinystories_ds_en = datasets.load_dataset("roneneldan/TinyStories")
        raw_es_stories = [f'stories_{i:02d}.json' for i in range(1, 22)]
        tinystories_ds_es = datasets.load_dataset("robrenaud/multilingual_tinystories", data_files=raw_es_stories)

    print('num translations   ', len(tinystories_ds_es_translated['train']))
    print('num spanish stories', len(tinystories_ds_es['train']))
    print('num english stories', len(tinystories_ds_en['train']))

    english_train  = tinystories_ds_en['train'].to_list()
    english_test    = tinystories_ds_en['validation'].to_list()

    spanish_full  = tinystories_ds_es['train'].to_list()
    spanish_split_index = int(len(spanish_full) * TRAIN_FRACTION)
    spanish_train = spanish_full[:spanish_split_index]
    spanish_test  = spanish_full[spanish_split_index:]

    translation_full = tinystories_ds_es_translated['train'].to_list()
    translation_split_index = int(len(translation_full) * TRAIN_FRACTION)
    translation_train = translation_full[:translation_split_index]
    translation_test  = translation_full[translation_split_index:]

    english_train_ds = TinyStoriesDataset(english_train, write_english_story_tinyprompt)
    english_test_ds = TinyStoriesDataset(english_test, write_english_story_tinyprompt)

    spanish_train_ds = TinyStoriesDataset(spanish_train, write_spanish_story_tinyprompt)
    spanish_test_ds = TinyStoriesDataset(spanish_test, write_spanish_story_tinyprompt)

    translation_train_ds = TinyStoriesDataset(translation_train, write_translation_story_tinyprompt)
    translation_test_ds = TinyStoriesDataset(translation_test, write_translation_story_tinyprompt)

    return {
        'english_train': english_train_ds,
        'english_test': english_test_ds,
        'spanish_train': spanish_train_ds,
        'spanish_test': spanish_test_ds,
        'translation_train': translation_train_ds,
        'translation_test': translation_test_ds
    }


def save_tokenizer(tokenizer, tokenizer_prefix):
    dir_name = os.path.dirname(tokenizer_prefix)
    base_name = os.path.basename(tokenizer_prefix)
    os.makedirs(os.path.dirname(tokenizer_prefix), exist_ok=True)
    tokenizer.save(tokenizer_prefix)


def train_tokenizer(datasets, vocab_size):
    # deleting the /tmp/data.txt file is punishing, since later crashes mean
    # we have to redo this work.  if we wrote a permanent file to a reasonable loc, we could skip this step once it succeeded.
    concat_text = ""
    for dataset in datasets:
        concat_text += dataset.text + '\n'
    location = "/tmp/data.txt"
    with open(location, 'w') as f:
        f.write(concat_text)
    tokenizer = tokenizers.ByteLevelBPETokenizer()
    tokenizer.train(files=[location], vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens)    
    return tokenizer


def token_id_is_lang_id(token_id):
    return token_id < num_langs

class ContextWindowAtomizer:
    def __init__(self, context_size, retain_switch_tokens=False):
        self.context_size = context_size
        self._built_contexts = []
        self._current_context = None
        self.retain_switch_tokens = retain_switch_tokens

    def add_tokenized_item(self, lang_tok, sequence):
        assert token_id_is_lang_id(lang_tok), "lang_tok must be a langid"

        if self.retain_switch_tokens:
            sequence = [lang_tok] + sequence 

        # the multitokenizer still needs to generate the switch tok, for fairness.
        if self._current_context is None:
            self._current_context = [lang_tok]

        assert len(sequence) < self.context_size * 1000 , "code handles very large tokenized items poorly"
        while len(sequence) > 0:
            if len(self._current_context) + len(sequence) < self.context_size:
                self._current_context.extend(sequence)
                sequence = []
            else:
                num_to_write = self.context_size - len(self._current_context)
                self._current_context.extend(sequence[:num_to_write])
                self._built_contexts.append(self._current_context)
                sequence = sequence[num_to_write:]
                if len(sequence) > 0:
                    self._current_context = [lang_tok]
                else:
                    self._current_context = None
                    
    def tensorize(self):
        return torch.tensor(self._built_contexts, dtype=torch.short)

def write_tokenized_file(datasets, tokenizer, output_file, retain_switch_tokens=False):
    atomizer = ContextWindowAtomizer(512, retain_switch_tokens=retain_switch_tokens)
    for dataset in datasets:
        for tokenized_item in tqdm(tokenizer.encode_batch(dataset.formatted_items), desc="Tokenizing items"):
            tokenized_item_ids = tokenized_item.ids
            partitioned_sequence = partition_list_by_lang(tokenized_item_ids)
            for lang_id, monolingual_tokens in partitioned_sequence:
                atomizer.add_tokenized_item(lang_id, monolingual_tokens)

    torch.save(atomizer.tensorize(), output_file)

def write_tokenized_dataset(datasets, tokenizer, output_prefix, retain_switch_tokens=False):
    os.makedirs(output_prefix, exist_ok=True)

    # write a combined tokenized train file, and independent tokenized test files to output_prefix.
    # if strip_switch is True, remove the [BEGIN_EN] and [BEGIN_ES] tokens from the stories.

    write_tokenized_file([datasets['translation_test']], tokenizer, f'{output_prefix}/translation_test.pt', retain_switch_tokens=retain_switch_tokens)  
    write_tokenized_file([datasets['english_train'], datasets['spanish_train'], datasets['translation_train']], 
                         tokenizer, f'{output_prefix}/train.pt', retain_switch_tokens=retain_switch_tokens)
    write_tokenized_file([datasets['english_test']], tokenizer, f'{output_prefix}/english_test.pt', retain_switch_tokens=retain_switch_tokens)
    write_tokenized_file([datasets['spanish_test']], tokenizer, f'{output_prefix}/spanish_test.pt', retain_switch_tokens=retain_switch_tokens)


def main():
    datasets = build_structured_datasets(smoke_test=False)
    vocab_size = 8192
    # shared_tokenizer_small = train_tokenizer([datasets['english_train'], datasets['spanish_train']], vocab_size)
    # shared_tokenizer_large = train_tokenizer([datasets['english_train'], datasets['spanish_train']], vocab_size * 2)
    # english_tokenizer = train_tokenizer([datasets['english_train']], vocab_size)
    # spanish_tokenizer = train_tokenizer([datasets['spanish_train']], vocab_size)

    # save_tokenizer(shared_tokenizer_small, f'{EXPT_NAME}/shared/8k/tokenizer.json')
    # save_tokenizer(shared_tokenizer_large, f'{EXPT_NAME}/shared/16k/tokenizer.json')
    # save_tokenizer(english_tokenizer, f'{EXPT_NAME}/multi_8k/english_tokenizer.json')
    # save_tokenizer(spanish_tokenizer, f'{EXPT_NAME}/multi_8k/spanish_tokenizer.json')

    shared_tokenizer_small = tokenizers.Tokenizer.from_file(f'{EXPT_NAME}/shared/8k/tokenizer.json')
    shared_tokenizer_large = tokenizers.Tokenizer.from_file(f'{EXPT_NAME}/shared/16k/tokenizer.json')
    english_tokenizer = tokenizers.Tokenizer.from_file(f'{EXPT_NAME}/multi_8k/english_tokenizer.json')
    spanish_tokenizer = tokenizers.Tokenizer.from_file(f'{EXPT_NAME}/multi_8k/spanish_tokenizer.json')    
    #write_tokenized_dataset(datasets, shared_tokenizer_large, f'{EXPT_NAME}/shared/16k', retain_switch_tokens=False)
    #write_tokenized_dataset(datasets, shared_tokenizer_small, f'{EXPT_NAME}/shared/8k', retain_switch_tokens=False)

    multi_tokenizer = MultiTokenizer([english_tokenizer, spanish_tokenizer])
    write_tokenized_dataset(datasets, multi_tokenizer, f'{EXPT_NAME}/multi_8k', retain_switch_tokens=True)

def main2():
    pass
            
if __name__ == '__main__':
    main()