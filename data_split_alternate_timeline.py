EXPT_NAME = 'expt_1'
TRAIN_FRACTION = 0.9

import datasets
import os
import torch
import tokenizers

from global_constants import special_tokens, num_langs
from multi_tokenizer import MultiTokenizer


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
        self.formatted_items = [formatter(item) for item in items]
        self.text = '\n'.join(self.formatted_items)


def build_structured_datasets():
    tinystories_ds_en = datasets.load_dataset("roneneldan/TinyStories")
    tinystories_ds_es_translated = datasets.load_dataset("robrenaud/multilingual_tinystories", data_files=["stories_00.json"])

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
    def __init__(self, context_size):
        self.context_size = context_size
        self.built_contexts = []
        self.current_context = []

    def add_tokenized_item(self, tokenized_item):
        def get_most_recent_langid(context):
            for token in reversed(context):
                if token_id_is_lang_id(token):
                    return token
            assert False, "no langid found in context"            


        assert len(tokenized_item) * 10 < self.context_size, "code handles very large tokenized items poorly"
        while len(tokenized_item) > 0:
            if len(self.current_context) + len(tokenized_item) < self.context_size:
                self.current_context.extend(tokenized_item)
                tokenized_item = []
            else:
                num_to_write = self.context_size - len(self.current_context)
                self.current_context.extend(tokenized_item[:num_to_write])
                self.built_contexts.append(self.current_context)

                lang_id = get_most_recent_langid(self.current_context)                

                self.current_context = [lang_id]
                tokenized_item = tokenized_item[num_to_write:]
                    
    def tensorize(self):
        return torch.tensor(self.built_contexts, dtype=torch.short)
        
def write_tokenized_file(datasets, tokenizer, output_file, strip_switch=False):
    atomizer = ContextWindowAtomizer(512)
    for dataset in datasets:
        for tokenized_item in tokenizer.encode_batch(dataset.formatted_items):
            if strip_switch:                
                tokenized_item = [t[0]] + [t for t in tokenized_item if not token_id_is_lang_id(t)]
            atomizer.add_tokenized_item(tokenized_item)
    torch.save(atomizer.tensorize(), output_file)

def write_tokenized_dataset(datasets, tokenizer, output_prefix, strip_switch=False):
    os.makedirs(output_prefix, exist_ok=True)

    # write a combined tokenized train file, and independent tokenized test files to output_prefix.
    # if strip_switch is True, remove the [BEGIN_EN] and [BEGIN_ES] tokens from the stories.

    write_tokenized_file([datasets['english_train'], datasets['spanish_train'], datasets['translation_train']], 
                         tokenizer, f'{output_prefix}/train.pt', strip_switch)
    write_tokenized_file([datasets['english_test']], tokenizer, f'{output_prefix}/english_test.pt', strip_switch)
    write_tokenized_file([datasets['spanish_test']], tokenizer, f'{output_prefix}/spanish_test.pt', strip_switch)
    write_tokenized_file([datasets['translation_test']], tokenizer, f'{output_prefix}/translation_test.pt', strip_switch)    


def main():
    datasets = build_structured_datasets()
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

    write_tokenized_dataset(datasets, shared_tokenizer_small, f'{EXPT_NAME}/shared/8k', strip_switch=True)
    write_tokenized_dataset(datasets, shared_tokenizer_large, f'{EXPT_NAME}/shared/16k', strip_switch=True)

    # multi_tokenizer = MultiTokenizer([english_tokenizer, spanish_tokenizer])
    # write_tokenized_dataset(datasets, multi_tokenizer, f'{EXPT_NAME}/multi_8k', strip_switch=False)


def main2():
    print(shared_tokenizer_small.get_vocab_size())
        
    
if __name__ == '__main__':
    main()