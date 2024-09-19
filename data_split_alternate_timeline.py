EXPT_NAME = 'expt_1'
TRAIN_FRACTION = 0.9

import datasets
import os
from tqdm import tqdm
from enum import Enum
import tokenizers
from typing import Iterator
from dataclasses import dataclass
from typing import List, Dict, Any

from global_constants import special_tokens

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
    def __init__(self, name, items, formatter):
        self.name = name
        self.items = items
        self.formatted_items = [formatter(item) for item in items]
        self.text = '\n'.join(self.formatted_items)
        self.tokenizers = {}
        self.tokenized = {}


def build_structured_datasets():
    tinystories_ds_es_translated = datasets.load_dataset("robrenaud/multilingual_tinystories", data_files=["stories_00.json"])
    tinystories_ds_en = datasets.load_dataset("roneneldan/TinyStories")
    raw_es_stories = [f'stories_{i:02d}.json' for i in range(1, 22)]
    tinystories_ds_es = datasets.load_dataset("robrenaud/multilingual_tinystories", data_files=raw_es_stories)

    # Display Dataset Sizes
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

    english_train_ds = TinyStoriesDataset('english_train', english_train, write_english_story_tinyprompt)
    english_test_ds = TinyStoriesDataset('english_test', english_test, write_english_story_tinyprompt)

    spanish_train_ds = TinyStoriesDataset('spanish_train', spanish_train, write_spanish_story_tinyprompt)
    spanish_test_ds = TinyStoriesDataset('spanish_test', spanish_test, write_spanish_story_tinyprompt)

    translation_train_ds = TinyStoriesDataset('translation_train', translation_train, write_translation_story_tinyprompt)
    translation_test_ds = TinyStoriesDataset('translation_test', translation_test, write_translation_story_tinyprompt)

    return {
        'english_train': english_train_ds,
        'english_test': english_test_ds,
        'spanish_train': spanish_train_ds,
        'spanish_test': spanish_test_ds,
        'translation_train': translation_train_ds,
        'translation_test': translation_test_ds
    }


def train_tokenizer(datasets, vocab_size):
    concat_text = ""
    for dataset in datasets:
        concat_text += dataset.text + '\n'
    location = "/tmp/data.txt"
    with open(location, 'w') as f:
        f.write(concat_text)
    tokenizer = tokenizers.ByteLevelBPETokenizer()
    tokenizer.train(files=[location], vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens)
    return tokenizer


def create_tokenizers():
    os.makedirs(f'{EXPT_NAME}/shared/tokenizer', exist_ok=True)
    os.makedirs(f'{EXPT_NAME}/multi/tokenizer', exist_ok=True)
    
    for lang in ['english', 'spanish']:
        train_filename = f'{EXPT_NAME}/train_{lang}.txt'
        tokenizer = tokenizers.ByteLevelBPETokenizer()
        tokenizer.train(files=[train_filename], vocab_size=8192, min_frequency=2, special_tokens=special_tokens)        
        tokenizer.save(f'{EXPT_NAME}/multi/tokenizer/{lang}')
    
    filenames = [f'{EXPT_NAME}/train_english.txt', f'{EXPT_NAME}/train_spanish.txt']
    for size in ['small', 'large']:
        shared_tokenizer = tokenizers.ByteLevelBPETokenizer()
        shared_tokenizer.train(files=filenames, vocab_size=8192 * 2 if size == 'large' else 1, 
                               min_frequency=2, special_tokens=special_tokens)        
        shared_tokenizer.save(f'{EXPT_NAME}/shared/{size}_tokenizer')

def main():
    datasets = build_structured_datasets()
    shared_tokenizer_small = train_tokenizer([datasets['english_train'], datasets['spanish_train']], 8192)
    shared_tokenizer_large = train_tokenizer([datasets['english_train'], datasets['spanish_train']], 8192 * 2)
    english_tokenizer = train_tokenizer([datasets['english_train']], 8192)
    spanish_tokenizer = train_tokenizer([datasets['spanish_train']], 8192)

    write_train_test_splits_by_task(datasets)    
    # create_tokenizers()

    tokenize_datasets_to_disk(datasets)
    
if __name__ == '__main__':
    main()