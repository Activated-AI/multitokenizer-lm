EXPT_NAME = 'expt_1'
TRAIN_FRACTION = 0.9

import datasets
import os
from tqdm import tqdm
from enum import Enum
import tokenizers
from typing import Iterator

special_tokens = ['[BEGIN_EN]', '[BEGIN_ES]', '[PROMPT_EN_STORY]', '[PROMPT_ES_STORY]', '[PROMPT_EN_ES_TRANS]', '[END]']


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

def function_applying_iterator(items, func):
    for item in items:
        yield func(item)

class FunctionApplyingIterable:
    def __init__(self, items, func):
        self.items = items
        self.func = func

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return function_applying_iterator(self.items, self.func)


Task = Enum('Task', 'english spanish translation')
        

class DatasetWithFormatters:
    def __init__(self):
        self.tinystories_ds_es_translated = datasets.load_dataset("robrenaud/multilingual_tinystories", data_files=["stories_00.json"])
        self.tinystories_ds_en = datasets.load_dataset("roneneldan/TinyStories")
        self.raw_es_stories = [f'stories_{i:02d}.json' for i in range(1, 22)]
        self.tinystories_ds_es = datasets.load_dataset("robrenaud/multilingual_tinystories", data_files=raw_es_stories)

        # Display Dataset Sizes
        print('num translations   ', len(self.tinystories_ds_es_translated['train']))
        print('num spanish stories', len(self.tinystories_ds_es['train']))
        print('num english stories', len(self.tinystories_ds_en['train']))

        # Prepare Datasets with Formatters
        self.datasets_with_formatters = {
            Task.english: (self.tinystories_ds_en['train'].to_list(), write_english_story_tinyprompt),
            Task.spanish: (self.tinystories_ds_es['train'].to_list(), write_spanish_story_tinyprompt),
            Task.translation: (self.tinystories_ds_es_translated['train'].to_list(), write_translation_story_tinyprompt)            
        }
        self.iterable_datasets = {
            "english" : FunctionApplyingIterable(self.tinystories_ds_en['train'].tolist(), write_english_story_tinyprompt),
            "spanish" : FunctionApplyingIterable(self.tinystories_ds_es['train'], write_spanish_story_tinyprompt),
            "translation" : FunctionApplyingIterable(self.tinystories_ds_es_translated['train'], write_translation_story_tinyprompt)
        }    
    
    def get_task_iterator(self, task: Task, is_train: bool) -> Iterator[str]:
        if is_train:
            split = 

@dataclass
class TinyStoriesDataset:
    name: str
    items: List[str]
    text: str
    tokenized: Dict[str, Any]

def write_train_test_splits_by_task(datasets_with_formatters):    
    os.makedirs(EXPT_NAME, exist_ok=True)

    # Write Training Data to Separate Files
    for task, ds, formatter in datasets_with_formatters:
        train_split = ds[:int(TRAIN_FRACTION * len(ds))]
        train_file_path = os.path.join(EXPT_NAME, f'train_{task}.txt')
        with open(train_file_path, 'w', encoding='utf-8') as f:
            for story in tqdm(FunctionApplyingIterable(train_split, formatter), desc=f'writing train_{task}.txt'):
                if story:  # Ensure that empty strings are not written
                    f.write(story + '\n')

    # Write Testing Data to Separate Files (unchanged)
    for task, ds, formatter in datasets_with_formatters:
        with open(f'{EXPT_NAME}/test_{task}.txt', 'w', encoding='utf-8') as f:
            split_index = int(TRAIN_FRACTION * len(ds))
            test_split = ds[split_index:]
            for story in tqdm(test_split, desc=f'writing test_{task}.txt'):
                formatted_story = formatter(story)
                if formatted_story:  # Ensure that empty strings are not written
                    f.write(formatted_story + '\n')

    
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
    datasets = DatasetsWithFormatters()

    write_train_test_splits_by_task(datasets)    
    # create_tokenizers()

    tokenize_datasets_to_disk(datasets)
    
if __name__ == '__main__':
    main()