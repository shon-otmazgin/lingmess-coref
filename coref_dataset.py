import logging
from collections import defaultdict

from datasets import Dataset, DatasetDict
from tqdm import tqdm

import util
import consts

logger = logging.getLogger(__name__)


def _tokenize(tokenizer, words, clusters, speakers):
    word_map = []
    new_word_map = []
    text = []
    last_speaker = None

    for idx, (word, speaker) in enumerate(zip(words, speakers)):
        if last_speaker != speaker:
            text += [consts.SPEAKER_START, speaker, consts.SPEAKER_END]
            new_word_map += [None, None, None]
            last_speaker = speaker
        word_map.append(len(text))
        new_word_map.append(idx)
        text.append(word)

    for cluster in clusters:
        for start, end in cluster:
            assert words[start:end + 1] == text[word_map[start]:word_map[end] + 1]

    encoded_text = tokenizer(text, add_special_tokens=True, is_split_into_words=True)

    new_clusters = [[(encoded_text.word_to_tokens(word_map[start]).start,
                      encoded_text.word_to_tokens(word_map[end]).end - 1)
                     for start, end in cluster] for cluster in clusters]

    return {'text': text,
            'input_ids': encoded_text['input_ids'],
            'gold_clusters': new_clusters,
            'subtoken_map': encoded_text.word_ids(),
            'new_word_map': new_word_map
            }


def encode(example, tokenizer):
    encoded_example = _tokenize(tokenizer, example['tokens'], example['clusters'], example['speakers'])

    gold_clusters = encoded_example['gold_clusters']
    encoded_example['num_clusters'] = len(gold_clusters) if gold_clusters else 0
    encoded_example['max_cluster_size'] = max(len(c) for c in gold_clusters) if gold_clusters else 0
    encoded_example['length'] = len(encoded_example['input_ids'])

    return encoded_example


def create(tokenizer, train_file=None, dev_file=None, test_file=None):
    if train_file is None and dev_file is None and test_file is None:
        raise Exception(f'Provide at least train/dev/test file to create the dataset')

    files = {'train': train_file, 'dev': dev_file, 'test': test_file}
    logger.info(f'Creating dataset for {files}')

    dataset_dict = {}
    for split, path in files.items():
        if path is not None:
            df = util.to_dataframe(path)
            dataset_dict[split] = Dataset.from_pandas(df)

    dataset = DatasetDict(dataset_dict)
    dataset = dataset.map(encode, batched=False, fn_kwargs={'tokenizer': tokenizer})
    dataset = dataset.remove_columns(column_names=['tokens', 'speakers', 'clusters'])

    return dataset


def create_batches(sampler, path_to_save=None):
    logger.info(f'Creating batches for {len(sampler.dataset)} examples...')

    # huggingface dataset cannot save tensors. so we will save lists and on train loop transform to tensors.
    batches_dict = defaultdict(lambda: [])

    for i, batch in enumerate(tqdm(sampler)):
        for k, v in batch.items():
            batches_dict[k].append(v)

    batches = Dataset.from_dict(batches_dict)
    logger.info(f'{len(batches)} batches created.')

    if path_to_save is not None:
        batches.save_to_disk(path_to_save)
        logger.info(f'Saving batches to {path_to_save}')

    return batches
