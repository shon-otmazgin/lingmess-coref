import pandas as pd
import glob
from datasets import Dataset, DatasetDict
import json
import uuid
from transformers import AutoTokenizer

from utilities.consts import SPEAKER_START, SPEAKER_END
from utilities.utils import flatten_list_of_lists

tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True, cache_dir='cache')


def _tokenize(words, clusters, speakers):
    word_idx_to_new_word_idx = dict()
    word_map = []
    text = []
    last_speaker = None

    for idx, (word, speaker) in enumerate(zip(words, speakers)):
        if last_speaker != speaker:
            text += [SPEAKER_START, speaker, SPEAKER_END]
            word_map += [None, None, None]
            last_speaker = speaker
        word_idx_to_new_word_idx[idx] = len(text)
        word_map.append(idx)
        text.append(word)

    for cluster in clusters:
        for start, end in cluster:
            assert words[start:end + 1] == text[word_idx_to_new_word_idx[start]:word_idx_to_new_word_idx[end] + 1]

    encoded_text = tokenizer(text, add_special_tokens=True, is_split_into_words=True)

    new_clusters = [[(encoded_text.word_to_tokens(word_idx_to_new_word_idx[start]).start,
                      encoded_text.word_to_tokens(word_idx_to_new_word_idx[end]).end - 1)
                     for start, end in cluster] for cluster in clusters]

    return {'text': text,
            'input_ids': encoded_text['input_ids'],
            'gold_clusters': new_clusters,
            'token_idx_to_word_idx': encoded_text.word_ids(),
            'word_map': word_map
            }


def read_jsonlines(files):
    docs = []
    for file in files:
        with open(file, 'r') as f:
            docs += [json.loads(line.strip()) for line in f]
    return docs


def data_to_dataframe(docs, gen_doc_key=False):
    df = pd.DataFrame(docs)

    df['tokens'] = df['sentences'].apply(lambda x: flatten_list_of_lists(x))

    if gen_doc_key:
        df['doc_key'] = [uuid.uuid4().bytes for _ in range(len(df.index))]
    if 'speakers' in df.columns:
        df['speakers'] = df['speakers'].apply(lambda x: flatten_list_of_lists(x))
    else:
        df['speakers'] = df['tokens'].apply(lambda x: [None] * len(x))

    cols_to_drop = set(df.columns) - {'doc_key', 'tokens', 'clusters', 'speakers'}
    df = df.drop(columns=cols_to_drop)

    print(df.isna().sum())
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


def encode(example):
    encoded_example = _tokenize(example['tokens'], example['clusters'], example['speakers'])

    gold_clusters = encoded_example['gold_clusters']
    encoded_example['num_clusters'] = len(gold_clusters) if gold_clusters else 0
    encoded_example['max_cluster_size'] = max(len(c) for c in gold_clusters) if gold_clusters else 0
    encoded_example['length'] = len(encoded_example['input_ids'])

    return encoded_example


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_file",
        type=str,
        # required=True,
        help=",
    )
        
    ontonotes_train_files = ['/home/nlp/shon711/fastcoref/data/ontonotes/train.english.jsonlines']
    ontonotes_dev_files = ['/home/nlp/shon711/fastcoref/data/ontonotes/dev.english.jsonlines']
    ontonotes_test_files = ['/home/nlp/shon711/fastcoref/data/ontonotes/test.english.jsonlines']

    df = data_to_dataframe(read_jsonlines(ontonotes_train_files))
    train_dataset = Dataset.from_pandas(df)
    df = data_to_dataframe(read_jsonlines(ontonotes_dev_files))
    dev_dataset = Dataset.from_pandas(df)
    df = data_to_dataframe(read_jsonlines(ontonotes_test_files))
    test_dataset = Dataset.from_pandas(df)

    ontonotes_dataset = DatasetDict({'train': train_dataset, 'dev': dev_dataset, 'test': test_dataset})
    ontonotes_dataset = ontonotes_dataset.map(encode, batched=False)
    ontonotes_dataset = ontonotes_dataset.remove_columns(column_names=['speakers', 'clusters', 'tokens'])
    ontonotes_dataset.save_to_disk('/home/nlp/shon711/fastcoref/data/datasets/ontonotes2')
    print(ontonotes_dataset)
