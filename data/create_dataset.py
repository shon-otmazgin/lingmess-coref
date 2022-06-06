import sys
import argparse

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


sys.path.append('../')
import util
import consts


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        required=False,
        help="Path to train set file. should be in jsonlines format."
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default=None,
        required=False,
        help="Path to development set file. should be in jsonlines format."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        required=False,
        help="Path to test set file. should be in jsonlines format."
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='roberta-base',
        required=True,
        help="Huggingface Tokenizer name."
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_prefix_space=True, cache_dir='cache')

    files = {'train': args.train_file, 'dev': args.dev_file, 'test': args.test_file}
    dataset_dict = {}
    for split, path in files.items():
        if path is not None:
            df = util.to_dataframe(path)
            dataset_dict[split] = Dataset.from_pandas(df)

    dataset = DatasetDict(dataset_dict)
    dataset = dataset.map(encode, batched=False, fn_kwargs={'tokenizer': tokenizer})
    dataset = dataset.remove_columns(column_names=['tokens', 'speakers', 'clusters'])
    dataset.save_to_disk('dataset')
    print(dataset)

