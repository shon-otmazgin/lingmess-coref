import uuid
import json
import pandas as pd


def flatten(l):
    return [item for sublist in l for item in sublist]


def read_jsonlines(file):
    with open(file, 'r') as f:
        docs = [json.loads(line.strip()) for line in f]
    return docs


def to_dataframe(file_path):
    df = pd.read_json(file_path, lines=True)
    assert 'sentences' in df.columns

    if 'doc_key' not in df.columns:
        df['doc_key'] = [uuid.uuid4().bytes for _ in range(len(df.index))]

    df['tokens'] = df['sentences'].apply(lambda x: flatten(x))
    if 'speakers' in df.columns:
        df['speakers'] = df['speakers'].apply(lambda x: flatten(x))
    else:
        df['speakers'] = df['tokens'].apply(lambda x: [None] * len(x))

    df = df[['doc_key', 'tokens', 'speakers', 'clusters']]

    df = df.dropna()
    df = df.reset_index(drop=True)
    return df
