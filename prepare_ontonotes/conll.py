import re
import tempfile
import json
import subprocess
import logging
from collections import defaultdict
import pandas as pd
from pathlib import Path


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")  # First line at each document
BEGIN_WIKI_DOCUMENT_REGEX = re.compile(r"#begin document (.*)")  # First line at each document
COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_doc_key(doc_id, part):
    return "{}_{}".format(doc_id, int(part))


def resolve_doc_key(doc_key):
    parts = doc_key.split('_')
    return '_'.join(parts[:-1]), int(parts[-1])


def write_conll_doc(doc, f_obj):
    placeholder = "  -" * 7
    if 'tokens' not in doc:
        raise NotImplementedError(f'The document should contain tokens field.')

    tokens = doc['tokens']
    clusters = doc['clusters']
    doc_id, part_id = resolve_doc_key(doc["doc_key"])

    max_word_len = max(len(w) for w in tokens)

    starts = defaultdict(lambda: [])
    ends = defaultdict(lambda: [])
    single_word = defaultdict(lambda: [])

    for cluster_id, cluster in enumerate(clusters):
        for start, end in cluster:
            if end == start:
                single_word[start].append(cluster_id)
            else:
                starts[start].append(cluster_id)
                ends[end].append(cluster_id)

    f_obj.write(f"#begin document ({doc_id}); part {part_id:0>3d}\n")
    for token_idx, token in enumerate(tokens):
        cluster_info_lst = []
        for cluster_marker in starts[token_idx]:
            cluster_info_lst.append(f"({cluster_marker}")
        for cluster_marker in single_word[token_idx]:
            cluster_info_lst.append(f"({cluster_marker})")
        for cluster_marker in ends[token_idx]:
            cluster_info_lst.append(f"{cluster_marker})")
        cluster_info = "|".join(cluster_info_lst) if cluster_info_lst else "-"

        f_obj.write(f"{doc_id}  {part_id}  {token_idx:>2}"
                    f"  {token:>{max_word_len}}{placeholder}  {cluster_info}\n")
    f_obj.write("\n")
    f_obj.write("#end document\n")


def official_conll_eval(gold_path, predicted_path, metric, official_stdout=True):
    cmd = ["prepare_ontonotes/conll-2012/scorer/v8.01/scorer.pl",
           metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        logger.error(stderr)

    if official_stdout:
        logger.info("Official result for {}".format(metric))
        logger.info(stdout)

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return {"r": recall, "p": precision, "f": f1}


def read_jsonlines(file):
    docs = []
    with open(file, 'r') as f:
        docs += [json.loads(line.strip()) for line in f]
    return docs


def evaluate_conll(gold_df, pred_df, official_stdout=True):
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as gold_file, \
            tempfile.NamedTemporaryFile(delete=False, mode="w") as pred_file:
        for index in gold_df.index:
            # for gold
            doc = gold_df.loc[index].to_dict()
            write_conll_doc(doc, gold_file)

            # for predictions
            doc_key = doc['doc_key']
            doc['clusters'] = pred_df[pred_df['doc_key'] == doc_key]['clusters'].values[0]
            write_conll_doc(doc, pred_file)
    print(gold_file.name)
    print(pred_file.name)
    results = {m: official_conll_eval(gold_file.name, pred_file.name, m, official_stdout) for m in ("muc", "bcub", "ceafe")}
    return results


if __name__ == '__main__':
    print(Path.cwd())
    gold_path = 'prepare_ontonotes/test.english.jsonlines'
    pred_path = '/home/nlp/shon711/lingmess-coref/test/test.english.output.jsonlines'
    # pred_path = '/home/nlp/shon711/lingmess-coref/test2/test.english.output.jsonlines'

    gold_df = pd.read_json(gold_path, lines=True)
    gold_df['tokens'] = gold_df['sentences'].apply(lambda x: flatten(x))

    pred_df = pd.read_json(pred_path, lines=True)
    cols = ['doc_key', 'tokens', 'clusters']
    gold_df = gold_df[cols]
    pred_df = pred_df[cols]

    assert gold_df['doc_key'].tolist() == pred_df['doc_key'].tolist()
    assert all([gold_tokens == pred_tokens
                for gold_tokens, pred_tokens in zip(gold_df['tokens'].tolist(), pred_df['tokens'].tolist())])

    conll_results = evaluate_conll(gold_df, pred_df)
    official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    logger.info('Official avg F1: %.2f' % official_f1)
