import re
import tempfile
import json
import subprocess
import logging
from collections import defaultdict
import util

logger = logging.getLogger(__name__)

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")  # First line at each document
BEGIN_WIKI_DOCUMENT_REGEX = re.compile(r"#begin document (.*)")  # First line at each document
COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)


def get_doc_key(doc_id, part):
    return "{}_{}".format(doc_id, int(part))


def resolve_doc_key(doc_key):
    parts = doc_key.split('_')
    return '_'.join(parts[:-1]), int(parts[-1])


def write_conll_doc(doc, f_obj):
    placeholder = "  -" * 7
    if 'tokens' in doc:
        tokens = doc['tokens']
    elif 'sentences' in doc: # this is only for OntoNotes. use tokens. we will jst flat the sentences.
        tokens = util.flatten(doc['sentences'])
    else:
        raise NotImplementedError(f'The document should contain tokens field.')
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
        f_obj.write("\n")
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
    f_obj.write("#end document\n\n")


def official_conll_eval(gold_path, predicted_path, metric, official_stdout=True):
    cmd = ["reference-coreference-scorers/scorer.pl", metric, gold_path, predicted_path, "none"]
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


def evaluate_conll(gold_path, predictions, subtoken_maps, word_maps, official_stdout=True):
    with tempfile.NamedTemporaryFile(delete=True, mode="w") as gold_file, \
            tempfile.NamedTemporaryFile(delete=True, mode="w") as pred_file:
        gold_docs = read_jsonlines(gold_path)
        for doc in gold_docs:
            # for gold
            write_conll_doc(doc, gold_file)

            # for predictions
            doc_key = doc['doc_key']
            doc['clusters'] = util.align_clusters(predictions[doc_key], subtoken_maps[doc_key], word_maps[doc_key])
            write_conll_doc(doc, pred_file)
        results = {m: official_conll_eval(gold_file.name, pred_file.name, m, official_stdout) for m in ("muc", "bcub", "ceafe")}
    return results


if __name__ == '__main__':
    import pandas as pd

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    gold_path = '/home/nlp/shon711/fastcoref/data/ontonotes/test.english.jsonlines'
    pred_path = '/home/nlp/shon711/fastcoref/multi_heads_coref/models/test/preds.jsonl'
    print(gold_path)
    print(pred_path)
    df = pd.read_json(pred_path, lines=True).transpose()
    df.columns = ['predicted_clusters', 'subtoken_map', 'word_map']
    print(df.shape)

    doc_to_prediction = df['predicted_clusters'].to_dict()
    doc_to_subtoken_map = df['subtoken_map'].to_dict()
    doc_to_word_map = df['word_map'].to_dict()
    conll_results = evaluate_conll(gold_path, doc_to_prediction, doc_to_subtoken_map, doc_to_word_map)
    official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    logger.info('Official avg F1: %.2f' % official_f1)
