import json
import os
import logging
from collections import OrderedDict
from data.conll import evaluate_conll
import numpy as np
import torch

from consts import CATEGORIES
from metrics import CorefEvaluator, MentionEvaluator
from util import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters, extract_clusters_for_decode
from tqdm import tqdm
import time
import wandb
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, args, eval_dataloader):
        self.args = args
        self.output_dir = args.output_dir
        self.eval_dataloader = eval_dataloader

    def evaluate(self, model, prefix="", official=False):
        # Eval!
        model.eval()

        logger.info(f"***** Running evaluation {prefix} *****")
        logger.info(f"  Examples number: {len(self.eval_dataloader.dataset)}")

        post_pruning_mention_evaluator = MentionEvaluator()
        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        doc_to_prediction = {}
        doc_to_subtoken_map = {}
        doc_to_new_word_map = {}

        categories_eval = {cat_name: {'cat_id': cat_id, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
                           for cat_name, cat_id in CATEGORIES.items()}

        total_loss = 0
        total_time = 0
        data_iterator = tqdm(self.eval_dataloader, desc="evaluation")
        for idx, batch in enumerate(data_iterator):
            doc_keys = batch['doc_key']
            subtoken_map = batch['subtoken_map']
            new_word_map = batch['new_word_map']
            gold_clusters = batch['gold_clusters']

            start_time = time.time()
            with torch.no_grad():
                outputs = model(batch, gold_clusters=gold_clusters, return_all_outputs=True)

            loss, outputs = outputs[0], outputs[1:]
            total_loss += loss.item()

            end_time = time.time()
            total_time += end_time - start_time

            gold_clusters = gold_clusters.cpu().numpy()
            outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)
            for i, output in enumerate(zip(*outputs_np)):
                doc_key = doc_keys[i]
                gold_clusters_i = extract_clusters(gold_clusters[i])
                mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters_i)
                gold_mentions = list(mention_to_gold_clusters.keys())

                # starts, end_offsets, coref_logits, mention_logits = output
                starts, end_offsets, coref_logits, mention_logits, labels_after_pruning, categories_labels = output
                # # TODO: add to metrics
                # for category in categories_eval:
                #     cat_id = categories_eval[category]['cat_id']
                #     cat_mask = categories_labels == cat_id
                #
                #     labels = labels_after_pruning[:, :-1][cat_mask]
                #     logits = coref_logits[:, :-1][cat_mask]
                #
                #     categories_eval[category]['tn'] += np.logical_and(labels == 0., logits < 0).sum()
                #     categories_eval[category]['fn'] += np.logical_and(labels == 1., logits < 0).sum()
                #
                #     categories_eval[category]['fp'] += np.logical_and(labels == 0., logits > 0).sum()
                #     categories_eval[category]['tp'] += np.logical_and(labels == 1., logits > 0).sum()

                max_antecedents = np.argmax(coref_logits, axis=1).tolist()

                mention_to_antecedent = set()
                for start, end, max_antecedent in zip(starts, end_offsets, max_antecedents):
                    if max_antecedent < len(starts):
                        mention = int(start), int(end)
                        ant = int(starts[max_antecedent]), int(end_offsets[max_antecedent])
                        mention_to_antecedent.add((mention, ant))
                predicted_clusters, _ = extract_clusters_for_decode(mention_to_antecedent)

                candidate_mentions = list(zip(starts, end_offsets))
                mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
                predicted_mentions = list(mention_to_predicted_clusters.keys())
                post_pruning_mention_evaluator.update(candidate_mentions, gold_mentions)
                mention_evaluator.update(predicted_mentions, gold_mentions)
                coref_evaluator.update(predicted_clusters, gold_clusters_i, mention_to_predicted_clusters, mention_to_gold_clusters)

                doc_to_prediction[doc_key] = predicted_clusters
                doc_to_subtoken_map[doc_key] = subtoken_map[i]
                doc_to_new_word_map[doc_key] = new_word_map[i]

        post_pruning_mention_precision, post_pruning_mentions_recall, post_pruning_mention_f1 = post_pruning_mention_evaluator.get_prf()
        mention_precision, mentions_recall, mention_f1 = mention_evaluator.get_prf()
        prec, rec, f1 = coref_evaluator.get_prf()

        results = {
            "loss": total_loss,
            "post pruning mention precision": post_pruning_mention_precision,
            "post pruning mention recall": post_pruning_mentions_recall,
            "post pruning mention f1": post_pruning_mention_f1,
            "mention precision": mention_precision,
            "mention recall": mentions_recall,
            "mention f1": mention_f1,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
        for category in categories_eval:
            tp, fp, fn, tn = categories_eval[category]['tp'], categories_eval[category]['fp'], categories_eval[category]['fn'], categories_eval[category]['tn']
            results[category] = {'true_pairs': tp + fn, 'false_pairs': tn + fp, 'precision': 0, 'recall': 0, 'f1': 0}
            if tp + fp:
                results[category]['precision'] = round((tp / (tp + fp)) * 100, 1)
            if tp + fn:
                results[category]['recall'] = round((tp / (tp + fn)) * 100, 1)
            if tp + 0.5 * (fp + fn):
                results[category]['f1'] = round((tp / (tp + 0.5 * (fp + fn))) * 100, 1)

        logger.info("***** Eval results {} *****".format(prefix))
        output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            if prefix:
                writer.write(f'\n{prefix}:\n')
            for key, value in results.items():
                if isinstance(value, float):
                    logger.info(f"  {key} = {value:.3f}")
                    writer.write(f"{key} = {value:.3f}\n")
                else:
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
            logger.info(f'Total time: {total_time:.6f} seconds')

        # if official:
        #     with open(os.path.join(self.args.output_dir, "preds.jsonl"), "w") as f:
        #         f.write(json.dumps(doc_to_prediction) + '\n')
        #         f.write(json.dumps(doc_to_subtoken_map) + '\n')
        #         f.write(json.dumps(doc_to_new_word_map) + '\n')
        #
        #     if self.args.conll_path_for_eval is not None:
        #         conll_results = evaluate_conll(self.args.conll_path_for_eval, doc_to_prediction, doc_to_subtoken_map, doc_to_new_word_map)
        #         official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        #         logger.info('Official avg F1: %.2f' % official_f1)

        return results
