import os
import logging
import numpy as np
import torch

from consts import CATEGORIES
from metrics import CorefEvaluator, MentionEvaluator
from util import create_clusters, create_mention_to_antecedent, update_metrics, \
    output_evaluation_metrics, write_prediction_to_jsonlines
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, args, eval_dataloader):
        self.args = args
        self.output_dir = args.output_dir
        self.eval_dataloader = eval_dataloader

    def evaluate(self, model, prefix="", official=False):
        # Eval!
        model.eval()

        logger.info(f"***** Running Inference {prefix} *****")
        logger.info(f"  Examples number: {len(self.eval_dataloader.dataset)}")

        metrics_dict = {'post_pruning': MentionEvaluator(), 'mentions': MentionEvaluator(), 'coref': CorefEvaluator()}
        doc_to_tokens = {}
        doc_to_subtoken_map = {}
        doc_to_new_word_map = {}
        doc_to_prediction = {}
        categories_eval = {cat_name: {'cat_id': cat_id, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
                           for cat_name, cat_id in CATEGORIES.items()}

        evaluation = False
        data_iterator = tqdm(self.eval_dataloader, desc="Inference", total=len(self.eval_dataloader.dataset))
        start_time = time.time()
        for idx, batch in enumerate(data_iterator):
            doc_keys = batch['doc_key']
            tokens = batch['tokens']
            subtoken_map = batch['subtoken_map']
            new_token_map = batch['new_token_map']
            gold_clusters = batch['gold_clusters']

            with torch.no_grad():
                outputs = model(batch, gold_clusters=None, return_all_outputs=True)

            outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)
            gold_clusters = gold_clusters.cpu().numpy() if gold_clusters is not None else gold_clusters

            span_starts, span_ends, coref_logits, categories_labels = outputs_np
            doc_indices, mention_to_antecedent = create_mention_to_antecedent(span_starts, span_ends, coref_logits)

            for i, doc_key in enumerate(doc_keys):
                doc_mention_to_antecedent = mention_to_antecedent[np.nonzero(doc_indices == i)]
                predicted_clusters = create_clusters(doc_mention_to_antecedent)

                doc_to_prediction[doc_key] = predicted_clusters
                doc_to_tokens[doc_key] = tokens[i]
                doc_to_subtoken_map[doc_key] = subtoken_map[i]
                doc_to_new_word_map[doc_key] = new_token_map[i]

                if gold_clusters is not None:
                    evaluation = True
                    update_metrics(metrics_dict, span_starts[i], span_ends[i], gold_clusters[i], predicted_clusters)

            data_iterator.update(n=len(doc_keys))

        results = {}
        if evaluation:
            results = output_evaluation_metrics(
                metrics_dict=metrics_dict, output_dir=self.output_dir, prefix=prefix, official=official
            )

        write_prediction_to_jsonlines(
            self.args, doc_to_prediction, doc_to_tokens, doc_to_subtoken_map, doc_to_new_word_map
        )

        logger.info(f'Inference time: {time.time() - start_time:.6f} seconds')
        return results
