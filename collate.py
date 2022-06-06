import logging
import torch
import math
from util import pad_clusters

logger = logging.getLogger(__name__)


class LongformerCollator:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        # pad to the longest doc in the batch
        batch = self.tokenizer.pad(batch)

        batch['input_ids'] = torch.tensor(batch['input_ids'], device=self.device)
        batch['attention_mask'] = torch.tensor(batch['attention_mask'], device=self.device)

        # if all the docs with no clusters pad it to 1 cluster with 1 mention. This is wierd edge case
        max_num_clusters, max_max_cluster_size = max(1, max(batch['num_clusters'])), max(1, max(batch['max_cluster_size']))
        padded_clusters = [pad_clusters(cluster, max_num_clusters, max_max_cluster_size) for cluster in batch['gold_clusters']]
        batch['gold_clusters'] = torch.tensor(padded_clusters, device=self.device)

        return batch


class DynamicBatchSampler:
    def __init__(self, dataset, collator, max_tokens, max_segment_len, max_doc_len=None):
        self.max_tokens = max_tokens
        self.dataset = dataset.sort('length', reverse=True)
        self.collator = collator
        self.max_segment_len = max_segment_len
        self.max_doc_len = max_doc_len

    def __iter__(self):
        batch = []
        per_example_batch_len = 0
        for example in self.dataset:
            if self.max_doc_len is not None and example['length'] > self.max_doc_len:
                continue
            if not batch:
                per_example_batch_len = self.calc_effective_per_example_batch_len(example['length'])
            elif (len(batch) + 1) * per_example_batch_len > self.max_tokens:
                yield self.collator(batch)
                batch = []
                per_example_batch_len = self.calc_effective_per_example_batch_len(example['length'])
            batch.append(example)
        if len(batch) > 0:
            yield self.collator(batch)

    def calc_effective_per_example_batch_len(self, example_len):
        return math.ceil(example_len / self.max_segment_len) * self.max_segment_len
