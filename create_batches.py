import logging
from collections import defaultdict

from tqdm import tqdm
from datasets import Dataset

logger = logging.getLogger(__name__)


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
