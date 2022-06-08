import logging
import os
import shutil

import datasets
import torch
from transformers import AutoConfig, AutoTokenizer, LongformerConfig, RobertaConfig, DebertaV2Config

from consts import SUPPORTED_MODELS
from create_batches import create_batches
from modeling import LingMessCoref
from training import train
from eval import Evaluator
from util import set_seed, save_all
from cli import parse_args
from collate import LongformerCollator, DynamicBatchSampler, SegmentCollator
import wandb

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


def main():
    args = parse_args()

    if args.experiment_name is not None:
        # TODO: change entity
        wandb.init(project=args.experiment_name, entity="shon711", config=args)

    if os.path.exists(args.output_dir):
        if args.overwrite_output_dir:
            shutil.rmtree(args.output_dir)
            logger.info(f'--overwrite_output_dir used. directory {args.output_dir} deleted!')
        else:
            raise ValueError(f"Output directory ({args.output_dir}) already exists. Use --overwrite_output_dir to overcome.")
    os.mkdir(args.output_dir)

    # Setup CUDA, GPU & distributed training
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = 1
    set_seed(args)

    # TODO: do something with the cache dir
    args.cache_dir = '~/.cache'

    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, use_fast=False)

    model, loading_info = LingMessCoref.from_pretrained(
        args.model_name_or_path, output_loading_info=True,
        config=config, cache_dir=args.cache_dir, args=args
    )

    if model.base_model_prefix not in SUPPORTED_MODELS:
        raise NotImplementedError(f'Model not supporting {args.model_type}, choose one of {SUPPORTED_MODELS}')
    args.base_model = model.base_model_prefix

    model.to(args.device)
    for key, val in loading_info.items():
        logger.info(f'{key}: {val}')

    t_params, h_params = [p / 1000000 for p in model.num_parameters()]
    logger.info(f'Parameters: {t_params + h_params:.1f}M, Transformer: {t_params:.1f}M, Head: {h_params:.1f}M')

    # load datasets and loaders for eval
    dataset = datasets.load_from_disk(args.dataset_path)
    if args.base_model == 'longformer':
        collator = LongformerCollator(tokenizer=tokenizer, device=args.device)
        max_doc_len = 4096
    else:
        collator = SegmentCollator(tokenizer=tokenizer, device=args.device, max_segment_len=args.max_segment_len)
        max_doc_len = None

    eval_dataloader = DynamicBatchSampler(
        dataset[args.eval_split],
        collator=collator,
        max_tokens=args.max_tokens_in_batch,
        max_segment_len=args.max_segment_len,
        max_doc_len=max_doc_len
    )
    evaluator = Evaluator(args=args, eval_dataloader=eval_dataloader)

    # Training
    if args.do_train:
        train_batches_path = f'{args.dataset_path}_batches_{args.base_model}_{args.max_tokens_in_batch}'
        try:
            logger.info(f'Reading Train batches from {train_batches_path}')
            train_batches = datasets.load_from_disk(train_batches_path)
        except FileNotFoundError:
            logger.info(f'Train batches not found !')
            train_dataset = datasets.load_from_disk(args.dataset_path)
            train_sampler = DynamicBatchSampler(
                train_dataset['train'],
                collator=collator,
                max_tokens=args.max_tokens_in_batch,
                max_segment_len=args.max_segment_len,
                max_doc_len=max_doc_len
            )
            train_batches = create_batches(sampler=train_sampler, path_to_save=train_batches_path)
        train_batches = train_batches.shuffle(seed=args.seed)
        logger.info(train_batches)

        global_step, tr_loss = train(args, train_batches, model, tokenizer, evaluator)
        logger.info(f"global_step = {global_step}, average loss = {tr_loss}")

    # Evaluation
    results = evaluator.evaluate(model, prefix="final_evaluation", official=True)

    return results


if __name__ == "__main__":
    main()
