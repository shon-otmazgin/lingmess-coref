import logging
import os
import shutil

import datasets
import torch
from transformers import AutoConfig, AutoTokenizer, LongformerConfig, RobertaConfig, BertConfig

from create_batches import create_batches
from modeling import LingMessCoref
from training import train
from eval import Evaluator
from util import set_seed, save_all
from cli import parse_args
from collate import LongformerCollator, DynamicBatchSampler, BertLikeCollator
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

    args.model_type = args.model_type.lower()
    if args.model_type == 'longformer':
        LingMessCoref.config_class = LongformerConfig
    elif args.model_type == 'roberta':
        LingMessCoref.config_class = RobertaConfig
    elif args.model_type == 'bert':
        LingMessCoref.config_class = BertConfig
    else:
        raise NotImplementedError(f'model_type {args.model_type} not supported. choose one of bert/roberta/longformer')

    LingMessCoref.base_model_prefix = args.model_type
    model, loading_info = LingMessCoref.from_pretrained(
        args.model_name_or_path, output_loading_info=True,
        config=config, cache_dir=args.cache_dir, args=args
    )
    model.to(args.device)
    for key, val in loading_info.items():
        logger.info(f'{key}: {val}')

    t_params, h_params = [p / 1000000 for p in model.num_parameters()]
    logger.info(f'Parameters: {t_params + h_params:.1f}M, Transformer: {t_params:.1f}M, Head: {h_params:.1f}M')

    # load datasets and loaders for eval
    dataset = datasets.load_from_disk(args.dataset_path)
    if args.model_type == 'longformer':
        collator = LongformerCollator(tokenizer=tokenizer, device=args.device)
        max_doc_len = 4096
    else:
        collator = BertLikeCollator(tokenizer=tokenizer, device=args.device, max_segment_len=args.max_segment_len)
        max_doc_len = None

    eval_dataloader = DynamicBatchSampler(
        dataset[args.split_to_eval],
        collator=collator,
        max_tokens_in_batch=args.max_tokens_in_batch,
        max_segment_len=args.max_segment_len,
        max_doc_len=max_doc_len
    )
    evaluator = Evaluator(args=args, eval_dataloader=eval_dataloader)

    # Training
    if args.do_train:
        train_batches_path = f'{args.dataset_path}_batches_{args.model_type}_{args.max_tokens_in_batch}'
        try:
            logger.info(f'Reading Train batches from {train_batches_path}')
            train_batches = datasets.load_from_disk(train_batches_path)
        except FileNotFoundError:
            logger.info(f'Train batches not found !')
            train_dataset = datasets.load_from_disk(args.dataset_path)
            train_sampler = DynamicBatchSampler(
                train_dataset['train'],
                collator=collator,
                max_tokens_in_batch=args.max_tokens_in_batch,
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
