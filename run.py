import logging
import os
import shutil

import datasets
import torch
from transformers import AutoConfig, AutoTokenizer, CONFIG_MAPPING, LongformerConfig

from modeling import LingMessCoref
# from training import train
from eval import Evaluator
from util import set_seed, save_all
from cli import parse_args
# from data import LongformerCollator, DynamicBatchSampler
import wandb

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


def main():
    args = parse_args()
    wandb.init(project=args.experiment_name, entity="shon711", config=args)

    if os.path.exists(args.output_dir):
        if args.overwrite_output_dir:
            shutil.rmtree(args.output_dir)
            logger.info(f'--overwrite_output_dir used. directory {args.output_dir} deleted!')
        else:
            raise ValueError(f"Output directory ({args.output_dir}) already exists. Use --overwrite_output_dir to overcome.")
    os.mkdir(args.output_dir)

    # Setup CUDA, GPU & distributed training
    device = torch.device('cuda:1' if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = 1      # torch.cuda.device_count()

    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    for key, val in vars(args).items():
        logger.info(f"{key} - {val}")

    # Set seed
    set_seed(args)

    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, use_fast=False)

    LingMessCoref.config_class = LongformerConfig
    LingMessCoref.base_model_prefix = args.model_type
    model, info = LingMessCoref.from_pretrained(args.model_name_or_path, output_loading_info=True,
                                                config=config,
                                                cache_dir=args.cache_dir,
                                                args=args)
    for key, val in info.items():
        logger.info(f'{key}: {val}')
    model.to(args.device)

    t_params, h_params = [p / 1000000 for p in model.num_parameters()]
    logger.info(f'model parameters: transformer: {t_params:.1f}M, head: {h_params:.1f}M, total: {t_params + h_params:.1f}M')

    # load datasets and loaders for eval
    ontonotes_dataset = datasets.load_from_disk(args.ontonotes_dataset_path)
    longformer_collator = LongformerCollator(tokenizer=tokenizer, device=args.device)
    eval_dataloader = DynamicBatchSampler(
        ontonotes_dataset['test'],
        collator=longformer_collator,
        max_tokens=15000,
        max_segment_len=args.max_segment_len,
        max_doc_len=4096
    )
    evaluator = Evaluator(args=args, tokenizer=tokenizer, eval_dataloader=eval_dataloader)

    # Training
    if args.do_train:
        train_batches_path = f'{args.train_dataset_path}_batches_longformer_{args.max_tokens_in_batch}'
        try:
            logger.info(f'Reading Train batches from {train_batches_path}')
            train_batches = datasets.load_from_disk(train_batches_path)
        except FileNotFoundError:
            logger.info(f'Train batches not found !')
            train_dataset = datasets.load_from_disk(args.train_dataset_path)
            train_sampler = DynamicBatchSampler(
                train_dataset['train'],
                collator=longformer_collator,
                max_tokens=args.max_tokens_in_batch,
                max_segment_len=args.max_segment_len,
                max_doc_len=4096
            )
            train_batches = create_batches(sampler=train_sampler, path_to_save=train_batches_path, leftovers=False)
        train_batches = train_batches.shuffle(seed=args.seed)
        logger.info(train_batches)

        global_step, tr_loss = train(args, train_batches, model, tokenizer, evaluator)
        logger.info(f"global_step = {global_step}, average loss = {tr_loss}")

        # Save any files starting with "checkpoint" as they're written to
        # logger.info(f"Uploads results to wandb")
        # wandb.save(os.path.abspath(args.output_dir))

    # Evaluation
    results = {}
    if args.do_eval:
        result = evaluator.evaluate(model, prefix="final_evaluation", official=True)
        results.update(result)

    return results


if __name__ == "__main__":
    main()
