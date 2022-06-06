import argparse

MODEL_TYPES = ['longformer']


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default="roberta",
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        "--model_name_or_path",
        default="distilroberta-base",
        type=str,
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    # Other parameters
    parser.add_argument(
        "--train_dataset_path",
        default=None,
        type=str,
        help="The path to the huggingface dataset object of ontonotes/silver dataset to be train on. "
             "this should be the dataset created using create_datasets.py script"
    )
    parser.add_argument(
        "--ontonotes_dataset_path",
        default=None,
        type=str,
        help="The path to the huggingface dataset object of ontonotes dataset. "
             "this should be the dataset created using create_datasets.py script"
    )
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--head_learning_rate", default=3e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--dropout_prob", default=0.3, type=float)
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_beta1", default=0.9, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")

    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--max_span_length", type=int, required=False, default=30)
    parser.add_argument("--top_lambda", type=float, default=0.4)

    parser.add_argument("--experiment_name", type=str, default=None)

    parser.add_argument("--ffnn_size", type=int, default=1024)
    parser.add_argument("--max_segment_len", type=int, default=512)
    parser.add_argument("--max_tokens_in_batch", type=int, default=20000)

    parser.add_argument("--save_if_best", action="store_true")

    parser.add_argument("--conll_path_for_eval", type=str, default=None)

    parser.add_argument("--freeze_params", default=None, type=str,
                        help="named parameters to keep fixed. If None or empty - train all")

    args = parser.parse_args()
    return args
