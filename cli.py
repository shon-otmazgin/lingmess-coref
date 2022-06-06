import argparse

MODEL_TYPES = ['longformer', 'roberta']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        default="longformer",
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="allenai/longformer-large-4096",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
    # Other parameters
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="The path to the huggingface dataset object of ontonotes/silver dataset to be train on. "
             "this should be the dataset created using create_datasets.py script"
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--head_learning_rate", default=3e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--dropout_prob", default=0.3, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--ffnn_size", type=int, default=2048)

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval every X updates steps.")

    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--max_span_length", type=int, default=30)
    parser.add_argument("--top_lambda", type=float, default=0.4)

    parser.add_argument("--experiment_name", type=str, default=None)

    parser.add_argument("--max_segment_len", type=int, default=512)
    parser.add_argument("--max_tokens_in_batch", type=int, default=5000)

    # parser.add_argument("--conll_path_for_eval", type=str, default=None)

    args = parser.parse_args()
    return args
