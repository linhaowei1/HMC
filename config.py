import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--baseline', type=str)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--idrandom', type=int, default=0)
    parser.add_argument('--base_dir', type=str, default='/data1/haowei')
    parser.add_argument('--mixed_precision',type=str)
    parser.add_argument("--sequence_file",type=str, help="sequence file")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--smax", default=600, type=int, help="smax")
    parser.add_argument('--warmup_ratio',  type=float)
    parser.add_argument('--replay_buffer_size', type=int)
    parser.add_argument('--max_length', type=int)
    parser.add_argument('--split_len', type=int, default=256)
    parser.add_argument('--overlap_len', type=int, default=32)
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--output_dir', type=str, default='./log')
    parser.add_argument('--eval_during_training', action="store_true")
    parser.add_argument(
        "--deberta_learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lstm_learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--linear_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--clipgrad", type=int, default=10)
    parser.add_argument('--thres_cosh',default=50,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--epoch", type=int, help="Total number of training epochs to perform.")
    return parser.parse_args()