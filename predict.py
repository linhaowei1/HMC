import logging
import config
from utils import utils
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import os
import torch
import math
from dataloader.data import get_dataset
from torch.utils.data import DataLoader
from approaches.train import Appr

logger = logging.getLogger(__name__)

args = config.parse_args()

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(mixed_precision=args.mixed_precision,
                          fp16=args.fp16, kwargs_handlers=[ddp_kwargs])

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
accelerator.wait_for_everyone()

args.relation = utils.Relation()
dataset = get_dataset(args)
model, tokenizer = utils.lookfor_model(args)
state_dict = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'), map_location='cpu')
model.first_label = state_dict['first_label'].to(accelerator.device)
model.second_label = state_dict['second_label'].to(accelerator.device)
model.third_label = state_dict['third_label'].to(accelerator.device)

def block_tokenize_testing(example):
    inputs = tokenizer(example['text'], truncation=False)
    example['input_ids'] = []
    example['attention_mask'] = []
    for i in range(len(inputs['input_ids'])):
        input_ids_list = []
        attention_mask_list = []
        for w in range(math.ceil(len(inputs['input_ids'][i]) / args.split_len)):
            if w == 0:
                input_ids = inputs['input_ids'][i][:args.split_len]
                attention_mask = inputs['attention_mask'][i][:args.split_len]
            else:
                window = args.split_len - args.overlap_len
                input_ids = inputs['input_ids'][i][w * window: w * window + args.split_len]
                attention_mask = inputs['attention_mask'][i][w * window: w * window + args.split_len]
            input_ids_list.append(input_ids + [tokenizer.pad_token_id] * (args.split_len - len(input_ids)))
            attention_mask_list.append(attention_mask + [0] * (args.split_len - len(input_ids)))
        example['input_ids'].append(input_ids_list)
        example['attention_mask'].append(attention_mask_list)
    return example

test_dataset = dataset['pred'].map(block_tokenize_testing, batched=True)

test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=1)

appr = Appr(args)
PRED = appr.pred(model, test_loader, accelerator)
with open(os.path.join(args.model_name_or_path, 'pred'), 'w') as f:
    import json
    json.dump(
        PRED, f, ensure_ascii=False
    )