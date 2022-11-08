import logging
import config
from utils import utils
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import os
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

train_dataset = dataset['train'].map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=args.max_length), batched=True)
test_dataset = dataset['test'].map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=args.max_length), batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'first_label', 'second_label', 'third_label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'first_label', 'second_label', 'third_label'])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)

appr = Appr(args)
appr.train(model, train_loader, test_loader, accelerator)