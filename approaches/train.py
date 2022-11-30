import logging
import math
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from sklearn.metrics import f1_score

class Appr(object):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def pred(self, model, test_loader, accelerator):
        model, test_loader = accelerator.prepare(model, test_loader)
        model.eval()
        progress_bar = tqdm(range(len(test_loader)), disable=not accelerator.is_local_main_process)
        PRED = []
        with torch.no_grad():
            for batch, inputs in enumerate(test_loader):
                res = model(**inputs)
                outp = res
                pred1 = outp[0].max(1)[1].item()
                pred1_decode = self.args.relation.idx2label['first'][pred1]
                pred2_candidate = [self.args.relation.idx2label['second'].index(l) \
                    for l in self.args.relation.idx2label['second'] if l.split('-')[0] == pred1_decode]
                
                pred2 = self.args.relation.idx2label['second'][pred2_candidate[0] + outp[1].index_select(1, torch.tensor(pred2_candidate, device=outp[1].device)).max(1)[1].item()]
                if pred2.split('-')[-1] == '未知':
                    PRED.append({'aydj':1, 'ay': pred1_decode})
                else:
                    pred3_candidate = [self.args.relation.idx2label['third'].index(l) \
                        for l in self.args.relation.idx2label['third'] if (l.split('-')[0] + '-' + l.split('-')[1]) == pred2]
                    pred3 = self.args.relation.idx2label['third'][pred3_candidate[0] + outp[2].index_select(1, torch.tensor(pred3_candidate, device=outp[2].device)).max(1)[1].item()]
                    if pred3.split('-')[-1] == '未知':
                        PRED.append({'aydj':2, 'ay': pred2.split('-')[-1]})
                    else:
                        PRED.append({'aydj':3, 'ay': pred3.split('-')[-1]})
                progress_bar.update(1)
        
        return PRED

    def train(self, model, train_loader, test_loader, accelerator):
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if
                            ('deberta' in n) and p.requires_grad],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.deberta_learning_rate,
            },
            {
                'params': [p for n, p in model.named_parameters() if
                            ('lstm' in n) and p.requires_grad],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.lstm_learning_rate,
            },
            {
                'params': [p for n, p in model.named_parameters() if
                            not ('deberta' in n or 'lstm' in n) and p.requires_grad],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.linear_learning_rate,
            }
        ]

        # Set the optimizer
        optimizer = AdamW(optimizer_grouped_parameters)

        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.epoch * num_update_steps_per_epoch
        else:
            self.args.epoch = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # TODO: Warm up can be important

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        # Prepare everything with the accelerator
        model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
        model.update_label_embedding()
        # Train!
        logger.info("***** Running training *****")
        logger.info(
            f"Pretrained Model = {self.args.model_name_or_path}, seed = {self.args.seed}")

        for epoch in range(self.args.epoch):
            print("Epoch {} started".format(epoch))
            self.train_epoch(model, optimizer, train_loader, test_loader, accelerator, lr_scheduler)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(self.args.output_dir + f"_epoch{epoch}")

    def train_epoch(self, model, optimizer, dataloader, test_loader, accelerator, lr_scheduler):
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        train_acc1 = 0.0
        train_acc2 = 0.0
        train_acc3 = 0.0
        training_loss = 0.0
        total_num = 0.0
        
        for batch, inputs in enumerate(dataloader):
            model.train()
            if len(inputs['input_ids']) == 0:
                progress_bar.update(1)
                continue
            res = model(**inputs)

            outp = res[1:]
            loss = res[0]
            loss = loss / self.args.gradient_accumulation_steps
            accelerator.backward(loss)
            if batch % self.args.gradient_accumulation_steps == 0 or batch == len(dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            #unwrapped_model = accelerator.unwrap_model(model)
            #unwrapped_model.update_label_embedding()
            pred1, pred2, pred3 = outp[0].max(1)[1], outp[1].max(1)[1], outp[2].max(1)[1]
            train_acc1 += (inputs['first_label'] == pred1).sum().item()
            train_acc2 += (inputs['second_label'] == pred2).sum().item()
            train_acc3 += (inputs['third_label'] == pred3).sum().item()
            training_loss += loss.item()
            total_num += inputs['third_label'].size(0)
            
            if batch == len(dataloader) - 1:
                acc1, acc2, acc3 = self.eval(model, test_loader, accelerator)
                logger.info("acc1 = {:.4f}, acc2 = {:.4f} acc3 = {:.4f} (seed={})".format(acc1, acc2, acc3, self.args.seed))
            progress_bar.update(1)
            # break
        return 

    def eval(self, model, dataloader, accelerator):
        model.eval()
        total_loss = 0
        total_num = 0
        train_acc1 = 0.0
        train_acc2 = 0.0
        train_acc3 = 0.0
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            for batch, inputs in enumerate(dataloader):
                res = model(**inputs)
                outp = res[1:]
                loss = res[0]
                pred1, pred2, pred3 = outp[0].max(1)[1], outp[1].max(1)[1], outp[2].max(1)[1]
                train_acc1 += (inputs['first_label'] == pred1).sum().item()
                train_acc2 += (inputs['second_label'] == pred2).sum().item()
                train_acc3 += (inputs['third_label'] == pred3).sum().item()
                total_num += inputs['third_label'].size(0)
                progress_bar.update(1)

        return train_acc1 / total_num, train_acc2 / total_num, train_acc3 / total_num
