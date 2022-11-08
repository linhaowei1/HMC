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

    def train(self, model, train_loader, test_loader, accelerator):
        

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in []) and p.requires_grad],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.learning_rate,
            }
        ]
        # Set the optimizer
        # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
        #                   weight_decay=self.args.weight_decay)
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
            acc1, acc2, acc3, training_loss = self.train_epoch(model, optimizer, train_loader, accelerator, lr_scheduler)
            print("train acc1 = {:.4f}, acc2 = {:.4f} acc3 = {:.4f}, training loss = {:.4f}".format(acc1, acc2, acc3, training_loss))

            acc1, acc2, acc3 = self.eval(model, test_loader, accelerator)

            logger.info(
                "last epoch acc1 = {:.4f}, acc2 = {:.4f} acc3 = {:.4f} (seed={})".format(acc1, acc2, acc3, self.args.seed))

    def train_epoch(self, model, optimizer, dataloader, accelerator, lr_scheduler):
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        train_acc1 = 0.0
        train_acc2 = 0.0
        train_acc3 = 0.0
        training_loss = 0.0
        total_num = 0.0
        
        for batch, inputs in enumerate(dataloader):
            model.train()
            res = model(**inputs, return_dict=False)

            outp = res[1:]
            loss = res[0]
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.update_label_embedding()
            pred1, pred2, pred3 = outp[0].max(1)[1], outp[1].max(1)[1], outp[2].max(1)[1]
            train_acc1 += (inputs['first_label'] == pred1).sum().item()
            train_acc2 += (inputs['second_label'] == pred2).sum().item()
            train_acc3 += (inputs['third_label'] == pred3).sum().item()
            training_loss += loss.item()
            total_num += inputs['third_label'].size(0)

            progress_bar.update(1)
            # break
        return train_acc1 / total_num, train_acc2 / total_num, train_acc3 / total_num, training_loss / total_num

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
                res = model(**inputs, return_dict=True)
                outp = res[1:]
                loss = res[0]
                pred1, pred2, pred3 = outp[0].max(1)[1], outp[1].max(1)[1], outp[2].max(1)[1]
                train_acc1 += (inputs['first_label'] == pred1).sum().item()
                train_acc2 += (inputs['second_label'] == pred2).sum().item()
                train_acc3 += (inputs['third_label'] == pred3).sum().item()
                total_num += inputs['third_label'].size(0)
                progress_bar.update(1)

        return train_acc1 / total_num, train_acc2 / total_num, train_acc3 / total_num
