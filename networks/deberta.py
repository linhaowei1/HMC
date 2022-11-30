from transformers import DebertaV2PreTrainedModel, DebertaV2Model
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from transformers.models.deberta.modeling_deberta import XDropout, ContextPooler, StableDropout
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)

# Copied from transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification with Deberta->DebertaV2
class DebertaV2ForHMC(DebertaV2PreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)

        output_dim = self.pooler.output_dim

        self.relation = args.relation
        self.cross_entropy = CrossEntropyLoss()
        self.lstm = nn.LSTM(input_size=output_dim, hidden_size=512, num_layers=1, bidirectional=True)

        self.feat_transform = nn.ModuleList(
            [nn.Sequential(nn.Linear(512, output_dim), nn.ReLU(),) for i in range(3)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def get_representation(self, input_ids, attention_mask):
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
        )
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        return pooled_output
    
    @torch.no_grad()
    def update_label_embedding(self):
        # please call this function after every step!
        self.eval()
        self.first_label = nn.Parameter(self.get_representation(self.first_input_ids, self.first_attention_mask))
        self.second_label = nn.Parameter(self.get_representation(self.second_input_ids, self.second_attention_mask))
        self.third_label = nn.Parameter(self.get_representation(self.third_input_ids, self.third_attention_mask))
        #self.first_label.requires_grad = True
        #self.second_label.requires_grad = True
        #self.third_label.requires_grad = True

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        first_label: Optional[torch.Tensor] = None,
        second_label: Optional[torch.Tensor] = None,
        third_label: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        input_ids = torch.stack(input_ids, dim=0).squeeze(1)
        attention_mask = torch.stack(attention_mask, dim=0).squeeze(1)

        if len(attention_mask.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        # (block_num, block_size)
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer).unsqueeze(1)

        h0 = torch.zeros(2, 1, 512).to(pooled_output.device)
        c0 = torch.zeros(2, 1, 512).to(pooled_output.device)
        _, (hn, _) = self.lstm(pooled_output, (h0, c0))
        hn = torch.mean(hn, dim=0)

        _output1, _output2, _output3 = [self.feat_transform[j](hn) for j in range(3)]
        
        first_logits = _output1 @ self.first_label.transpose(0,1)
        second_logits = _output2 @ self.second_label.transpose(0,1)
        third_logits = _output3 @ self.third_label.transpose(0,1)

        loss = None
        
        if first_label is not None:
            loss = self.cross_entropy(first_logits, first_label)
            loss += self.cross_entropy(second_logits, second_label)
            loss += self.cross_entropy(third_logits, third_label)

        output = (first_logits, second_logits, third_logits) + outputs[1:]
        return ((loss,) + output) if loss is not None else output
