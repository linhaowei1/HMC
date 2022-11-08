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

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

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
        self.first_label = self.get_representation(self.first_input_ids, self.first_attention_mask)
        self.second_label = self.get_representation(self.second_input_ids, self.second_attention_mask)
        self.third_label = self.get_representation(self.third_input_ids, self.third_attention_mask)
        self.first_label.requires_grad = True
        self.second_label.requires_grad = True
        self.third_label.requires_grad = True

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        first_class_labels: Optional[torch.Tensor] = None,
        second_class_labels: Optional[torch.Tensor] = None,
        third_class_labels: Optional[torch.Tensor] = None,
        second_class_index: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)

        first_logits = pooled_output @ self.first_label.transpose()
        second_logits = pooled_output @ self.second_label.transpose()
        third_logits = pooled_output @ self.third_label.transpose()

        loss = None
        
        if first_class_labels is not None:
            loss = self.cross_entropy(first_logits, first_class_labels)
            loss += self.cross_entropy(second_logits, second_class_labels)
            loss += self.cross_entropy(third_logits, third_class_labels)

        output = (first_logits, second_logits, third_logits) + outputs[1:]
        return ((loss,) + output) if loss is not None else output
