"""
ClinicalBERT fine-tuned for ICU mortality classification.

Architecture:
  ClinicalBERT (frozen or fine-tuned) → [CLS] token → dropout → linear → sigmoid
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class ClinicalBERTClassifier(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT",
                 dropout=0.3, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768 for BERT-base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits.squeeze(-1)  # shape [batch_size]

    def predict_proba(self, input_ids, attention_mask, token_type_ids=None):
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
            return torch.sigmoid(logits)
