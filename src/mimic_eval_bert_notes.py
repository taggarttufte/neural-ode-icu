#!/usr/bin/env python3
"""
mimic_eval_bert_notes.py

Inference-only evaluation of the best BERT notes checkpoint
on the canonical test split.

Fixes the training script bug (evaluated last epoch, not best checkpoint).

Requires:
  - data/canonical_split.npz  (from mimic_create_split.py)
  - data/mimic_iv_notes.npz
  - results/mimic_bert_notes_best.pt  (epoch 2 checkpoint)

Saves: results/preds_bert_notes.npz  {preds, labels}

Usage:
  python src/mimic_eval_bert_notes.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score
import os

NOTES_PATH   = "data/mimic_iv_notes.npz"
SPLIT_PATH   = "data/canonical_split.npz"
MODEL_PATH   = "results/mimic_bert_notes_best.pt"
SAVE_PATH    = "results/preds_bert_notes.npz"

BERT_MODEL   = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN      = 512
BATCH_SIZE   = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Model (must match mimic_train_bert_notes.py exactly) ──────────────────────
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls    = out.last_hidden_state[:, 0, :]
        return self.classifier(cls).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────
class NotesTestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading canonical split...")
    split    = np.load(SPLIT_PATH)
    test_ids = set(split["test_ids"].tolist())

    print("Loading notes data...")
    n           = np.load(NOTES_PATH, allow_pickle=True)
    note_ids    = n["stay_ids"].astype(int)
    note_texts  = n["texts"]
    note_labels = n["labels"].astype(int)

    # Select test patients that have notes
    mask_arr    = np.array([sid in test_ids for sid in note_ids])
    test_texts  = note_texts[mask_arr]
    test_labels = note_labels[mask_arr]
    print(f"Test patients (notes) : {len(test_labels):,}  mortality : {test_labels.mean():.4f}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    dataset   = NotesTestDataset(test_texts, test_labels, tokenizer, MAX_LEN)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)

    print("Loading BERT notes best checkpoint (epoch 2)...")
    model = BERTClassifier(BERT_MODEL).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            logits    = model(input_ids, attn_mask)
            preds     = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)
    print(f"BERT Notes  AUROC : {auroc:.4f}  AUPRC : {auprc:.4f}")

    np.savez(SAVE_PATH, preds=preds, labels=labels)
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
