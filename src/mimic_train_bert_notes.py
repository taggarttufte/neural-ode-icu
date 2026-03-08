"""
ClinicalBERT fine-tuning on MIMIC-IV clinical notes (first 48h, no leakage).

Loads: data/mimic_iv_notes.npz  (output of mimic_prep_notes.py)
Saves: results/mimic_bert_notes_best.pt
       results/mimic_bert_notes_results.txt

Usage:
    python src/mimic_train_bert_notes.py --epochs 1    # test
    python src/mimic_train_bert_notes.py --epochs 5    # full run
"""

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clinicalbert.model import ClinicalBERTClassifier

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
DATA_PATH  = "/home/j19w245/neural-ode-icu/data/mimic_iv_notes.npz"
OUT_DIR    = "/home/j19w245/neural-ode-icu/results"


class NotesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            with autocast():
                logits = model(ids, attn)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(batch['label'].numpy())
    return roc_auc_score(all_labels, all_probs), average_precision_score(all_labels, all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=1)
    parser.add_argument('--batch-size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=2e-5)
    parser.add_argument('--max-length', type=int,   default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading notes data...")
    d      = np.load(DATA_PATH, allow_pickle=True)
    texts  = d['texts']
    labels = d['labels']
    N      = len(labels)
    print(f"  {N} stays | mortality {labels.mean():.1%}")

    # Train / val / test split (70 / 15 / 15)
    rng   = np.random.default_rng(42)
    idx   = rng.permutation(N)
    n_tr  = int(N * 0.70)
    n_va  = int(N * 0.15)
    tr    = idx[:n_tr]
    va    = idx[n_tr:n_tr + n_va]
    te    = idx[n_tr + n_va:]
    print(f"Split: {len(tr)} / {len(va)} / {len(te)}")

    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds     = NotesDataset(texts[tr], labels[tr], tokenizer, args.max_length)
    val_ds       = NotesDataset(texts[va], labels[va], tokenizer, args.max_length)
    test_ds      = NotesDataset(texts[te], labels[te], tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    model     = ClinicalBERTClassifier(MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler    = GradScaler()
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auroc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ids      = batch['input_ids'].to(device)
            attn     = batch['attention_mask'].to(device)
            labels_b = batch['label'].to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(ids, attn)
                loss   = criterion(logits.squeeze(), labels_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        val_auroc, val_auprc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | loss {total_loss/len(train_loader):.4f} | "
              f"val AUROC {val_auroc:.4f} | AUPRC {val_auprc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(),
                       f"{OUT_DIR}/mimic_bert_notes_best.pt")

    test_auroc, test_auprc = evaluate(model, test_loader, device)
    print(f"\nTest AUROC: {test_auroc:.4f} | AUPRC: {test_auprc:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/mimic_bert_notes_results.txt", "w") as f:
        f.write(f"epochs={args.epochs}\n")
        f.write(f"test_auroc={test_auroc:.4f}\n")
        f.write(f"test_auprc={test_auprc:.4f}\n")
        f.write(f"best_val_auroc={best_val_auroc:.4f}\n")
        f.write(f"n_patients={N}\n")
    print(f"Saved to {OUT_DIR}/mimic_bert_notes_results.txt")


if __name__ == "__main__":
    main()
