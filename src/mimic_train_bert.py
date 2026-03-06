"""
ClinicalBERT fine-tuning on MIMIC-IV data.

Usage:
    python src/mimic_train_bert.py --epochs 1       # test run
    python src/mimic_train_bert.py --epochs 5       # full run
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
DATA_PATH  = "/home/j19w245/neural-ode-icu/data/mimic_iv_processed_v2.npz"
OUT_DIR    = "/home/j19w245/neural-ode-icu/results"

FEATURE_NAMES = [
    'HR', 'SBP', 'DBP', 'MBP', 'RespRate', 'Temp', 'SpO2',
    'Glucose', 'Creatinine', 'Sodium', 'Potassium', 'Hematocrit', 'WBC',
    'Bicarbonate', 'BUN', 'GCS_Motor', 'GCS_Verbal', 'GCS_Eye',
    'UrineOut', 'Age'
]


def patient_to_text(values, mask, max_chars=2000):
    """Serialize a patient's hourly time series to text for BERT."""
    parts = []
    for t in range(values.shape[0]):
        observed = []
        for v, name in enumerate(FEATURE_NAMES):
            if mask[t, v] == 1:
                observed.append(f"{name}={values[t, v]:.1f}")
        if observed:
            parts.append(f"Hr{t}: {', '.join(observed)}.")
    text = " ".join(parts)
    return text[:max_chars] if text else "No observations."


class MIMICTextDataset(Dataset):
    def __init__(self, values, masks, labels, tokenizer, max_length=512):
        self.values    = values
        self.masks     = masks
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = patient_to_text(self.values[idx], self.masks[idx])
        enc  = self.tokenizer(
            text,
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

    print("Loading data...")
    d      = np.load(DATA_PATH)
    values = d['values']
    masks  = d['masks']
    labels = d['labels']
    N = len(labels)

    rng    = np.random.default_rng(42)
    idx    = rng.permutation(N)
    n_tr   = int(N * 0.70); n_va = int(N * 0.15)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

    print(f"Split: {len(tr)} / {len(va)} / {len(te)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = MIMICTextDataset(values[tr], masks[tr], labels[tr], tokenizer, args.max_length)
    val_ds   = MIMICTextDataset(values[va], masks[va], labels[va], tokenizer, args.max_length)
    test_ds  = MIMICTextDataset(values[te], masks[te], labels[te], tokenizer, args.max_length)

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
            ids    = batch['input_ids'].to(device)
            attn   = batch['attention_mask'].to(device)
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
            torch.save(model.state_dict(), f"{OUT_DIR}/mimic_bert_best.pt")

    test_auroc, test_auprc = evaluate(model, test_loader, device)
    print(f"\nTest AUROC: {test_auroc:.4f} | AUPRC: {test_auprc:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/mimic_bert_results.txt", "w") as f:
        f.write(f"epochs={args.epochs}\n")
        f.write(f"test_auroc={test_auroc:.4f}\n")
        f.write(f"test_auprc={test_auprc:.4f}\n")


if __name__ == "__main__":
    main()
