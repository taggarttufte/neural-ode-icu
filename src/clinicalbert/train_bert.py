"""
Fine-tune ClinicalBERT on PhysioNet 2012 mortality prediction.

Usage:
    python -m src.clinicalbert.train_bert --epochs 5 --batch-size 16 --lr 2e-5
"""

import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from src.clinicalbert.model import ClinicalBERTClassifier
from src.clinicalbert.serialize import patient_to_text

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
DATA_DIR = "data/processed"


class PhysioNetTextDataset(Dataset):
    def __init__(self, patients, tokenizer, max_length=512):
        self.patients = patients
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        text = patient_to_text(p['times'], p['values'], p['mask'])
        label = float(p['label'])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].numpy()
            probs = model.predict_proba(ids, mask).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    return auroc, auprc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--freeze-bert', action='store_true',
                        help='Freeze BERT weights, only train classifier head')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training (faster on modern GPUs)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader worker processes for faster data loading')
    parser.add_argument('--early-stop-patience', type=int, default=5,
                        help='Stop if val AUROC does not improve for N epochs (0=disable)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load preprocessed data
    print("Loading data...")
    with open(f"{DATA_DIR}/train.pkl", 'rb') as f:
        train_patients = pickle.load(f)
    with open(f"{DATA_DIR}/val.pkl", 'rb') as f:
        val_patients = pickle.load(f)
    with open(f"{DATA_DIR}/test.pkl", 'rb') as f:
        test_patients = pickle.load(f)

    print(f"Train: {len(train_patients)} | Val: {len(val_patients)} | Test: {len(test_patients)}")

    # Tokenizer + datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = PhysioNetTextDataset(train_patients, tokenizer)
    val_ds = PhysioNetTextDataset(val_patients, tokenizer)
    test_ds = PhysioNetTextDataset(test_patients, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True)

    # Model
    model = ClinicalBERTClassifier(freeze_bert=args.freeze_bert).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.1f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # Class weighting: gentle nudge (2.0) rather than full 6.21 ratio to avoid gradient instability
    pos_weight = torch.tensor([2.0]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler() if args.fp16 else None

    best_val_auroc = 0
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            if args.fp16:
                with autocast():
                    logits = model(ids, mask)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(ids, mask)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()

        val_auroc, val_auprc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} "
              f"| Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save(model.state_dict(), "results/best_clinicalbert.pt")
            print(f"  Saved best model (Val AUROC: {val_auroc:.4f})")
        else:
            patience_counter += 1
            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                print(f"  Early stopping: no improvement for {args.early_stop_patience} epochs")
                break

    # Final test evaluation
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load("results/best_clinicalbert.pt"))
    test_auroc, test_auprc = evaluate(model, test_loader, device)
    print(f"\nTest AUROC: {test_auroc:.4f} | Test AUPRC: {test_auprc:.4f}")


if __name__ == '__main__':
    main()
