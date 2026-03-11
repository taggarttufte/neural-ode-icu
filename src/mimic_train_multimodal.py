#!/usr/bin/env python3
"""
mimic_train_multimodal.py

Multimodal fusion: GRU time-series encoder + ClinicalBERT notes encoder
Concatenate latent representations → binary mortality classifier

Architecture:
  GRU(input=20, hidden=128, layers=2) → FC → z_ts   (64-dim)
  ClinicalBERT → CLS token            → z_notes     (768-dim)
  concat([z_ts, z_notes]) → FC(832, 256) → ReLU → Dropout → FC(256, 1)

Data:
  Structured : data/mimic_iv_processed_v2.npz  (74829, 48, 20)
  Notes      : data/mimic_iv_notes.npz          (47142 stays)
  Intersection aligned on stay_id (~47k stays expected)

Usage:
  python src/mimic_train_multimodal.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Config ────────────────────────────────────────────────────────────────────
STRUCTURED_PATH = "data/mimic_iv_processed_v2.npz"
NOTES_PATH      = "data/mimic_iv_notes.npz"
SPLIT_PATH      = "data/canonical_split.npz"
SAVE_PATH       = "results/mimic_multimodal_best.pt"
RESULTS_PATH    = "results/mimic_multimodal_results.json"
PREDS_PATH      = "results/preds_multimodal.npz"

BERT_MODEL      = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN         = 512
GRU_HIDDEN      = 128
GRU_LAYERS      = 2
LATENT_TS       = 64        # GRU output dim after projection
LATENT_NOTES    = 768       # BERT CLS dim (fixed)
FUSION_HIDDEN   = 256
BATCH_SIZE      = 32
EPOCHS          = 25
LR_BERT         = 1e-5      # lower LR for pretrained BERT (unused when frozen)
LR_OTHER        = 1e-4      # GRU + fusion head
WARMUP_EPOCHS   = 3
FREEZE_BERT     = True      # freeze BERT — use as fixed feature extractor
SEED            = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class MultimodalDataset(Dataset):
    def __init__(self, ts_data, texts, labels, tokenizer, max_len):
        """
        ts_data : np.ndarray (N, 48, 20)
        texts   : list/array of strings
        labels  : np.ndarray (N,)
        """
        self.ts        = ts_data
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ts    = torch.tensor(self.ts[idx], dtype=torch.float32)     # (48, 20)
        label = torch.tensor(self.labels[idx], dtype=torch.float32) # scalar

        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids      = enc["input_ids"].squeeze(0)       # (max_len,)
        attention_mask = enc["attention_mask"].squeeze(0)  # (max_len,)

        return ts, input_ids, attention_mask, label


# ── Model ─────────────────────────────────────────────────────────────────────
class GRUEncoder(nn.Module):
    """Encodes (batch, seq=48, features=20) → fixed latent vector (batch, latent_dim)."""
    def __init__(self, input_dim=20, hidden_dim=128, num_layers=2, latent_dim=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (batch, 48, 20)
        _, h = self.gru(x)    # h: (num_layers, batch, hidden)
        h = h[-1]             # last layer: (batch, hidden)
        return self.proj(h)   # (batch, latent_dim)


class MultimodalFusion(nn.Module):
    """
    Fuses GRU time-series encoding with ClinicalBERT CLS token.
    Input:
      ts         : (batch, 48, 20)  — structured vitals
      input_ids  : (batch, max_len) — tokenized notes
      attn_mask  : (batch, max_len) — attention mask
    Output:
      logits     : (batch,)         — raw (pre-sigmoid) mortality score
    """
    def __init__(self, bert_model_name, freeze_bert=False,
                 latent_ts=64, latent_notes=768, fusion_hidden=256):
        super().__init__()
        self.gru_enc = GRUEncoder(
            input_dim=20, hidden_dim=GRU_HIDDEN,
            num_layers=GRU_LAYERS, latent_dim=latent_ts
        )
        self.bert = AutoModel.from_pretrained(bert_model_name)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(latent_ts + latent_notes, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, ts, input_ids, attention_mask):
        z_ts    = self.gru_enc(ts)                              # (B, latent_ts)
        bert_out = self.bert(input_ids, attention_mask)
        z_notes = bert_out.last_hidden_state[:, 0, :]           # CLS: (B, 768)
        z       = torch.cat([z_ts, z_notes], dim=1)             # (B, latent_ts+768)
        logits  = self.classifier(z).squeeze(1)                 # (B,)
        return logits


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_and_align():
    """Load structured + notes, align on stay_id using canonical split."""
    print("Loading structured data...")
    s         = np.load(STRUCTURED_PATH)
    ts_values = s["values"]      # (74829, 48, 20)
    ts_ids    = s["stay_ids"]    # (74829,)
    ts_labels = s["labels"]      # (74829,)

    print("Loading notes data...")
    n           = np.load(NOTES_PATH, allow_pickle=True)
    note_ids    = n["stay_ids"]  # (47142,)
    note_texts  = n["texts"]     # (47142,) strings
    note_labels = n["labels"]    # (47142,)

    print("Loading canonical split...")
    sp = np.load(SPLIT_PATH)
    all_ids    = sp["all_ids"]
    all_labels = sp["all_labels"].astype(np.float32)
    train_idx  = sp["train_idx"]
    val_idx    = sp["val_idx"]
    test_idx   = sp["test_idx"]

    # Build index maps for O(1) lookup
    ts_id_map   = {int(sid): i for i, sid in enumerate(ts_ids)}
    note_id_map = {int(sid): i for i, sid in enumerate(note_ids)}

    # Align on the canonical intersection (all_ids)
    ts_aligned    = np.stack([ts_values[ts_id_map[int(sid)]] for sid in all_ids])
    texts_aligned = np.array([note_texts[note_id_map[int(sid)]] for sid in all_ids])

    print(f"Structured stays : {len(ts_ids):,}")
    print(f"Notes stays      : {len(note_ids):,}")
    print(f"Intersection     : {len(all_ids):,}")
    print(f"Aligned shape    : {ts_aligned.shape}")
    print(f"Mortality rate   : {all_labels.mean():.3f}")

    return ts_aligned, texts_aligned, all_labels, train_idx, val_idx, test_idx


# ── Training Loop ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for ts, input_ids, attn_mask, labels in loader:
            ts         = ts.to(device)
            input_ids  = input_ids.to(device)
            attn_mask  = attn_mask.to(device)
            labels     = labels.to(device)

            logits = model(ts, input_ids, attn_mask)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    return avg_loss, auroc, auprc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs("results", exist_ok=True)

    ts, texts, labels, tr_idx, val_idx, te_idx = load_and_align()
    print(f"Train: {len(tr_idx)}  Val: {len(val_idx)}  Test: {len(te_idx)}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    def make_loader(idxs, shuffle):
        ds = MultimodalDataset(ts[idxs], texts[idxs], labels[idxs],
                               tokenizer, MAX_LEN)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    train_loader = make_loader(tr_idx, shuffle=True)
    val_loader   = make_loader(val_idx, shuffle=False)
    test_loader  = make_loader(te_idx,  shuffle=False)

    model = MultimodalFusion(
        BERT_MODEL,
        freeze_bert=FREEZE_BERT,
        latent_ts=LATENT_TS,
        latent_notes=LATENT_NOTES,
        fusion_hidden=FUSION_HIDDEN
    ).to(device)

    # Class-weighted loss (mortality ~11%)
    pos_weight = torch.tensor([(1 - labels.mean()) / labels.mean()]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Differential LRs: BERT gets 10x lower LR than GRU + classifier
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(),       "lr": LR_BERT},
        {"params": model.gru_enc.parameters(),    "lr": LR_OTHER},
        {"params": model.classifier.parameters(), "lr": LR_OTHER},
    ], weight_decay=1e-2)

    # Linear warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_auroc = 0.0
    history = []

    print("\n── Training ──────────────────────────────────────────────────────")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_auroc, tr_auprc = run_epoch(
            model, train_loader, optimizer, criterion, train=True)
        va_loss, va_auroc, va_auprc = run_epoch(
            model, val_loader, optimizer, criterion, train=False)
        scheduler.step()

        print(f"Epoch {epoch:02d} | "
              f"Train loss {tr_loss:.4f}  AUROC {tr_auroc:.4f} | "
              f"Val loss {va_loss:.4f}  AUROC {va_auroc:.4f}  AUPRC {va_auprc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_auroc": tr_auroc,
            "val_loss":   va_loss, "val_auroc":   va_auroc, "val_auprc": va_auprc
        })

        if va_auroc > best_val_auroc:
            best_val_auroc = va_auroc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved best checkpoint (val AUROC {va_auroc:.4f})")

    # Final test eval using best checkpoint
    print("\n── Test Evaluation ───────────────────────────────────────────────")
    model.load_state_dict(torch.load(SAVE_PATH))
    te_loss, te_auroc, te_auprc = run_epoch(
        model, test_loader, None, criterion, train=False)
    print(f"Test AUROC : {te_auroc:.4f}")
    print(f"Test AUPRC : {te_auprc:.4f}")

    # Save predictions for bootstrap CI analysis
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ts_b, input_ids, attn_mask, labels_b in test_loader:
            ts_b       = ts_b.to(device)
            input_ids  = input_ids.to(device)
            attn_mask  = attn_mask.to(device)
            logits     = model(ts_b, input_ids, attn_mask)
            probs      = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels_b.numpy())
    np.savez(PREDS_PATH, preds=np.array(all_preds), labels=np.array(all_labels))
    print(f"Predictions saved → {PREDS_PATH}")

    history.append({
        "test_auroc": te_auroc,
        "test_auprc": te_auprc,
        "best_val_auroc": best_val_auroc
    })
    with open(RESULTS_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
