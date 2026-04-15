"""
Multi-task ClinicalBERT for ICU mortality prediction.

Two output heads sharing Bio_ClinicalBERT encoder:
  Head A: mortality prediction (primary task)
  Head B: CMO/DNR detection (auxiliary task, pseudo-labeled from note keywords)

After training, produces:
  - Correlation analysis between mortality and CMO/DNR predictions
  - Stratified AUROC by CMO/DNR flag
  - "Clean" subgroup AUROC (patients with low CMO/DNR signal)

Requirements:
    pip install transformers==4.44.2 scikit-learn scipy
"""

import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME    = "emilyalsentzer/Bio_ClinicalBERT"
NOTES_NPZ     = "data/mimic_iv_notes.npz"
SPLIT_NPZ     = "data/canonical_split.npz"
OUT_DIR       = "results"
CKPT_PATH     = "results/bert_multitask_best.pt"
PREDS_PATH    = "results/preds_bert_multitask.npz"

MAX_LEN       = 512
BATCH_SIZE    = 32
GRAD_ACCUM    = 1           # effective batch = 32
EPOCHS        = 15
LR_BERT       = 2e-5        # BERT encoder (used after unfreeze)
LR_HEADS      = 1e-3        # classification heads
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
CMO_LAMBDA    = 0.3         # weight of CMO/DNR loss
DROPOUT       = 0.3
FREEZE_EPOCHS = 3           # freeze BERT encoder for first N epochs (train heads only)
SEED          = 42

# CMO/DNR threshold for "clean" subgroup
CMO_CLEAN_THRESH = 0.3

# ── CMO/DNR keyword patterns ─────────────────────────────────────────────────
CMO_DNR_PATTERNS = [
    r"\bcomfort\s+measures\s+only\b",
    r"\bcomfort\s+care\s+only\b",
    r"\bcmo\b",
    r"\bdnr\b",
    r"\bdo\s+not\s+resuscitate\b",
    r"\bdo\s+not\s+intubate\b",
    r"\bdni\b",
    r"\bwithdrawal\s+of\s+(life.sustaining|care|support)\b",
    r"\bwithdraw\s+(life|care|support)\b",
    r"\bhospice\b",
    r"\bpalliative\s+care\s+only\b",
    r"\bcode\s+status.*changed.*to.*(dnr|comfort|cmo)\b",
    r"\bgoals\s+of\s+care.*(comfort|withdraw|palliat)\b",
]
CMO_REGEX = re.compile("|".join(CMO_DNR_PATTERNS), re.IGNORECASE)

def make_cmo_label(text):
    return 1 if CMO_REGEX.search(text) else 0

# ── Seed ─────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading notes and split...")
notes_data  = np.load(NOTES_NPZ, allow_pickle=True)
notes_ids   = notes_data["stay_ids"]
notes_texts = notes_data["texts"]

split_data  = np.load(SPLIT_NPZ, allow_pickle=True)
train_ids   = set(split_data["train_ids"].tolist())
val_ids     = set(split_data["val_ids"].tolist())
test_ids    = set(split_data["test_ids"].tolist())
id2label    = dict(zip(split_data["all_ids"].tolist(), split_data["all_labels"].tolist()))

def build_split(id_set):
    texts, mort_labels, cmo_labels, sids = [], [], [], []
    for sid, txt in zip(notes_ids.tolist(), notes_texts.tolist()):
        if sid in id_set:
            txt = str(txt)
            texts.append(txt)
            mort_labels.append(int(id2label[sid]))
            cmo_labels.append(make_cmo_label(txt))
            sids.append(sid)
    return texts, mort_labels, cmo_labels, sids

train_texts, train_mort, train_cmo, _ = build_split(train_ids)
val_texts,   val_mort,   val_cmo,   _ = build_split(val_ids)
test_texts,  test_mort,  test_cmo,  test_sids = build_split(test_ids)

print(f"Train: {len(train_texts)}  Val: {len(val_texts)}  Test: {len(test_texts)}")
print(f"Train mortality rate:  {np.mean(train_mort):.3f}")
print(f"Train CMO/DNR rate:    {np.mean(train_cmo):.3f}")
print(f"Val   CMO/DNR rate:    {np.mean(val_cmo):.3f}")
print(f"Test  CMO/DNR rate:    {np.mean(test_cmo):.3f}")

# Cross-tab: mortality vs CMO/DNR
train_m = np.array(train_mort)
train_c = np.array(train_cmo)
print(f"\nTrain cross-tab:")
print(f"  CMO/DNR=1 & died:    {((train_c==1) & (train_m==1)).sum()}")
print(f"  CMO/DNR=1 & survived:{((train_c==1) & (train_m==0)).sum()}")
print(f"  CMO/DNR=0 & died:    {((train_c==0) & (train_m==1)).sum()}")
print(f"  CMO/DNR=0 & survived:{((train_c==0) & (train_m==0)).sum()}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Dataset ───────────────────────────────────────────────────────────────────
class NotesDataset(Dataset):
    def __init__(self, texts, mort_labels, cmo_labels):
        self.texts       = texts
        self.mort_labels = mort_labels
        self.cmo_labels  = cmo_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "mort_label":     torch.tensor(self.mort_labels[idx], dtype=torch.float32),
            "cmo_label":      torch.tensor(self.cmo_labels[idx],  dtype=torch.float32),
        }

train_ds = NotesDataset(train_texts, train_mort, train_cmo)
val_ds   = NotesDataset(val_texts,   val_mort,   val_cmo)
test_ds  = NotesDataset(test_texts,  test_mort,  test_cmo)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Dual-head model ───────────────────────────────────────────────────────────
class DualHeadBERT(nn.Module):
    def __init__(self, model_name, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.mortality_head = nn.Linear(hidden, 1)
        self.cmo_head       = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])  # [CLS] token
        mort_logit = self.mortality_head(cls).squeeze(-1)
        cmo_logit  = self.cmo_head(cls).squeeze(-1)
        return mort_logit, cmo_logit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = DualHeadBERT(MODEL_NAME, dropout=DROPOUT).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,} total")

# ── Freeze BERT initially ─────────────────────────────────────────────────────
for p in model.bert.parameters():
    p.requires_grad = False
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Phase 1 (BERT frozen): {train_params:,} trainable")

# ── Optimizer (differential LR) ──────────────────────────────────────────────
bert_params = list(model.bert.parameters())
head_params = list(model.mortality_head.parameters()) + list(model.cmo_head.parameters())

optimizer = torch.optim.AdamW([
    {"params": bert_params, "lr": LR_BERT},
    {"params": head_params, "lr": LR_HEADS},
], weight_decay=WEIGHT_DECAY)

total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

mort_criterion = nn.BCEWithLogitsLoss()
cmo_criterion  = nn.BCEWithLogitsLoss()

# ── Eval helper ───────────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    mort_probs, cmo_probs = [], []
    mort_true,  cmo_true  = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mort_logit, cmo_logit = model(ids, mask)
            mort_probs.extend(torch.sigmoid(mort_logit).cpu().numpy())
            cmo_probs.extend(torch.sigmoid(cmo_logit).cpu().numpy())
            mort_true.extend(batch["mort_label"].numpy())
            cmo_true.extend(batch["cmo_label"].numpy())

    mort_probs = np.array(mort_probs)
    cmo_probs  = np.array(cmo_probs)
    mort_true  = np.array(mort_true)
    cmo_true   = np.array(cmo_true)

    mort_auroc = roc_auc_score(mort_true, mort_probs)
    mort_auprc = average_precision_score(mort_true, mort_probs)

    if len(np.unique(cmo_true)) > 1:
        cmo_auroc = roc_auc_score(cmo_true, cmo_probs)
    else:
        cmo_auroc = float('nan')

    return mort_auroc, mort_auprc, cmo_auroc, mort_probs, cmo_probs, mort_true, cmo_true

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_auroc = 0.0
patience_counter = 0
PATIENCE = 4
log = []

print(f"\nStarting multi-task training ({EPOCHS} epochs, BERT frozen for first {FREEZE_EPOCHS})...")
for epoch in range(1, EPOCHS + 1):
    # Unfreeze BERT after FREEZE_EPOCHS
    if epoch == FREEZE_EPOCHS + 1:
        print(f"\n  *** Unfreezing BERT encoder at epoch {epoch} ***")
        for p in model.bert.parameters():
            p.requires_grad = True
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Phase 2 (BERT unfrozen): {train_params:,} trainable\n")

    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        ids      = batch["input_ids"].to(device)
        mask     = batch["attention_mask"].to(device)
        mort_lbl = batch["mort_label"].to(device)
        cmo_lbl  = batch["cmo_label"].to(device)

        mort_logit, cmo_logit = model(ids, mask)
        loss_mort = mort_criterion(mort_logit, mort_lbl)
        loss_cmo  = cmo_criterion(cmo_logit, cmo_lbl)
        loss      = (loss_mort + CMO_LAMBDA * loss_cmo) / GRAD_ACCUM

        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    mort_auroc, mort_auprc, cmo_auroc, _, _, _, _ = evaluate(val_loader)
    print(f"Epoch {epoch}: loss={avg_loss:.4f}  val_mort_AUROC={mort_auroc:.4f}  "
          f"val_cmo_AUROC={cmo_auroc:.4f}  val_mort_AUPRC={mort_auprc:.4f}")

    log.append({"epoch": epoch, "loss": avg_loss,
                "val_mort_auroc": mort_auroc, "val_cmo_auroc": cmo_auroc,
                "val_mort_auprc": mort_auprc})

    if mort_auroc > best_val_auroc:
        best_val_auroc = mort_auroc
        patience_counter = 0
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"  ✓ New best val mortality AUROC: {mort_auroc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

# ── Test evaluation ───────────────────────────────────────────────────────────
print("\nLoading best checkpoint for test eval...")
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

mort_auroc, mort_auprc, cmo_auroc, mort_probs, cmo_probs, mort_true, cmo_true = evaluate(test_loader)

print(f"\n{'='*60}")
print(f"TEST RESULTS")
print(f"{'='*60}")
print(f"Mortality AUROC:  {mort_auroc:.4f}   AUPRC: {mort_auprc:.4f}")
print(f"CMO/DNR   AUROC:  {cmo_auroc:.4f}")

# ── Correlation analysis ──────────────────────────────────────────────────────
r, p_val = pearsonr(mort_probs, cmo_probs)
print(f"\n--- Correlation Analysis ---")
print(f"Pearson r between mortality and CMO/DNR predictions: {r:.3f} (p={p_val:.2e})")
if r > 0.7:
    interp = "HIGH — model heavily using code status as mortality proxy"
elif r > 0.4:
    interp = "MODERATE — partial overlap between mortality and code status signals"
else:
    interp = "LOW — mostly independent signals"
print(f"Interpretation: {interp}")

# ── Stratified analysis by keyword flag ───────────────────────────────────────
print(f"\n--- Stratified AUROC by Actual CMO/DNR Keyword Flag ---")
cmo_flag     = np.array(cmo_true)
cmo_mask     = cmo_flag == 1
non_cmo_mask = cmo_flag == 0

auroc_cmo = None
auroc_non_cmo = None

if cmo_mask.sum() > 10 and len(np.unique(mort_true[cmo_mask])) > 1:
    auroc_cmo = roc_auc_score(mort_true[cmo_mask], mort_probs[cmo_mask])
    print(f"  CMO/DNR patients     (n={cmo_mask.sum():,}, mortality={mort_true[cmo_mask].mean():.3f}): AUROC = {auroc_cmo:.4f}")
else:
    print(f"  CMO/DNR patients     (n={cmo_mask.sum():,}): insufficient data for AUROC")

if non_cmo_mask.sum() > 10 and len(np.unique(mort_true[non_cmo_mask])) > 1:
    auroc_non_cmo = roc_auc_score(mort_true[non_cmo_mask], mort_probs[non_cmo_mask])
    print(f"  Non-CMO/DNR patients (n={non_cmo_mask.sum():,}, mortality={mort_true[non_cmo_mask].mean():.3f}): AUROC = {auroc_non_cmo:.4f}")
else:
    print(f"  Non-CMO/DNR patients (n={non_cmo_mask.sum():,}): insufficient data for AUROC")

# ── "Clean" subgroup analysis ─────────────────────────────────────────────────
print(f"\n--- 'Clean' Subgroup (predicted CMO/DNR prob < {CMO_CLEAN_THRESH}) ---")
clean_mask = cmo_probs < CMO_CLEAN_THRESH
auroc_clean = None
auprc_clean = None

if clean_mask.sum() > 10 and len(np.unique(mort_true[clean_mask])) > 1:
    auroc_clean = roc_auc_score(mort_true[clean_mask], mort_probs[clean_mask])
    auprc_clean = average_precision_score(mort_true[clean_mask], mort_probs[clean_mask])
    print(f"  n={clean_mask.sum():,}, mortality rate={mort_true[clean_mask].mean():.3f}")
    print(f"  Mortality AUROC = {auroc_clean:.4f}   AUPRC = {auprc_clean:.4f}")
    print(f"  (Model's performance when predicting from physiology, not code status)")
else:
    print(f"  n={clean_mask.sum():,}: insufficient data")

# ── "Confounded" subgroup: high CMO/DNR prob ──────────────────────────────────
print(f"\n--- 'Confounded' Subgroup (predicted CMO/DNR prob >= {1-CMO_CLEAN_THRESH}) ---")
conf_mask = cmo_probs >= (1 - CMO_CLEAN_THRESH)
if conf_mask.sum() > 10 and len(np.unique(mort_true[conf_mask])) > 1:
    auroc_conf = roc_auc_score(mort_true[conf_mask], mort_probs[conf_mask])
    print(f"  n={conf_mask.sum():,}, mortality rate={mort_true[conf_mask].mean():.3f}")
    print(f"  Mortality AUROC = {auroc_conf:.4f}")
    print(f"  (High code status signal — model may be exploiting care decisions)")

print(f"\n{'='*60}")

# ── Save predictions and results ──────────────────────────────────────────────
np.savez(PREDS_PATH,
         mort_probs=mort_probs, mort_true=mort_true,
         cmo_probs=cmo_probs,   cmo_true=cmo_true,
         stay_ids=np.array(test_sids))
print(f"Predictions saved to {PREDS_PATH}")

results = {
    "config": {
        "model": MODEL_NAME, "cmo_lambda": CMO_LAMBDA,
        "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "lr_bert": LR_BERT, "lr_heads": LR_HEADS,
        "dropout": DROPOUT,
    },
    "test": {
        "mort_auroc": float(mort_auroc),
        "mort_auprc": float(mort_auprc),
        "cmo_auroc":  float(cmo_auroc),
    },
    "correlation": {
        "pearson_r": float(r),
        "p_value":   float(p_val),
        "interpretation": interp,
    },
    "stratified": {
        "cmo_patients_n":       int(cmo_mask.sum()),
        "cmo_patients_auroc":   float(auroc_cmo)     if auroc_cmo     else None,
        "non_cmo_patients_n":   int(non_cmo_mask.sum()),
        "non_cmo_patients_auroc": float(auroc_non_cmo) if auroc_non_cmo else None,
    },
    "clean_subgroup": {
        "threshold": CMO_CLEAN_THRESH,
        "n":     int(clean_mask.sum()),
        "auroc": float(auroc_clean) if auroc_clean else None,
        "auprc": float(auprc_clean) if auprc_clean else None,
    },
    "training_log": log,
}

with open(os.path.join(OUT_DIR, "bert_multitask_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUT_DIR}/bert_multitask_results.json")

print("\nDone.")
