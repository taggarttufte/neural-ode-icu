"""
Multi-task LoRA fine-tuning of BioMistral-7B for ICU mortality prediction.

Two output heads sharing the same encoder:
  Head A: mortality prediction (primary task)
  Head B: CMO/DNR prediction (auxiliary task, pseudo-labeled from note keywords)

After training, produces:
  - Correlation analysis between mortality and CMO/DNR predictions
  - Stratified AUROC by CMO/DNR flag (did model cheat vs. learn physiology?)
  - "Clean" subgroup AUROC: patients with low CMO/DNR signal only

Requirements:
    pip install peft==0.13.2 accelerate transformers datasets scikit-learn
"""

import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME    = "/home/j19w245/models/biomistral-7b"
NOTES_NPZ     = "data/mimic_iv_notes.npz"
SPLIT_NPZ     = "data/canonical_split.npz"
OUT_DIR       = "results"
CKPT_DIR      = "results/lora_multitask_best"
PREDS_PATH    = "results/preds_lora_multitask.npz"

MAX_LEN       = 512
BATCH_SIZE    = 8
GRAD_ACCUM    = 4           # effective batch = 32
EPOCHS        = 6
LR            = 2e-4
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
CMO_LAMBDA    = 0.3         # weight of CMO/DNR loss relative to mortality loss
SEED          = 42

# LoRA
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
TARGET_MODS   = ["q_proj", "v_proj", "k_proj", "o_proj"]

# CMO/DNR threshold for "clean" subgroup analysis
CMO_CLEAN_THRESH = 0.3      # patients with pred CMO/DNR prob < this are "clean"

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
    texts, mort_labels, cmo_labels = [], [], []
    for sid, txt in zip(notes_ids.tolist(), notes_texts.tolist()):
        if sid in id_set:
            txt = str(txt)
            texts.append(txt)
            mort_labels.append(int(id2label[sid]))
            cmo_labels.append(make_cmo_label(txt))
    return texts, mort_labels, cmo_labels

train_texts, train_mort, train_cmo = build_split(train_ids)
val_texts,   val_mort,   val_cmo   = build_split(val_ids)
test_texts,  test_mort,  test_cmo  = build_split(test_ids)

print(f"Train: {len(train_texts)}  Val: {len(val_texts)}  Test: {len(test_texts)}")
print(f"Train mortality rate:  {np.mean(train_mort):.3f}")
print(f"Train CMO/DNR rate:    {np.mean(train_cmo):.3f}")
print(f"Test  CMO/DNR rate:    {np.mean(test_cmo):.3f}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ── Dual-head model ───────────────────────────────────────────────────────────
class DualHeadLoRA(nn.Module):
    """
    BioMistral-7B encoder (LoRA) with two binary classification heads.
    Uses last non-padding token hidden state as the sequence representation
    (standard approach for decoder-only models).
    """
    def __init__(self, model_name, lora_cfg):
        super().__init__()
        base = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.encoder       = get_peft_model(base, lora_cfg)
        self.encoder.print_trainable_parameters()
        hidden_size        = self.encoder.config.hidden_size     # 4096 for Mistral-7B
        self.mortality_head = nn.Linear(hidden_size, 1)
        self.cmo_head       = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Last non-padding token index per sample
        last_idx = attention_mask.sum(dim=1) - 1          # (B,)
        hidden = out.last_hidden_state[
            torch.arange(input_ids.size(0), device=input_ids.device), last_idx
        ].float()                                          # (B, hidden_size)
        mort_logit = self.mortality_head(hidden).squeeze(-1)   # (B,)
        cmo_logit  = self.cmo_head(hidden).squeeze(-1)         # (B,)
        return mort_logit, cmo_logit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODS,
    bias="none",
)

print("Loading BioMistral-7B with dual-head LoRA...")
model = DualHeadLoRA(MODEL_NAME, lora_cfg)

# Move classification heads to same device as encoder output
# (encoder uses device_map="auto", so outputs land on cuda:0)
model.mortality_head = model.mortality_head.to(device)
model.cmo_head       = model.cmo_head.to(device)

# ── Loss and optimizer ────────────────────────────────────────────────────────
mort_criterion = nn.BCEWithLogitsLoss()
cmo_criterion  = nn.BCEWithLogitsLoss()

trainable_params = (
    [p for p in model.encoder.parameters() if p.requires_grad]
    + list(model.mortality_head.parameters())
    + list(model.cmo_head.parameters())
)
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

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

    # NaN guard
    mort_probs = np.nan_to_num(mort_probs, nan=0.5)
    cmo_probs  = np.nan_to_num(cmo_probs,  nan=0.5)

    mort_auroc = roc_auc_score(mort_true, mort_probs)
    mort_auprc = average_precision_score(mort_true, mort_probs)
    cmo_auroc  = roc_auc_score(cmo_true,  cmo_probs)

    return mort_auroc, mort_auprc, cmo_auroc, mort_probs, cmo_probs, mort_true, cmo_true

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_auroc = 0.0
log = []

print("\nStarting multi-task training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        ids        = batch["input_ids"].to(device)
        mask       = batch["attention_mask"].to(device)
        mort_lbl   = batch["mort_label"].to(device)
        cmo_lbl    = batch["cmo_label"].to(device)

        mort_logit, cmo_logit = model(ids, mask)
        loss_mort  = mort_criterion(mort_logit, mort_lbl)
        loss_cmo   = cmo_criterion(cmo_logit,  cmo_lbl)
        loss       = (loss_mort + CMO_LAMBDA * loss_cmo) / GRAD_ACCUM

        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    mort_auroc, mort_auprc, cmo_auroc, _, _, _, _ = evaluate(val_loader)
    print(f"Epoch {epoch}: loss={avg_loss:.4f}  val_mort_AUROC={mort_auroc:.4f}  "
          f"val_cmo_AUROC={cmo_auroc:.4f}  val_mort_AUPRC={mort_auprc:.4f}")

    log.append({"epoch": epoch, "loss": avg_loss,
                 "val_mort_auroc": mort_auroc, "val_cmo_auroc": cmo_auroc})

    if mort_auroc > best_val_auroc:
        best_val_auroc = mort_auroc
        model.encoder.save_pretrained(CKPT_DIR)
        torch.save(
            {"mortality_head": model.mortality_head.state_dict(),
             "cmo_head":       model.cmo_head.state_dict()},
            os.path.join(CKPT_DIR, "heads.pt"),
        )
        print(f"  ✓ New best val mortality AUROC: {mort_auroc:.4f} — saved")

# ── Test evaluation ───────────────────────────────────────────────────────────
print("\nLoading best checkpoint for test eval...")
from peft import PeftModel

base_reload = AutoModel.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
model.encoder = PeftModel.from_pretrained(base_reload, CKPT_DIR)
heads = torch.load(os.path.join(CKPT_DIR, "heads.pt"), map_location=device)
model.mortality_head.load_state_dict(heads["mortality_head"])
model.cmo_head.load_state_dict(heads["cmo_head"])

mort_auroc, mort_auprc, cmo_auroc, mort_probs, cmo_probs, mort_true, cmo_true = evaluate(test_loader)

print(f"\n{'='*60}")
print(f"Test mortality AUROC:  {mort_auroc:.4f}  AUPRC: {mort_auprc:.4f}")
print(f"Test CMO/DNR AUROC:    {cmo_auroc:.4f}")

# ── Correlation analysis ──────────────────────────────────────────────────────
r, p = pearsonr(mort_probs, cmo_probs)
print(f"\nCorrelation between mortality and CMO/DNR predictions:")
print(f"  Pearson r = {r:.3f}  (p = {p:.2e})")
print(f"  Interpretation: {'high — model using code status as shortcut'  if r > 0.7 else"
      f"                   'moderate — partial overlap'                  if r > 0.4 else"
      f"                   'low — mostly independent signals'}")

# ── Stratified analysis ───────────────────────────────────────────────────────
print(f"\nStratified AUROC by actual CMO/DNR keyword flag:")
cmo_flag     = np.array(test_cmo)
cmo_mask     = cmo_flag == 1
non_cmo_mask = cmo_flag == 0

if cmo_mask.sum() > 10 and np.unique(mort_true[cmo_mask]).size > 1:
    auroc_cmo = roc_auc_score(mort_true[cmo_mask], mort_probs[cmo_mask])
    print(f"  CMO/DNR patients     (n={cmo_mask.sum()}): AUROC = {auroc_cmo:.4f}")

if non_cmo_mask.sum() > 10 and np.unique(mort_true[non_cmo_mask]).size > 1:
    auroc_non_cmo = roc_auc_score(mort_true[non_cmo_mask], mort_probs[non_cmo_mask])
    print(f"  Non-CMO/DNR patients (n={non_cmo_mask.sum()}): AUROC = {auroc_non_cmo:.4f}")

# ── "Clean" subgroup: low predicted CMO/DNR probability ──────────────────────
print(f"\n'Clean' subgroup (predicted CMO/DNR prob < {CMO_CLEAN_THRESH}):")
clean_mask = cmo_probs < CMO_CLEAN_THRESH
if clean_mask.sum() > 10 and np.unique(mort_true[clean_mask]).size > 1:
    auroc_clean = roc_auc_score(mort_true[clean_mask], mort_probs[clean_mask])
    auprc_clean = average_precision_score(mort_true[clean_mask], mort_probs[clean_mask])
    print(f"  n={clean_mask.sum()}, mortality rate={mort_true[clean_mask].mean():.3f}")
    print(f"  AUROC = {auroc_clean:.4f}   AUPRC = {auprc_clean:.4f}")
    print(f"  (This is the model's performance on pure physiology cases)")

print(f"{'='*60}")

# ── Save predictions and results ─────────────────────────────────────────────
np.savez(PREDS_PATH,
         mort_probs=mort_probs, mort_true=mort_true,
         cmo_probs=cmo_probs,   cmo_true=cmo_true)
print(f"\nPredictions saved to {PREDS_PATH}")

results = {
    "config": {"lora_r": LORA_R, "cmo_lambda": CMO_LAMBDA,
                "max_len": MAX_LEN, "epochs": EPOCHS},
    "test": {"mort_auroc": mort_auroc, "mort_auprc": mort_auprc,
              "cmo_auroc": cmo_auroc},
    "correlation": {"pearson_r": float(r), "p_value": float(p)},
    "stratified": {
        "cmo_patients_auroc":     float(auroc_cmo)     if cmo_mask.sum() > 10     else None,
        "non_cmo_patients_auroc": float(auroc_non_cmo) if non_cmo_mask.sum() > 10 else None,
        "clean_subgroup_auroc":   float(auroc_clean)   if clean_mask.sum() > 10   else None,
    },
    "training_log": log,
}
with open(os.path.join(OUT_DIR, "lora_multitask_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Done.")
