"""
LoRA fine-tuning of BioMistral-7B for ICU mortality prediction.
Uses clinical notes from mimic_iv_notes.npz (first-48h, discharge excluded).
Canonical train/val/test split from canonical_split.npz.

Requirements:
    pip install peft accelerate bitsandbytes transformers datasets
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "/home/j19w245/models/biomistral-7b"
NOTES_NPZ    = "data/mimic_iv_notes.npz"
SPLIT_NPZ    = "data/canonical_split.npz"
OUT_DIR      = "results"
CKPT_PATH    = "results/mimic_lora_best.pt"
PREDS_PATH   = "results/preds_lora.npz"

MAX_LEN      = 512
BATCH_SIZE   = 8          # fits A40 48GB in fp16 with rank-16 LoRA
GRAD_ACCUM   = 4          # effective batch = 32
EPOCHS       = 6
LR           = 2e-4       # LoRA params only (base frozen)
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED         = 42

# LoRA hyperparams
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODS  = ["q_proj", "v_proj", "k_proj", "o_proj"]   # Llama attention

# ── Seed ────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading notes and split...")
notes_data = np.load(NOTES_NPZ, allow_pickle=True)
notes_ids   = notes_data["stay_ids"]          # (N,)
notes_texts = notes_data["texts"]             # (N,) array of strings

split_data  = np.load(SPLIT_NPZ, allow_pickle=True)
train_ids   = set(split_data["train_ids"].tolist())
val_ids     = set(split_data["val_ids"].tolist())
test_ids    = set(split_data["test_ids"].tolist())
all_ids_arr = split_data["all_ids"]
all_lbls    = split_data["all_labels"]
id2label    = dict(zip(all_ids_arr.tolist(), all_lbls.tolist()))

# Build per-split lists
def build_split(id_set):
    texts, labels = [], []
    for sid, txt in zip(notes_ids.tolist(), notes_texts.tolist()):
        if sid in id_set:
            texts.append(str(txt))
            labels.append(int(id2label[sid]))
    return texts, labels

train_texts, train_labels = build_split(train_ids)
val_texts,   val_labels   = build_split(val_ids)
test_texts,  test_labels  = build_split(test_ids)

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
print(f"Train mortality: {np.mean(train_labels):.3f}")

# ── Tokenizer ────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Dataset ──────────────────────────────────────────────────────────────────
class NotesDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts
        self.labels = labels

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
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_ds = NotesDataset(train_texts, train_labels)
val_ds   = NotesDataset(val_texts,   val_labels)
test_ds  = NotesDataset(test_texts,  test_labels)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ── Model + LoRA ─────────────────────────────────────────────────────────────
print("Loading BioMistral-7B and applying LoRA...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
base_model.config.pad_token_id = tokenizer.pad_token_id

lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODS,
    bias="none",
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

# ── Optimizer / scheduler ────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
total_steps   = (len(train_loader) // GRAD_ACCUM) * EPOCHS
warmup_steps  = int(total_steps * WARMUP_RATIO)
scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# Pos weight for class imbalance
pos_weight = torch.tensor(
    [(1 - np.mean(train_labels)) / np.mean(train_labels)], dtype=torch.float32
).to(device)
criterion = torch.nn.CrossEntropyLoss()   # model returns logits for 2 classes

# ── Eval helper ──────────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    all_probs, all_labels_out = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["labels"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels_out.extend(lbl.cpu().numpy())
    all_probs  = np.array(all_probs)
    all_labels_out = np.array(all_labels_out)
    nan_count = np.isnan(all_probs).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN probabilities detected — replacing with 0.5")
        all_probs = np.nan_to_num(all_probs, nan=0.5)
    auroc = roc_auc_score(all_labels_out, all_probs)
    auprc = average_precision_score(all_labels_out, all_probs)
    return auroc, auprc, all_probs, all_labels_out

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_auroc = 0.0
results_log    = []

print("\nStarting training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    step_count = 0

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["labels"].to(device)

        out  = model(input_ids=ids, attention_mask=mask, labels=lbl)
        loss = out.loss / GRAD_ACCUM
        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step_count += 1

    avg_loss = total_loss / len(train_loader)
    val_auroc, val_auprc, _, _ = evaluate(val_loader)
    print(f"Epoch {epoch}: loss={avg_loss:.4f}  val_AUROC={val_auroc:.4f}  val_AUPRC={val_auprc:.4f}")

    results_log.append({
        "epoch": epoch,
        "train_loss": avg_loss,
        "val_auroc": val_auroc,
        "val_auprc": val_auprc,
    })

    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        model.save_pretrained(CKPT_PATH.replace(".pt", "_adapter"))
        print(f"  ✓ New best val AUROC: {val_auroc:.4f} — adapter saved")

# ── Test eval ────────────────────────────────────────────────────────────────
print("\nLoading best adapter for test eval...")
from peft import PeftModel
best_model = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, torch_dtype=torch.bfloat16, device_map="auto"
    ),
    CKPT_PATH.replace(".pt", "_adapter"),
)
best_model.config.pad_token_id = tokenizer.pad_token_id

test_auroc, test_auprc, test_probs, test_lbl = evaluate(test_loader)
print(f"\nTest AUROC: {test_auroc:.4f}  Test AUPRC: {test_auprc:.4f}")

np.savez(PREDS_PATH, probs=test_probs, labels=test_lbl)
print(f"Predictions saved to {PREDS_PATH}")

# Save training log
with open(os.path.join(OUT_DIR, "lora_training_log.json"), "w") as f:
    json.dump({"config": {
        "model": MODEL_NAME, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
    }, "log": results_log, "test_auroc": test_auroc, "test_auprc": test_auprc}, f, indent=2)

print("Done.")
