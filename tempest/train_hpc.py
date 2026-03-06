"""
train_hpc.py — Tempest HPC version of train.py

Changes from local version:
  - --data-dir argument (MIMIC data lives on secure storage, not in repo)
  - --output-dir argument (save results to scratch or project dir)
  - num_workers=4 (Linux multiprocessing works correctly)
  - XGBoost device flag respected
  - No Windows-specific workarounds

Usage on Tempest:
  python train_hpc.py --data-dir /scratch/NETID/mimic-iv/processed \
                      --output-dir /scratch/NETID/results \
                      --epochs 50 --batch-size 64 --latent-dim 32 \
                      --early-stop-patience 10
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import sys

# Add src to path so model/dataset imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')
from dataset import PhysioNetDataset, collate_fn, N_VARS
from model import LatentODE


def load_data(processed_dir):
    with open(os.path.join(processed_dir, 'train.pkl'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(processed_dir, 'val.pkl'), 'rb') as f:
        val = pickle.load(f)
    with open(os.path.join(processed_dir, 'test.pkl'), 'rb') as f:
        test = pickle.load(f)
    return train, val, test


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            times = batch['times'].to(device)
            values = batch['values'].to(device)
            mask = batch['mask'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            labels = batch['labels'].to(device)
            probs = model.predict_proba(times, values, mask, seq_lengths)
            loss = nn.BCELoss()(probs, labels)
            total_loss += loss.item()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    return total_loss / len(loader), auroc, auprc


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    print("Loading data...")
    train_data, val_data, test_data = load_data(args.data_dir)
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # num_workers=4 works on Linux (unlike Windows)
    train_loader = DataLoader(PhysioNetDataset(train_data), batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_fn,
                               num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(PhysioNetDataset(val_data),   batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn,
                               num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(PhysioNetDataset(test_data),  batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn,
                               num_workers=4, pin_memory=True, persistent_workers=True)

    model = LatentODE(input_dim=N_VARS, hidden_dim=args.hidden_dim,
                       latent_dim=args.latent_dim,
                       ode_hidden_dim=args.ode_hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    pos_weight = torch.tensor([(1 - 0.139) / 0.139]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auroc = 0.0
    start_epoch = 1
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': [], 'val_auprc': []}

    checkpoint_path = os.path.join(checkpoints_dir, 'latest_model.pt')
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_auroc = checkpoint['best_auroc']
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        history = checkpoint['history']
        print(f"Resumed from epoch {checkpoint['epoch']} "
              f"(best AUROC: {best_auroc:.4f}, no-improve streak: {epochs_no_improve})")

    print(f"\nTraining for {args.epochs} epochs (starting at {start_epoch})...")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            times = batch['times'].to(device)
            values = batch['values'].to(device)
            mask = batch['mask'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits, kl_loss = model(times, values, mask, seq_lengths)
            recon_loss = bce_loss(logits, labels)
            loss = recon_loss + args.kl_weight * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        val_loss, val_auroc, val_auprc = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        history['val_auprc'].append(val_auprc)

        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | AUROC: {val_auroc:.4f} | AUPRC: {val_auprc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best_model.pt'))
            print(f"  --> New best AUROC: {best_auroc:.4f} (saved)")
        else:
            epochs_no_improve += 1

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_auroc': best_auroc,
            'epochs_no_improve': epochs_no_improve,
            'history': history,
        }, checkpoint_path)

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"\nEarly stopping: no improvement for {epochs_no_improve} epochs.")
            break

    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'best_model.pt'),
                                     map_location=device, weights_only=True))
    test_loss, test_auroc, test_auprc = evaluate(model, test_loader, device)
    print(f"\nTest Results:")
    print(f"  AUROC : {test_auroc:.4f}")
    print(f"  AUPRC : {test_auprc:.4f}")

    np.save(os.path.join(args.output_dir, 'history.npy'), history)
    print(f"\nHistory saved to {args.output_dir}/history.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   type=str, required=True,
                        help='Path to processed data directory (train/val/test.pkl)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save checkpoints and history')
    parser.add_argument('--epochs',              type=int,   default=50)
    parser.add_argument('--batch-size',          type=int,   default=64)
    parser.add_argument('--latent-dim',          type=int,   default=32)
    parser.add_argument('--hidden-dim',          type=int,   default=64)
    parser.add_argument('--ode-hidden-dim',      type=int,   default=64)
    parser.add_argument('--lr',                  type=float, default=1e-3)
    parser.add_argument('--kl-weight',           type=float, default=0.1)
    parser.add_argument('--early-stop-patience', type=int,   default=10)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    train(args)
