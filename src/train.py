"""
Training script for the Latent ODE ICU mortality model.

Usage:
    python src/train.py
    python src/train.py --epochs 50 --batch-size 64 --latent-dim 32
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

from dataset import PhysioNetDataset, collate_fn, N_VARS
from model import LatentODE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    avg_loss = total_loss / len(loader)
    return avg_loss, auroc, auprc


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    train_data, val_data, test_data = load_data(processed_dir)
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    train_loader = DataLoader(
        PhysioNetDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        PhysioNetDataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        PhysioNetDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Model
    model = LatentODE(
        input_dim=N_VARS,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        ode_hidden_dim=args.ode_hidden_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    # Class weights to handle 13.9% mortality imbalance
    pos_weight = torch.tensor([(1 - 0.139) / 0.139]).to(device)  # ~6.2x
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auroc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': [], 'val_auprc': []}

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
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

        # Save best model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            os.makedirs(os.path.join(results_dir, 'checkpoints'), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(results_dir, 'checkpoints', 'best_model.pt'))
            print(f"  --> New best AUROC: {best_auroc:.4f} (saved)")

    # Final test evaluation
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(results_dir, 'checkpoints', 'best_model.pt')))
    test_loss, test_auroc, test_auprc = evaluate(model, test_loader, device)
    print(f"\nTest Results:")
    print(f"  AUROC : {test_auroc:.4f}")
    print(f"  AUPRC : {test_auprc:.4f}")

    # Save history
    np.save(os.path.join(results_dir, 'history.npy'), history)
    print(f"\nHistory saved to results/history.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--ode-hidden-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--kl-weight', type=float, default=0.1)
    args = parser.parse_args()
    train(args)
