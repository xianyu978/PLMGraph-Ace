#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Public training script for prepared graph inputs.

Notes
-----
1. This public script assumes that graph samples have already been prepared offline.
2. It does NOT include the full raw-data preprocessing pipeline.
3. Model selection is based on validation AUC.
4. Final metrics are computed with a fixed probability threshold (default: 0.5).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool


# =========================
# Utilities
# =========================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(gpu_id=None):
    if torch.cuda.is_available():
        if gpu_id is None:
            return torch.device("cuda:0")
        return torch.device(f"cuda:{int(gpu_id)}")
    return torch.device("cpu")


def rbf_expand(distance: np.ndarray, K: int = 6, gamma: float = 1.0):
    centers = np.arange(1, K + 1, dtype=np.float32)
    dist = distance[..., None].astype(np.float32)
    return np.exp(-gamma * (dist - centers) ** 2)


def build_graph_seq_rbf(
    feat_x: torch.Tensor,
    k_hop: int = 2,
    rbf_K: int = 6,
    rbf_gamma: float = 1.0
):
    L = feat_x.size(0)
    es, et, attrs = [], [], []

    for i in range(L):
        for d in range(1, k_hop + 1):
            for j in (i - d, i + d):
                if 0 <= j < L:
                    es.append(i)
                    et.append(j)
                    rbf = rbf_expand(np.array([abs(i - j)]), K=rbf_K, gamma=rbf_gamma)[0]
                    attr = np.concatenate(
                        [rbf, np.array([1.0 if abs(i - j) == 1 else 0.0], dtype=np.float32)],
                        axis=0,
                    )
                    attrs.append(attr)

    if not es and L > 1:
        es += [0, 1]
        et += [1, 0]
        rbf = rbf_expand(np.array([1]), K=rbf_K, gamma=rbf_gamma)[0]
        attr = np.concatenate([rbf, np.array([1.0], dtype=np.float32)], axis=0)
        attrs += [attr, attr]

    edge_index = torch.tensor([es, et], dtype=torch.long)
    edge_attr = torch.tensor(np.asarray(attrs, dtype=np.float32), dtype=torch.float32)

    data = Data(x=feat_x, edge_index=edge_index, edge_attr=edge_attr)
    data.center_idx = torch.tensor([L // 2], dtype=torch.long)
    data.pos_idx = torch.arange(L, dtype=torch.long)
    return data


# =========================
# Dataset
# =========================

class GraphFolderDataset(Dataset):
    """
    Expected folder structure:
      graphs_root/
        train/
          sample_00000000/
            esm.npy
            aa_onehot.npy
            label.txt
            species_id.txt
        val/
        test/
    """

    def __init__(self, root, split, species_filter=None, k_hop=2, rbf_K=6, rbf_gamma=1.0):
        self.root = Path(root)
        self.split = split
        self.species_filter = set(species_filter) if species_filter is not None else None
        self.k_hop = k_hop
        self.rbf_K = rbf_K
        self.rbf_gamma = rbf_gamma

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        self.samples = []
        for sd in sorted(split_dir.iterdir()):
            if not sd.is_dir():
                continue
            species_path = sd / "species_id.txt"
            if not species_path.exists():
                continue
            sp = int(species_path.read_text().strip())
            if self.species_filter is not None and sp not in self.species_filter:
                continue
            self.samples.append(sd)

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {split_dir} for species_filter={species_filter}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sd = self.samples[idx]

        esm = np.load(sd / "esm.npy")          # [L, 1280]
        aa = np.load(sd / "aa_onehot.npy")     # [L, 21]
        y = int((sd / "label.txt").read_text().strip())
        sp = int((sd / "species_id.txt").read_text().strip())

        L = esm.shape[0]
        relpos = np.arange(-(L // 2), L - (L // 2), dtype=np.float32)[:, None] / (L // 2 if L > 2 else 1.0)
        center = np.zeros((L, 1), dtype=np.float32)
        center[L // 2, 0] = 1.0

        x_np = np.concatenate([esm, aa, relpos, center], axis=1)  # 1280 + 21 + 1 + 1 = 1303
        x = torch.tensor(x_np, dtype=torch.float32)

        g = build_graph_seq_rbf(
            x,
            k_hop=self.k_hop,
            rbf_K=self.rbf_K,
            rbf_gamma=self.rbf_gamma
        )
        g.y = torch.tensor([y], dtype=torch.long)
        g.species = torch.tensor([sp], dtype=torch.long)
        return g


# =========================
# Model
# =========================

class GNNEncoder(nn.Module):
    def __init__(self, in_dim, edge_feat_dim, hid=256, layers=3):
        super().__init__()

        def mlp(din, dout):
            return nn.Sequential(
                nn.Linear(din, hid),
                nn.ReLU(),
                nn.Linear(hid, dout)
            )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GINEConv(mlp(in_dim, hid), edge_dim=edge_feat_dim))
        self.norms.append(nn.LayerNorm(hid))

        for _ in range(max(0, layers - 1)):
            self.convs.append(GINEConv(mlp(hid, hid), edge_dim=edge_feat_dim))
            self.norms.append(nn.LayerNorm(hid))

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None and edge_attr.dtype != x.dtype:
            edge_attr = edge_attr.to(x.dtype)

        for conv, ln in zip(self.convs, self.norms):
            x = ln(F.relu(conv(x, edge_index, edge_attr)))
        return x


class CenterClassifier(nn.Module):
    def __init__(self, hid=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 2)
        )

    def forward(self, h):
        return self.head(h)


class PublicModel(nn.Module):
    def __init__(self, in_dim, edge_feat_dim, hid=256, layers=3, mode="fuse"):
        super().__init__()
        self.mode = mode

        self.enc = GNNEncoder(in_dim, edge_feat_dim, hid, layers)
        self.cls = CenterClassifier(hid)

        self.esm_head = nn.Sequential(
            nn.Linear(1280, hid),
            nn.ReLU(),
            nn.Linear(hid, 2)
        )

        self.fuse_head = nn.Sequential(
            nn.Linear(hid + 1280, hid),
            nn.ReLU(),
            nn.Linear(hid, 2)
        )

    def forward(self, data):
        batch = data.batch

        if hasattr(data, "pos_idx"):
            pos_idx = data.pos_idx.to(data.x.device)
        else:
            node_ids = torch.arange(data.x.size(0), device=data.x.device)
            pos_idx = node_ids - data.ptr[batch]

        center = data.center_idx[batch].to(data.x.device)
        mask = (pos_idx == center).float().unsqueeze(-1)

        center_esm = global_add_pool(data.x[:, :1280] * mask, batch)

        if self.mode == "esm_mlp":
            return self.esm_head(center_esm)

        x = self.enc(data.x, data.edge_index, data.edge_attr)
        center_gnn = global_add_pool(x * mask, batch)

        if self.mode == "fuse":
            return self.fuse_head(torch.cat([center_gnn, center_esm], dim=-1))

        return self.cls(center_gnn)


# =========================
# Metrics
# =========================

@torch.no_grad()
def get_scores(model, loader, device):
    model.eval()
    y_true, y_prob = [], []

    for batch in loader:
        batch = batch.to(device)
        prob = torch.softmax(model(batch), dim=1)[:, 1]
        y_true.extend(batch.y.cpu().tolist())
        y_prob.extend(prob.cpu().tolist())

    return np.asarray(y_true, dtype=np.int64), np.asarray(y_prob, dtype=np.float32)


def metrics_from_threshold(y_true, y_prob, thr=0.5):
    pred = (y_prob >= thr).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    sp = tn / (tn + fp + 1e-8)
    sn = tp / (tp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    pre = tp / (tp + fp + 1e-8)
    f1 = 2 * pre * sn / (pre + sn + 1e-8)

    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")

    return {
        "AUC": float(auc),
        "Sp": float(sp),
        "Sn": float(sn),
        "Acc": float(acc),
        "Pre": float(pre),
        "F1": float(f1),
        "Threshold": float(thr),
    }


@torch.no_grad()
def evaluate_loader(model, loader, device, thr=0.5):
    y_true, y_prob = get_scores(model, loader, device)
    return metrics_from_threshold(y_true, y_prob, thr=thr)


# =========================
# Train
# =========================

def train_one_species(args, species_id, device):
    train_ds = GraphFolderDataset(args.graphs_root, "train", species_filter=[species_id])
    val_ds = GraphFolderDataset(args.graphs_root, "val", species_filter=[species_id])
    test_ds = GraphFolderDataset(args.graphs_root, "test", species_filter=[species_id])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    sample0 = train_ds[0]
    edge_feat_dim = int(sample0.edge_attr.size(-1))
    in_dim = int(sample0.x.size(-1))

    model = PublicModel(
        in_dim=in_dim,
        edge_feat_dim=edge_feat_dim,
        hid=args.hid,
        layers=args.layers,
        mode=args.mode,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # class weight from training set only
    y_count = [0, 0]
    for g in train_ds:
        y_count[int(g.y.item())] += 1

    n_neg = y_count[0]
    n_pos = max(1, y_count[1])

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, (n_neg + 1e-8) / (n_pos + 1e-8)], device=device)
    )

    best_auc = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0

        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += batch.y.size(0)
            correct += (logits.argmax(1) == batch.y).sum().item()

        yv, pv = get_scores(model, val_loader, device)
        auc = roc_auc_score(yv, pv) if len(set(yv)) > 1 else float("-inf")

        print(
            f"[species={species_id}] "
            f"epoch={ep:02d} "
            f"train_acc={correct / max(total, 1):.4f} "
            f"val_auc={auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed: no best_state was saved.")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    val_metrics = evaluate_loader(model, val_loader, device, thr=args.threshold)
    test_metrics = evaluate_loader(model, test_loader, device, thr=args.threshold)

    return {
        "species_id": species_id,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }


def collect_species_ids(graphs_root):
    graphs_root = Path(graphs_root)
    train_dir = graphs_root / "train"
    species_ids = set()

    for sd in sorted(train_dir.iterdir()):
        if not sd.is_dir():
            continue
        p = sd / "species_id.txt"
        if p.exists():
            species_ids.add(int(p.read_text().strip()))

    if not species_ids:
        raise RuntimeError("No species_id.txt found under train/")

    return sorted(species_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_root", type=str, required=True)
    parser.add_argument("--mode", type=str, default="fuse", choices=["gnn", "esm_mlp", "fuse"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hid", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_csv", type=str, default="public_summary.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.gpu_id)
    print(f"Using device: {device}")

    species_ids = collect_species_ids(args.graphs_root)
    print(f"Found species IDs: {species_ids}")

    summary = []
    for species_id in species_ids:
        result = train_one_species(args, species_id, device)
        summary.append(result)
        print(result)

    df = pd.DataFrame(summary)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved summary to: {args.out_csv}")


if __name__ == "__main__":
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    main()
