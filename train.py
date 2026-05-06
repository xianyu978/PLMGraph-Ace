import os, re, csv, json, hashlib, glob
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool


CSV_DIR   = "/data/luow/Supplemental_data_S1/Saccharomyces_cerevisiae9/"
ESM2_MODEL= "/data/luow/ACP-CLB-main/ESM2"

MODE      = "fuse"
EPOCHS    = 40
BATCH_SIZE= 64
LR        = 5e-5
WEIGHT_DECAY = 5e-4
HID       = 512
LAYERS    = 4
SEED      = 42
SP_TARGET = 0.9
GPU_ID    = 1


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def set_seed(seed: int):
    if seed is None: return
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def pick_device(gpu_id=None):
    if torch.cuda.is_available():
        if gpu_id is None:
            return "cuda:0"
        else:
            try:
                torch.cuda.set_device(int(gpu_id))
            except Exception:
                pass
            return f"cuda:{int(gpu_id)}"
    return "cpu"


AA20 = set(list("ACDEFGHIKLMNPQRSTVWY"))
AA20_STR = "ACDEFGHIKLMNPQRSTVWY"


def makedirs(p): os.makedirs(p, exist_ok=True)


def norm_for_onehot(pep: str, window: int = 31) -> str:
    pep = (pep or "").strip().upper().replace(" ", "")
    pep = "".join([ch if (ch in AA20 or ch == "X") else "X" for ch in pep])
    if len(pep) > window: pep = pep[:window]
    if len(pep) < window: pep = pep + "X" * (window - len(pep))
    return pep


def norm_for_models(pep_onehot: str, window: int = 31) -> str:
    pep = (pep_onehot or "").replace("*", "A")
    pep = "".join([ch if ch in AA20 else "A" for ch in pep])
    if len(pep) > window: pep = pep[:window]
    if len(pep) < window: pep = pep + pep[-1:] * (window - len(pep))
    return pep


def model_hash(pep_onehot: str, window: int = 31) -> str:
    s = norm_for_models(pep_onehot, window)
    return hashlib.md5(s.encode()).hexdigest()


def aa_onehot_21(pep_onehot: str) -> np.ndarray:
    L = len(pep_onehot)
    mat = np.zeros((L, 21), dtype=np.float32)
    for i, ch in enumerate(pep_onehot):
        if ch in AA20_STR:
            mat[i, AA20_STR.index(ch)] = 1.0
        else:
            mat[i, 20] = 1.0
    return mat


def species_from_filename(path):
    b = os.path.basename(path)
    m = re.match(r"(train|valid|val|test)[-_]([^._]+)", b, flags=re.I)
    return m.group(2) if m else "Unknown"


def split_from_filename(path):
    b = os.path.basename(path).lower()
    if b.startswith("train"): return "train"
    if b.startswith("valid") or b.startswith("val"): return "val"
    return "test"


def step_A_make_lists(csv_dir, lists_dir, window=31, enforce_center_K=True):
    makedirs(lists_dir)
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")) +
                   glob.glob(os.path.join(csv_dir, "*.txt")))

    outf = {
        "train": open(os.path.join(lists_dir, "train.jsonl"), "w", encoding="utf-8"),
        "val":   open(os.path.join(lists_dir, "val.jsonl"),   "w", encoding="utf-8"),
        "test":  open(os.path.join(lists_dir, "test.jsonl"),  "w", encoding="utf-8"),
    }

    species_to_id, next_sid = OrderedDict(), 0
    drop_nonK = 0

    for path in paths:
        sp_name = species_from_filename(path)
        sp = split_from_filename(path)

        if sp_name not in species_to_id:
            species_to_id[sp_name] = next_sid; next_sid += 1

        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if len(row) == 1 and row[0].strip() in ("0", "1"): continue
                if len(row) < 3: continue

                acc = (row[0] or "").strip()
                pos = (row[1] or "").strip()
                pep_raw = (row[2] or "").strip().upper().replace(" ", "")
                lab_raw = (row[3] if len(row) >= 4 else "0").strip()

                pep_onehot = norm_for_onehot(pep_raw, window)
                if len(pep_onehot) != window: continue

                if enforce_center_K and pep_onehot[window // 2] != "K":
                    drop_nonK += 1
                    continue

                lab = 1 if lab_raw in ("1", "pos", "positive", "yes", "y") else 0
                kpos = int(pos) if pos.isdigit() else -1

                rec = {
                    "species": sp_name,
                    "species_id": species_to_id[sp_name],
                    "protein_id": acc if acc else f"{sp_name}_unk",
                    "k_pos": kpos,
                    "peptide31": pep_onehot,
                    "label": lab
                }

                outf[sp].write(json.dumps(rec, ensure_ascii=False) + "\n")

    for f in outf.values(): f.close()

    with open(os.path.join(lists_dir, "species_id.json"), "w", encoding="utf-8") as w:
        json.dump(species_to_id, w, ensure_ascii=False, indent=2)

    print("[A] Finished:", species_to_id)

    if enforce_center_K and drop_nonK:
        print(f"[A] Dropped windows: {drop_nonK}")

    return species_to_id


def step_B2_esm2_windows(lists_dir, repr_dir, esm2_model, device, window=31, b2_limit=0):
    from transformers import AutoModel, AutoTokenizer

    makedirs(repr_dir)
    peps = set()

    for sp in ("train", "val", "test"):
        p = os.path.join(lists_dir, f"{sp}.jsonl")
        if os.path.exists(p):
            for line in open(p, "r", encoding="utf-8"):
                peps.add(json.loads(line)["peptide31"])

    todo = []

    for pep_onehot in peps:
        h = model_hash(pep_onehot, window)
        out = os.path.join(repr_dir, f"{h}.npy")
        if not os.path.exists(out): todo.append(pep_onehot)

    if b2_limit and len(todo) > b2_limit:
        todo = todo[:b2_limit]

    print(f"[B2] Total={len(peps)} Pending={len(todo)}")

    tok = AutoTokenizer.from_pretrained(esm2_model)
    dev = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    mdl = AutoModel.from_pretrained(esm2_model).eval().to(dev)

    for pep_onehot in todo:
        h = model_hash(pep_onehot, window)
        out = os.path.join(repr_dir, f"{h}.npy")
        seq4model = norm_for_models(pep_onehot, window)

        with torch.no_grad():
            t = tok(seq4model, return_tensors="pt", add_special_tokens=True).to(mdl.device)
            outm = mdl(**t)
            H = outm.last_hidden_state[0, 1:-1, :].detach().cpu().numpy()
            np.save(out, H.astype(np.float32))


def step_B3_build_graphs(lists_dir, repr_dir, graphs_root, window=31):
    makedirs(graphs_root)

    for sp, sub in (("train", "train"), ("val", "val"), ("test", "test")):
        os.makedirs(os.path.join(graphs_root, sub), exist_ok=True)
        idx = 0
        in_list = os.path.join(lists_dir, f"{sp}.jsonl")

        if not os.path.exists(in_list): continue

        with open(in_list, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                pep_onehot = r["peptide31"]

                if len(pep_onehot) != window or pep_onehot[window // 2] != "K":
                    continue

                h = model_hash(pep_onehot, window)
                efile = os.path.join(repr_dir, f"{h}.npy")

                if not os.path.exists(efile): continue

                esm = np.load(efile)
                L = len(pep_onehot)

                if esm.shape[0] != L:
                    esm = (esm[:L] if esm.shape[0] > L else
                           np.vstack([esm, np.repeat(esm[-1:], L - esm.shape[0], axis=0)]))

                aa = aa_onehot_21(pep_onehot)

                sd = os.path.join(graphs_root, sub, f"sample_{idx:08d}")
                os.makedirs(sd, exist_ok=True)

                np.save(os.path.join(sd, "esm.npy"), esm.astype(np.float32))
                np.save(os.path.join(sd, "aa_onehot.npy"), aa.astype(np.float32))
                open(os.path.join(sd, "label.txt"), "w").write(str(int(r["label"])))
                open(os.path.join(sd, "species_id.txt"), "w").write(str(int(r["species_id"])))

                idx += 1

        print(f"[B3] {sp}: {idx}")


def rbf_expand(distance: np.ndarray, K: int = 6, gamma: float = 1.0):
    centers = np.arange(1, K + 1, dtype=np.float32)
    dist = distance[..., None].astype(np.float32)
    return np.exp(-gamma * (dist - centers) ** 2)


def build_graph_seq_rbf(feat_x: torch.Tensor, k_hop: int = 2, rbf_K: int = 6, rbf_gamma: float = 1.0):
    L = feat_x.size(0)
    es, et, attrs = [], [], []

    for i in range(L):
        for d in range(1, k_hop + 1):
            for j in (i - d, i + d):
                if 0 <= j < L:
                    es.append(i); et.append(j)
                    rbf = rbf_expand(np.array([abs(i - j)]), K=rbf_K, gamma=rbf_gamma)[0]
                    attr = np.concatenate([rbf, np.array([1.0 if abs(i - j) == 1 else 0.0], dtype=np.float32)], 0)
                    attrs.append(attr)

    if not es and L > 1:
        es += [0, 1]; et += [1, 0]
        rbf = rbf_expand(np.array([1]), K=rbf_K, gamma=rbf_gamma)[0]
        attr = np.concatenate([rbf, np.array([1.0], dtype=np.float32)], 0)
        attrs += [attr, attr]

    edge_index = torch.tensor([es, et], dtype=torch.long)
    edge_attr  = torch.tensor(np.asarray(attrs, dtype=np.float32), dtype=torch.float32)

    data = Data(x=feat_x, edge_index=edge_index, edge_attr=edge_attr)
    data.center_idx = torch.tensor([L // 2], dtype=torch.long)
    data.pos_idx = torch.arange(L, dtype=torch.long)

    return data


class GraphDataset(InMemoryDataset):
    def __init__(self, root, split, in_dim, species_filter=None):
        self.split = split
        self.in_dim = in_dim
        self.species_filter = set(species_filter) if species_filter is not None else None
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self): return [f"{self.split}.pt"]

    def process(self):
        base = os.path.join(self.root, self.split)
        items = sorted(os.listdir(base))
        data_list = []

        for it in items:
            sd = os.path.join(base, it)

            if not os.path.isdir(sd): continue

            sp = int(open(os.path.join(sd, "species_id.txt")).read().strip())

            if self.species_filter is not None and sp not in self.species_filter:
                continue

            esm = np.load(os.path.join(sd, "esm.npy"))
            aa  = np.load(os.path.join(sd, "aa_onehot.npy"))
            L = esm.shape[0]

            relpos = np.arange(-(L // 2), L - (L // 2), dtype=np.float32)[:, None] / (L // 2 if L > 2 else 1.0)
            center = np.zeros((L, 1), dtype=np.float32); center[L // 2, 0] = 1.0

            x_np = np.concatenate([esm, aa, relpos, center], axis=1)
            x = torch.tensor(x_np, dtype=torch.float32)

            g = build_graph_seq_rbf(x, k_hop=2, rbf_K=6, rbf_gamma=1.0)
            g.y = torch.tensor([int(open(os.path.join(sd, "label.txt")).read().strip())], dtype=torch.long)
            g.species = torch.tensor([sp], dtype=torch.long)

            data_list.append(g)

        if not data_list:
            dummy = Data(x=torch.zeros((31, self.in_dim), dtype=torch.float32),
                         edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                         edge_attr=torch.tensor([[0, 0, 0, 0, 0, 0, 1.0],
                                                 [0, 0, 0, 0, 0, 0, 1.0]], dtype=torch.float32))
            dummy.center_idx = torch.tensor([15])
            dummy.pos_idx = torch.arange(31)
            dummy.y = torch.tensor([0])
            dummy.species = torch.tensor([0])
            data_list = [dummy]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, edge_feat_dim, hid=256, layers=3):
        super().__init__()

        def mlp(din, dout): return nn.Sequential(nn.Linear(din, hid), nn.ReLU(), nn.Linear(hid, dout))

        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
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
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 2))

    def forward(self, h): return self.head(h)


class NoDANNModel(nn.Module):
    def __init__(self, in_dim, edge_feat_dim, hid=256, layers=3, mode="esm_mlp"):
        super().__init__()
        self.mode = mode
        self.enc = GNNEncoder(in_dim, edge_feat_dim, hid, layers)
        self.cls = CenterClassifier(hid)
        self.esm_head = nn.Sequential(nn.Linear(1280, hid), nn.ReLU(), nn.Linear(hid, 2))
        self.fuse_head = nn.Sequential(nn.Linear(hid + 1280, hid), nn.ReLU(), nn.Linear(hid, 2))

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
            return self.fuse_head(torch.cat([center_gnn, center_esm], -1))

        return self.cls(center_gnn)


@torch.no_grad()
def get_scores(model, loader, device):
    model.eval()
    y_true = []
    y_prob = []

    for b in loader:
        b = b.to(device)
        p = torch.softmax(model(b), dim=1)[:, 1]
        y_true += b.y.cpu().tolist()
        y_prob += p.cpu().tolist()

    y_true = np.asarray(y_true, np.int64)
    y_prob = np.asarray(y_prob, np.float32)

    return y_true, y_prob


def threshold_at_specificity_roc(y_true, y_prob, target_sp=0.9):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    sp = 1 - fpr

    order = np.argsort(sp)
    sp = sp[order]
    thr = thr[order]

    i = np.searchsorted(sp, target_sp)

    if i <= 0:
        T = float(thr[0])
    elif i >= len(sp):
        T = float(thr[-1])
    else:
        sp0, sp1 = sp[i - 1], sp[i]
        th0, th1 = thr[i - 1], thr[i]

        if sp1 == sp0:
            T = float((th0 + th1) / 2.0)
        else:
            w = (target_sp - sp0) / (sp1 - sp0)
            T = float(th0 + w * (th1 - th0))

    return T


def metrics_from_threshold(y_true, y_prob, thr):
    pred = (y_prob >= thr).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    sp = tn / (tn + fp + 1e-8)
    sn = tp / (tp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    pre = tp / (tp + fp + 1e-8)
    f1 = 2 * pre * sn / (pre + sn + 1e-8)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")

    return dict(AUC=float(auc), Sp=float(sp), Sn=float(sn), Acc=float(acc), Pre=float(pre), F1=float(f1))


@torch.no_grad()
def eval_fixed_sp(model, loader, device, target_sp=0.9):
    y_true, y_prob = get_scores(model, loader, device)

    if len(set(y_true)) < 2:
        return {"AUC": float("nan"), "Threshold": 0.5, "Sp": float("nan"),
                "Sn": float("nan"), "Acc": float("nan"), "Pre": float("nan"), "F1": float("nan")}

    T = threshold_at_specificity_roc(y_true, y_prob, target_sp=target_sp)
    m = metrics_from_threshold(y_true, y_prob, T)
    m["Threshold"] = float(T)

    return m


def train_one_target(graphs_root, in_dim, target_species_id, epochs, batch_size, lr, weight_decay, device,
                     hid=256, layers=3, mode="esm_mlp", seed=None, sp_target=0.9):
    train_ds = GraphDataset(graphs_root, "train", in_dim, species_filter=[target_species_id])
    val_ds   = GraphDataset(graphs_root, "val",   in_dim, species_filter=[target_species_id])
    test_ds  = GraphDataset(graphs_root, "test",  in_dim, species_filter=[target_species_id])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    edge_feat_dim = int(train_ds[0].edge_attr.size(-1)) if getattr(train_ds[0], "edge_attr", None) is not None else 0

    model = NoDANNModel(in_dim=in_dim, edge_feat_dim=edge_feat_dim, hid=hid, layers=layers, mode=mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_count = [0, 0]

    for g in train_ds:
        y_count[int(g.y.item())] += 1

    n_neg, n_pos = y_count[0], max(1, y_count[1])
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, (n_neg + 1e-8) / (n_pos + 1e-8)], device=device))

    best_auc = -1.0
    best_state = None

    for ep in range(1, max(1, epochs) + 1):
        model.train()
        total = 0
        correct = 0

        for b in train_loader:
            b = b.to(device)
            logit = model(b)
            loss = ce(logit, b.y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += b.y.size(0)
            correct += (logit.argmax(1) == b.y).sum().item()

        yv, pv = get_scores(model, val_loader, device)
        auc = roc_auc_score(yv, pv) if len(set(yv)) > 1 else float("-inf")
        val_fix = eval_fixed_sp(model, val_loader, device, target_sp=sp_target)

        print(f"  [Target {target_species_id}] Ep{ep:02d} "
              f"train_acc={correct / max(total, 1):.3f} "
              f"val_auc={auc:.4f} "
              f"val_sn={val_fix['Sn']:.3f} "
              f"val_f1={val_fix['F1']:.3f}")

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_m = eval_fixed_sp(model, test_loader, device, target_sp=sp_target)

    return test_m


def main():
    csv_dir = Path(CSV_DIR).resolve()
    assert csv_dir.exists(), f"CSV_DIR does not exist: {csv_dir}"

    work_root = (csv_dir.parent / f"work_{csv_dir.name}").resolve()
    lists_dir   = work_root / "lists"
    repr_dir    = work_root / "esm2_repr"
    graphs_dir  = work_root / "graphs"
    results_dir = work_root / "results"

    for d in (lists_dir, repr_dir, graphs_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = pick_device(GPU_ID)

    print(f"[Device] {device}")
    print(f"[Workdir] {work_root}")

    species_to_id = step_A_make_lists(str(csv_dir), str(lists_dir), window=31, enforce_center_K=True)
    step_B2_esm2_windows(str(lists_dir), str(repr_dir), ESM2_MODEL, device, window=31, b2_limit=0)
    step_B3_build_graphs(str(lists_dir), str(repr_dir), str(graphs_dir), window=31)

    in_dim = 1303
    summary = []

    for sp_name, sp_id in species_to_id.items():
        print(f"\n[Train] {sp_name} id={sp_id}")

        m = train_one_target(graphs_root=str(graphs_dir),
                             in_dim=in_dim,
                             target_species_id=sp_id,
                             epochs=EPOCHS,
                             batch_size=BATCH_SIZE,
                             lr=LR,
                             weight_decay=WEIGHT_DECAY,
                             device=device,
                             hid=HID,
                             layers=LAYERS,
                             mode=MODE,
                             seed=SEED,
                             sp_target=SP_TARGET)

        row = {"species": sp_name, "species_id": sp_id}
        row.update(m)
        summary.append(row)

        pretty = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}
        print(f"[Result] {sp_name}: {pretty}")

    try:
        import pandas as pd
        import datetime as dt

        df = pd.DataFrame(summary)
        out_csv = results_dir / "summary_results.csv"
        df.to_csv(out_csv, index=False)

        print(f"[Saved] {out_csv}")
        print(df)

        with open(results_dir / "run_log.txt", "a", encoding="utf-8") as w:
            w.write(f"{dt.datetime.now()}\tCSV_DIR={csv_dir}\tMODE={MODE}\tSEED={SEED}\tDEVICE={device}\tOUT={out_csv}\n")

    except Exception:
        print(summary)


if __name__ == "__main__":
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    main()
