"""
================================================================================
Few-Shot Classification on the BACH (ICIAR 2018) Breast Cancer Histology Dataset
Siamese network with a Poincare (hyperbolic) prototypical head + entropic
Wasserstein structural loss.
================================================================================

Approach (as discussed):
  * A shared CNN encoder (Siamese backbone) maps each image to
       (a) a GLOBAL embedding, projected into the Poincare ball  -> hierarchy /
           grade-aware comparison via geodesic distance, and
       (b) a small SET of LOCAL feature points, projected into the ball -> an
           entropic-Wasserstein (Sinkhorn-OT) structural comparison between the
           internal layout of two images.
  * Episodic N-way K-shot training (prototypical style):
       prototypes are HYPERBOLIC centroids (Einstein / tangent-space midpoint),
       queries classified by softmax over negative Poincare distance.
  * Wasserstein term is added as an auxiliary structural consistency loss that
       pulls same-class local-structure together and pushes different-class apart.

Evaluation:
  * 1 / 3 / 5 / 7 / 10-shot, N-way (= 4 classes for BACH).
  * Per-class disjoint split: some images per class held out as the test pool,
    the rest used to sample support/query episodes for training and the
    support set at test time.
  * CLAHE preprocessing on the (large) .tif histology images.

Result artifacts produced (typical few-shot-paper figures):
  * accuracy_vs_shots.png         - mean accuracy +/- 95% CI across shots
  * confusion_matrix_Kshot.png    - per-shot confusion matrices
  * tsne_embeddings_Kshot.png     - t-SNE of Poincare embeddings (test pool)
  * support_query_episode.png     - a sampled episode visualisation
  * gradcam_examples.png          - Grad-CAM overlays on sample images
  * clahe_before_after.png        - CLAHE preprocessing illustration
  * results_summary.csv           - numeric results table

--------------------------------------------------------------------------------
EXPECTED DATA LAYOUT (edit DATA_ROOT below to match your machine):

  <DATA_ROOT>/
      Benign/    *.tif   (~101 files)
      InSitu/    *.tif
      Invasive/  *.tif
      Normal/    *.tif

In the uploaded structure this is:
  ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos/{Benign,InSitu,Invasive,Normal}
--------------------------------------------------------------------------------

Dependencies:
  torch torchvision numpy scikit-learn matplotlib opencv-python pillow tifffile

Notes on stability:
  * We train the hyperbolic geometry in the Poincare ball with curvature c,
    clip embedding norms away from the boundary, and use float32-safe
    expressions (artanh/arcosh with eps). For very heavy use the Lorentz model
    is more stable, but the ball is clearer to read and fine at this scale.
"""

import os
import csv
import math
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIG
# =============================================================================

class CFG:
    # ---- EDIT THIS to point at the folder that directly contains the 4 class
    #      subfolders (Benign / InSitu / Invasive / Normal). ----
    DATA_ROOT = "/kaggle/input/datasets/truthisneverlinear/bach-breast-cancer-histology-images/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos"

    CLASSES = ["Benign", "InSitu", "Invasive", "Normal"]
    N_WAY = 4                      # BACH has 4 classes
    SHOTS = [1, 3, 5, 7, 10]       # K values to evaluate
    Q_QUERIES = 5                  # queries per class per episode (train + eval)

    IMG_SIZE = 224
    USE_CLAHE = True

    # test pool: how many images PER CLASS are held out for testing.
    # remaining images are the "training samples" used for episodes / support.
    N_TEST_PER_CLASS = 30          # ~101 total -> ~71 train / 30 test per class

    EMBED_DIM = 128                # global embedding dim
    LOCAL_POINTS = 16              # local feature points per image for OT (4x4 grid)
    CURV_C = 1.0                   # Poincare ball curvature magnitude

    LAMBDA_WASS = 0.3              # weight of the Wasserstein structural loss
    SINKHORN_EPS = 0.1             # entropic regularisation
    SINKHORN_ITERS = 50

    TRAIN_EPISODES = 200           # episodes per training run (keep modest)
    TRAIN_K = 5                    # shots used during training episodes
    LR = 1e-3
    EVAL_EPISODES = 200            # episodes averaged at evaluation time

    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR = "."

EPS = 1e-6


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# =============================================================================
# 1. CLAHE + IMAGE LOADING
# =============================================================================

def apply_clahe(rgb_uint8):
    """CLAHE on the L channel in LAB space. Input/output: HxWx3 uint8 RGB."""
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def load_image(path, size=CFG.IMG_SIZE, use_clahe=True):
    """Load a (large) .tif histology image -> resized, CLAHE'd uint8 RGB."""
    # PIL handles BACH .tif (3-channel) well; fall back to cv2 if needed.
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
    except Exception:
        arr = cv2.imread(path, cv2.IMREAD_COLOR)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    if use_clahe:
        arr = apply_clahe(arr)
    return arr  # uint8 HxWx3


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(arr_uint8, augment=False):
    """uint8 RGB -> normalised CHW float tensor, with light optional augment."""
    x = arr_uint8.astype(np.float32) / 255.0
    if augment:
        if random.random() < 0.5:
            x = x[:, ::-1, :].copy()           # h-flip
        if random.random() < 0.5:
            x = x[::-1, :, :].copy()           # v-flip
        if random.random() < 0.5:
            k = random.randint(1, 3)
            x = np.rot90(x, k, axes=(0, 1)).copy()
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


# =============================================================================
# 2. DATASET INDEX + EPISODE SAMPLER
# =============================================================================

class BachIndex:
    """Scans the BACH folders, caches preprocessed images, builds train/test
    splits per class, and samples N-way K-shot episodes."""

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.paths = defaultdict(list)     # class_idx -> [paths]
        for ci, cname in enumerate(cfg.CLASSES):
            cdir = os.path.join(cfg.DATA_ROOT, cname)
            if not os.path.isdir(cdir):
                raise FileNotFoundError(f"Missing class folder: {cdir}")
            files = sorted([f for f in os.listdir(cdir)
                            if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))])
            self.paths[ci] = [os.path.join(cdir, f) for f in files]
            print(f"  {cname:10s}: {len(self.paths[ci])} files")

        # per-class disjoint train/test split
        rng = random.Random(cfg.SEED)
        self.train_idx = defaultdict(list)
        self.test_idx = defaultdict(list)
        for ci in range(cfg.N_WAY):
            idxs = list(range(len(self.paths[ci])))
            rng.shuffle(idxs)
            self.test_idx[ci] = idxs[:cfg.N_TEST_PER_CLASS]
            self.train_idx[ci] = idxs[cfg.N_TEST_PER_CLASS:]

        self._cache = {}   # path -> uint8 image (lazy)

    def get_img(self, ci, i):
        path = self.paths[ci][i]
        if path not in self._cache:
            self._cache[path] = load_image(path, self.cfg.IMG_SIZE, self.cfg.USE_CLAHE)
        return self._cache[path]

    def sample_episode(self, k_shot, n_query, pool="train", augment=True):
        """Returns support/query tensors and labels for an N-way k-shot episode."""
        cfg = self.cfg
        sup_x, sup_y, qry_x, qry_y = [], [], [], []
        for ci in range(cfg.N_WAY):
            avail = self.train_idx[ci] if pool == "train" else self.test_idx[ci]
            chosen = random.sample(avail, min(k_shot + n_query, len(avail)))
            sup_sel = chosen[:k_shot]
            qry_sel = chosen[k_shot:k_shot + n_query]
            for i in sup_sel:
                sup_x.append(to_tensor(self.get_img(ci, i), augment))
                sup_y.append(ci)
            for i in qry_sel:
                qry_x.append(to_tensor(self.get_img(ci, i), augment=False))
                qry_y.append(ci)
        return (torch.stack(sup_x), torch.tensor(sup_y),
                torch.stack(qry_x), torch.tensor(qry_y))

    def test_support_query(self, k_shot):
        """For final evaluation: sample k_shot support per class from the TRAIN
        pool (the 'training samples'), and use ALL remaining TEST-pool images as
        queries (test on other images of the folder)."""
        cfg = self.cfg
        sup_x, sup_y = [], []
        for ci in range(cfg.N_WAY):
            sel = random.sample(self.train_idx[ci], k_shot)
            for i in sel:
                sup_x.append(to_tensor(self.get_img(ci, i), augment=False))
                sup_y.append(ci)
        qry_x, qry_y = [], []
        for ci in range(cfg.N_WAY):
            for i in self.test_idx[ci]:
                qry_x.append(to_tensor(self.get_img(ci, i), augment=False))
                qry_y.append(ci)
        return (torch.stack(sup_x), torch.tensor(sup_y),
                torch.stack(qry_x), torch.tensor(qry_y))


# =============================================================================
# 3. POINCARE BALL OPERATIONS (curvature c)
# =============================================================================

def artanh(x):
    x = torch.clamp(x, -1 + EPS, 1 - EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def project_to_ball(x, c=CFG.CURV_C):
    """Clip points to stay strictly inside the ball of radius 1/sqrt(c)."""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    maxnorm = (1.0 - 1e-3) / math.sqrt(c)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def expmap0(v, c=CFG.CURV_C):
    """Exponential map at the origin: tangent vector -> ball."""
    sqrt_c = math.sqrt(c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(EPS)
    coef = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    return project_to_ball(coef * v, c)


def poincare_distance(x, y, c=CFG.CURV_C):
    """Geodesic distance in the Poincare ball. x:(...,D) y:(...,D) -> (...)."""
    sqrt_c = math.sqrt(c)
    diff = x - y
    num = diff.pow(2).sum(dim=-1)
    xn = x.pow(2).sum(dim=-1)
    yn = y.pow(2).sum(dim=-1)
    denom = (1 - c * xn).clamp_min(EPS) * (1 - c * yn).clamp_min(EPS)
    arg = 1 + 2 * c * num / denom
    arg = torch.clamp(arg, min=1 + EPS)
    return (1.0 / sqrt_c) * torch.acosh(arg)


def poincare_pairwise(a, b, c=CFG.CURV_C):
    """Pairwise distances. a:(N,D) b:(M,D) -> (N,M)."""
    N, M = a.size(0), b.size(0)
    a_e = a.unsqueeze(1).expand(N, M, a.size(-1))
    b_e = b.unsqueeze(0).expand(N, M, b.size(-1))
    return poincare_distance(a_e, b_e, c)


def hyperbolic_centroid(points, c=CFG.CURV_C):
    """Approximate hyperbolic centroid via tangent-space (Karcher) mean:
    log at origin -> Euclidean mean -> exp at origin. points:(K,D) -> (D,)."""
    # log map at origin
    sqrt_c = math.sqrt(c)
    p_norm = points.norm(dim=-1, keepdim=True).clamp_min(EPS)
    log = (artanh(sqrt_c * p_norm) / (sqrt_c * p_norm)) * points
    mean_tangent = log.mean(dim=0, keepdim=True)
    return expmap0(mean_tangent, c).squeeze(0)


# =============================================================================
# 4. ENTROPIC WASSERSTEIN (SINKHORN) ON POINCARE LOCAL POINTS
# =============================================================================

def sinkhorn_wasserstein(cost, eps=CFG.SINKHORN_EPS, iters=CFG.SINKHORN_ITERS):
    """Entropic OT between two uniform distributions given a cost matrix.
    cost:(n,m) -> scalar transport cost <T, cost>. Differentiable."""
    n, m = cost.shape
    mu = torch.full((n,), 1.0 / n, device=cost.device)
    nu = torch.full((m,), 1.0 / m, device=cost.device)
    K = torch.exp(-cost / eps) + 1e-9
    u = torch.ones_like(mu)
    for _ in range(iters):
        u = mu / (K @ (nu / (K.t() @ u)).clamp_min(1e-9)).clamp_min(1e-9)
    v = nu / (K.t() @ u).clamp_min(1e-9)
    T = torch.diag(u) @ K @ torch.diag(v)
    return (T * cost).sum()


def wasserstein_local(points_a, points_b, c=CFG.CURV_C):
    """Wasserstein distance between two sets of hyperbolic local points,
    using Poincare distance as the ground cost. points:(P,D)."""
    cost = poincare_pairwise(points_a, points_b, c)   # (P,P)
    return sinkhorn_wasserstein(cost)


# =============================================================================
# 5. SIAMESE ENCODER (shared) -> global Poincare embed + local point set
# =============================================================================

class SiameseHyperbolic(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        from torchvision import models
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(*list(backbone.children())[:-2])  # -> (B,512,7,7)
        self.feat_ch = 512

        # global head: GAP -> linear -> tangent vector -> exp map
        self.global_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(self.feat_ch, cfg.EMBED_DIM)
        )
        # local head: 1x1 conv to a small dim, pooled to LOCAL_POINTS grid points
        self.local_dim = 32
        self.local_conv = nn.Conv2d(self.feat_ch, self.local_dim, kernel_size=1)
        self.grid = int(math.sqrt(cfg.LOCAL_POINTS))   # 16 -> 4x4

    def forward(self, x):
        f = self.stem(x)                          # (B,512,7,7)
        # global
        g = F.adaptive_avg_pool2d(f, 1)           # (B,512,1,1)
        g = self.global_fc(g)                     # (B,EMBED_DIM) tangent
        g = expmap0(g, self.cfg.CURV_C)           # (B,EMBED_DIM) on ball

        # local: reduce channels, pool to grid x grid, flatten to points
        l = self.local_conv(f)                    # (B,local_dim,7,7)
        l = F.adaptive_avg_pool2d(l, self.grid)   # (B,local_dim,grid,grid)
        B = l.size(0)
        l = l.permute(0, 2, 3, 1).reshape(B, self.grid * self.grid, self.local_dim)
        l = expmap0(l, self.cfg.CURV_C)           # points on ball (B,P,local_dim)
        return g, l

    # --- Grad-CAM support: expose last conv feature map ---
    def features_and_global(self, x):
        f = self.stem(x)
        g = F.adaptive_avg_pool2d(f, 1)
        g = self.global_fc(g)
        g = expmap0(g, self.cfg.CURV_C)
        return f, g


# =============================================================================
# 6. EPISODE LOSS  (prototypical Poincare CE + Wasserstein structural term)
# =============================================================================

def episode_loss_and_acc(model, sup_x, sup_y, qry_x, qry_y, cfg=CFG, train=True):
    dev = cfg.DEVICE
    sup_x, qry_x = sup_x.to(dev), qry_x.to(dev)
    sup_y, qry_y = sup_y.to(dev), qry_y.to(dev)

    g_sup, l_sup = model(sup_x)          # (S,D), (S,P,d)
    g_qry, l_qry = model(qry_x)          # (Q,D), (Q,P,d)

    # ---- hyperbolic prototypes per class ----
    protos = []
    proto_local = []
    for ci in range(cfg.N_WAY):
        mask = (sup_y == ci)
        protos.append(hyperbolic_centroid(g_sup[mask], cfg.CURV_C))
        # representative local set = mean over support local points (per grid pos)
        proto_local.append(l_sup[mask].mean(dim=0))     # (P,d)
    protos = torch.stack(protos)                         # (N,D)

    # ---- prototypical classification via Poincare distance ----
    dists = poincare_pairwise(g_qry, protos, cfg.CURV_C)  # (Q,N)
    logits = -dists
    ce = F.cross_entropy(logits, qry_y)
    pred = logits.argmax(dim=1)
    acc = (pred == qry_y).float().mean().item()

    # ---- Wasserstein structural consistency term ----
    # For each query, OT distance to its true-class proto-local set should be
    # small; to other classes' sets, larger (margin). Cheap version: pull to
    # correct, push from a random wrong class.
    wass = torch.tensor(0.0, device=dev)
    if cfg.LAMBDA_WASS > 0:
        margin = 1.0
        for qi in range(l_qry.size(0)):
            true_c = qry_y[qi].item()
            wrong_c = random.choice([k for k in range(cfg.N_WAY) if k != true_c])
            d_pos = wasserstein_local(l_qry[qi], proto_local[true_c], cfg.CURV_C)
            d_neg = wasserstein_local(l_qry[qi], proto_local[wrong_c], cfg.CURV_C)
            wass = wass + d_pos + F.relu(margin - d_neg)
        wass = wass / l_qry.size(0)

    loss = ce + cfg.LAMBDA_WASS * wass
    return loss, acc, (logits.detach(), qry_y.detach())


# =============================================================================
# 7. TRAIN + EVAL
# =============================================================================

def train_model(index, cfg=CFG):
    model = SiameseHyperbolic(cfg).to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    model.train()
    running = []
    for ep in range(cfg.TRAIN_EPISODES):
        sx, sy, qx, qy = index.sample_episode(cfg.TRAIN_K, cfg.Q_QUERIES,
                                              pool="train", augment=True)
        loss, acc, _ = episode_loss_and_acc(model, sx, sy, qx, qy, cfg, train=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        running.append(acc)
        if (ep + 1) % 50 == 0:
            print(f"    episode {ep+1:4d}/{cfg.TRAIN_EPISODES}  "
                  f"loss={loss.item():.3f}  train_acc(50)={np.mean(running[-50:]):.3f}")
    return model


def evaluate_kshot(model, index, k_shot, cfg=CFG, episodes=None):
    """Average accuracy over episodes for a given K, plus pooled predictions
    (from a single full test-pool pass) for the confusion matrix."""
    episodes = episodes or cfg.EVAL_EPISODES
    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(episodes):
            sx, sy, qx, qy = index.sample_episode(k_shot, cfg.Q_QUERIES,
                                                 pool="test", augment=False)
            _, acc, _ = episode_loss_and_acc(model, sx, sy, qx, qy, cfg, train=False)
            accs.append(acc)
        # one full pass over the entire test pool for confusion matrix / t-SNE
        sx, sy, qx, qy = index.test_support_query(k_shot)
        _, full_acc, (logits, ytrue) = episode_loss_and_acc(
            model, sx, sy, qx, qy, cfg, train=False)
        preds = logits.argmax(dim=1).cpu().numpy()
        ytrue = ytrue.cpu().numpy()
    mean = float(np.mean(accs))
    ci95 = float(1.96 * np.std(accs) / math.sqrt(len(accs)))
    return mean, ci95, preds, ytrue


# =============================================================================
# 8. RESULT FIGURES
# =============================================================================

def fig_clahe(index, cfg=CFG):
    fig, axes = plt.subplots(2, cfg.N_WAY, figsize=(4 * cfg.N_WAY, 8))
    for ci, cname in enumerate(cfg.CLASSES):
        i = index.train_idx[ci][0]
        path = index.paths[ci][i]
        raw = load_image(path, cfg.IMG_SIZE, use_clahe=False)
        cla = apply_clahe(raw)
        axes[0, ci].imshow(raw); axes[0, ci].set_title(f"{cname}\noriginal"); axes[0, ci].axis("off")
        axes[1, ci].imshow(cla); axes[1, ci].set_title("CLAHE"); axes[1, ci].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, "clahe_before_after.png"), dpi=130)
    plt.close()


def fig_episode(index, cfg=CFG, k_shot=3):
    sx, sy, qx, qy = index.sample_episode(k_shot, 2, pool="train", augment=False)
    def denorm(t):
        x = t.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
        return np.clip(x, 0, 1)
    n = sx.size(0)
    cols = k_shot
    fig, axes = plt.subplots(cfg.N_WAY, cols, figsize=(2.2 * cols, 2.2 * cfg.N_WAY))
    if cols == 1:
        axes = axes.reshape(cfg.N_WAY, 1)
    for idx in range(n):
        r = sy[idx].item(); 
        # find column slot
        c = idx % cols
        axes[r, c].imshow(denorm(sx[idx])); axes[r, c].axis("off")
        if c == 0:
            axes[r, c].set_ylabel(cfg.CLASSES[r])
    fig.suptitle(f"Support set — {cfg.N_WAY}-way {k_shot}-shot episode")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, "support_query_episode.png"), dpi=130)
    plt.close()


def fig_accuracy_curve(results, cfg=CFG):
    shots = sorted(results.keys())
    means = [results[k]["mean"] * 100 for k in shots]
    cis = [results[k]["ci95"] * 100 for k in shots]
    plt.figure(figsize=(7, 5))
    plt.errorbar(shots, means, yerr=cis, marker="o", capsize=4, lw=2)
    for k, m in zip(shots, means):
        plt.annotate(f"{m:.1f}%", (k, m), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xlabel("Shots (K)"); plt.ylabel("Accuracy (%)")
    plt.title(f"BACH {cfg.N_WAY}-way few-shot — Poincare + Wasserstein Siamese")
    plt.grid(alpha=0.3); plt.xticks(shots)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, "accuracy_vs_shots.png"), dpi=130)
    plt.close()


def fig_confusion(preds, ytrue, k_shot, cfg=CFG):
    cm = confusion_matrix(ytrue, preds, labels=list(range(cfg.N_WAY)))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(fraction=0.046)
    for i in range(cfg.N_WAY):
        for j in range(cfg.N_WAY):
            plt.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                     color="white" if cmn[i, j] > 0.5 else "black")
    plt.xticks(range(cfg.N_WAY), cfg.CLASSES, rotation=45, ha="right")
    plt.yticks(range(cfg.N_WAY), cfg.CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion matrix — {k_shot}-shot")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, f"confusion_matrix_{k_shot}shot.png"), dpi=130)
    plt.close()


def fig_tsne(model, index, k_shot, cfg=CFG):
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for ci in range(cfg.N_WAY):
            for i in index.test_idx[ci]:
                x = to_tensor(index.get_img(ci, i)).unsqueeze(0).to(cfg.DEVICE)
                g, _ = model(x)
                embs.append(g.cpu().numpy()[0]); labels.append(ci)
    embs = np.array(embs); labels = np.array(labels)
    if len(embs) < 5:
        return
    ts = TSNE(n_components=2, perplexity=min(30, len(embs) - 1), init="pca",
              random_state=cfg.SEED).fit_transform(embs)
    plt.figure(figsize=(7, 6))
    for ci, cname in enumerate(cfg.CLASSES):
        m = labels == ci
        plt.scatter(ts[m, 0], ts[m, 1], label=cname, alpha=0.7, s=30)
    plt.legend(); plt.title(f"t-SNE of Poincare embeddings (test pool, {k_shot}-shot model)")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, f"tsne_embeddings_{k_shot}shot.png"), dpi=130)
    plt.close()


def fig_gradcam(model, index, cfg=CFG):
    """Grad-CAM on the last conv block, w.r.t. the norm of the global Poincare
    embedding (a proxy 'how strongly this region drives the embedding')."""
    model.eval()
    fmaps, grads = {}, {}

    target_layer = model.stem[-1]  # last resnet block

    def fwd_hook(m, i, o): fmaps["v"] = o
    def bwd_hook(m, gi, go): grads["v"] = go[0]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    fig, axes = plt.subplots(2, cfg.N_WAY, figsize=(4 * cfg.N_WAY, 8))
    for ci, cname in enumerate(cfg.CLASSES):
        i = index.test_idx[ci][0]
        raw = index.get_img(ci, i)
        x = to_tensor(raw).unsqueeze(0).to(cfg.DEVICE).requires_grad_(True)

        model.zero_grad()
        g, _ = model(x)
        score = g.norm()                       # scalar target
        score.backward()

        fm = fmaps["v"][0]                      # (C,h,w)
        gr = grads["v"][0]                      # (C,h,w)
        weights = gr.mean(dim=(1, 2))           # (C,)
        cam = F.relu((weights[:, None, None] * fm).sum(0))
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.5 * raw + 0.5 * heat)

        axes[0, ci].imshow(raw); axes[0, ci].set_title(cname); axes[0, ci].axis("off")
        axes[1, ci].imshow(overlay); axes[1, ci].set_title("Grad-CAM"); axes[1, ci].axis("off")

    h1.remove(); h2.remove()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, "gradcam_examples.png"), dpi=130)
    plt.close()


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    cfg = CFG
    set_seed(cfg.SEED)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    print(f"Device: {cfg.DEVICE}")
    print("Scanning dataset...")
    index = BachIndex(cfg)

    print("Saving CLAHE + episode illustration figures...")
    fig_clahe(index, cfg)
    fig_episode(index, cfg, k_shot=3)

    print("Training the Siamese hyperbolic model (episodic)...")
    model = train_model(index, cfg)

    print("Evaluating across shots:", cfg.SHOTS)
    results = {}
    for k in cfg.SHOTS:
        mean, ci95, preds, ytrue = evaluate_kshot(model, index, k, cfg)
        results[k] = {"mean": mean, "ci95": ci95}
        print(f"  K={k:2d}-shot:  acc = {mean*100:.2f}%  +/- {ci95*100:.2f}")
        fig_confusion(preds, ytrue, k, cfg)
        if k in (1, 5, 10):
            fig_tsne(model, index, k, cfg)

    print("Generating accuracy curve + Grad-CAM...")
    fig_accuracy_curve(results, cfg)
    fig_gradcam(model, index, cfg)

    # results table
    with open(os.path.join(cfg.OUT_DIR, "results_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shots", "accuracy_%", "ci95_%"])
        for k in sorted(results):
            w.writerow([k, f"{results[k]['mean']*100:.2f}", f"{results[k]['ci95']*100:.2f}"])

    print("\nDone. Artifacts written to:", os.path.abspath(cfg.OUT_DIR))
    print("  accuracy_vs_shots.png, confusion_matrix_*shot.png, tsne_*.png,")
    print("  clahe_before_after.png, support_query_episode.png, gradcam_examples.png,")
    print("  results_summary.csv")


if __name__ == "__main__":
    main()
