"""
================================================================================
Few-Shot Classification on the BreakHis Breast Cancer Histopathology Dataset
Siamese network with a Euclidean prototypical head + Manhattan (L1)
structural loss.
================================================================================

Approach (as discussed):
  * A shared CNN encoder (Siamese backbone) maps each image to
       (a) a GLOBAL embedding, kept in plain EUCLIDEAN space -> tumour-type
           comparison via ordinary L2 distance, and
       (b) a small SET of LOCAL feature points, also in Euclidean space -> a
           MANHATTAN (L1) structural comparison between the internal layout
           of two images (grid position vs. grid position).
  * Episodic N-way K-shot training (prototypical style):
       prototypes are EUCLIDEAN centroids (plain arithmetic mean),
       queries classified via an explicit CROSS-ENTROPY loss computed over
       the negative Euclidean distance to each class prototype.
  * A Manhattan-distance term is added as an auxiliary structural consistency
       loss that pulls same-class local-structure together and pushes
       different-class apart.

Evaluation:
  * 1 / 3 / 5 / 7 / 10-shot, N-way (= 8 classes for BreakHis).
  * Per-class disjoint split: some images per class held out as the test pool,
    the rest used to sample support/query episodes for training and the
    support set at test time.
  * CLAHE preprocessing on the histology images.

Result artifacts produced (typical few-shot-paper figures):
  * accuracy_vs_shots.png         - mean accuracy +/- 95% CI across shots
  * confusion_matrix_Kshot.png    - per-shot confusion matrices
  * tsne_embeddings_Kshot.png     - t-SNE of Euclidean embeddings (test pool)
  * support_query_episode.png     - a sampled episode visualisation
  * gradcam_examples.png          - Grad-CAM overlays on sample images
  * clahe_before_after.png        - CLAHE preprocessing illustration
  * results_summary.csv           - numeric results table

--------------------------------------------------------------------------------
EXPECTED DATA LAYOUT (edit DATA_ROOT below to match your machine):

BreakHis (Kaggle: ambarish/breakhis) is NOT a flat "one folder per class"
dataset: every tumour-subtype folder contains one sub-folder per patient, and
every patient folder contains one sub-folder per magnification factor. The
standard BreaKHis_v1 layout is:

  <DATA_ROOT>/
      benign/SOB/adenosis/<patient_id>/<mag>X/*.png
      benign/SOB/fibroadenoma/<patient_id>/<mag>X/*.png
      benign/SOB/phyllodes_tumor/<patient_id>/<mag>X/*.png
      benign/SOB/tubular_adenoma/<patient_id>/<mag>X/*.png
      malignant/SOB/ductal_carcinoma/<patient_id>/<mag>X/*.png
      malignant/SOB/lobular_carcinoma/<patient_id>/<mag>X/*.png
      malignant/SOB/mucinous_carcinoma/<patient_id>/<mag>X/*.png
      malignant/SOB/papillary_carcinoma/<patient_id>/<mag>X/*.png

  where <mag> in {40, 100, 200, 400}. There are 8 tumour subtypes in total
  (4 benign + 4 malignant), which is why N_WAY below is set to 8.

  On Kaggle this dataset mounts under (confirmed from the "Data" panel):
    /kaggle/input/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast
  (note the folder name "BreaKHis_v1" is duplicated -- once for the dataset
  upload's top-level folder, once for the archive's own top-level folder --
  double-check this against your own notebook's "Data" panel if you re-add
  the dataset under a different slug).
--------------------------------------------------------------------------------

Dependencies:
  torch torchvision numpy scikit-learn matplotlib opencv-python pillow

Notes:
  * The geometry is plain Euclidean (no curvature / no ball projection), and
    the local-structure comparison uses the Manhattan (L1) distance instead
    of an optimal-transport (Wasserstein) matching cost.
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
    # ---- EDIT THIS to point at the folder that directly contains the
    #      "benign" and "malignant" top-level subfolders of BreaKHis_v1
    #      (see the "EXPECTED DATA LAYOUT" note above). ----
    DATA_ROOT = "/kaggle/input/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

    # The 8 BreakHis tumour subtypes (4 benign + 4 malignant).
    CLASSES = [
        "adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma",
        "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma",
    ]
    N_WAY = 8                      # num classes: BreakHis has 8 tumour subtypes
    SHOTS = [1, 3, 5, 7, 10]       # K values to evaluate
    Q_QUERIES = 5                  # queries per class per episode (train + eval)

    IMG_SIZE = 224
    USE_CLAHE = True

    # test pool: how many images PER CLASS are held out for testing.
    # remaining images are the "training samples" used for episodes / support.
    N_TEST_PER_CLASS = 30          # every BreakHis class has well over 30 images

    EMBED_DIM = 128                # global embedding dim
    LOCAL_POINTS = 16              # local feature points per image (4x4 grid)
    CURV_C = 1.0                   # kept only for interface compatibility; the
                                    # geometry is now Euclidean (flat / zero
                                    # curvature), so this value is unused.

    LAMBDA_MANHATTAN = 0.3         # weight of the Manhattan structural loss
    # SINKHORN_EPS / SINKHORN_ITERS are no longer used: the entropic
    # Wasserstein (Sinkhorn-OT) term was replaced with a plain Manhattan
    # (L1) distance, which needs no regularisation or iterative solving.
    # SINKHORN_EPS = 0.1
    # SINKHORN_ITERS = 50

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
    """Load a BreakHis histology image (700x460 PNG) -> resized, CLAHE'd
    uint8 RGB."""
    # PIL handles BreakHis .png (3-channel) well; fall back to cv2 if needed.
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

# Which of the 8 BreakHis tumour subtypes fall under the top-level "benign"
# vs "malignant" branch of the BreaKHis_v1/histology_slides/breast tree
# (needed to build the correct on-disk path for each class below).
BREAKHIS_GROUP = {
    "adenosis": "benign",
    "fibroadenoma": "benign",
    "phyllodes_tumor": "benign",
    "tubular_adenoma": "benign",
    "ductal_carcinoma": "malignant",
    "lobular_carcinoma": "malignant",
    "mucinous_carcinoma": "malignant",
    "papillary_carcinoma": "malignant",
}


def resolve_data_root(cfg=CFG):
    """BreakHis Kaggle uploads vary in exactly how many folders deep the
    dataset ends up nested (slug name, casing, extra wrapper folders, etc.),
    so a hard-coded DATA_ROOT guess can go stale. If the configured
    DATA_ROOT doesn't exist, search under /kaggle/input for a folder named
    'breast' that directly contains both a 'benign' and a 'malignant'
    subfolder (the BreaKHis_v1/histology_slides/breast layout) and use that
    instead, so you don't have to hand-edit the path every time."""
    if os.path.isdir(cfg.DATA_ROOT):
        return cfg.DATA_ROOT

    print(f"[WARN] DATA_ROOT not found: {cfg.DATA_ROOT}")
    print("       Searching /kaggle/input for the BreakHis 'breast' folder...")
    search_root = "/kaggle/input"
    if os.path.isdir(search_root):
        for dirpath, dirnames, _filenames in os.walk(search_root):
            if os.path.basename(dirpath) == "breast" and \
               "benign" in dirnames and "malignant" in dirnames:
                print(f"       Found: {dirpath}")
                return dirpath

    raise FileNotFoundError(
        f"Could not locate the BreakHis 'breast' folder anywhere under "
        f"'{search_root}'. Open the 'Data' panel of your Kaggle notebook, "
        f"copy the exact path shown there, and set CFG.DATA_ROOT to it "
        f"(it should be the folder that directly contains 'benign' and "
        f"'malignant')."
    )


class BreakHisIndex:
    """Scans the BreakHis folders, caches preprocessed images, builds
    train/test splits per class, and samples N-way K-shot episodes.

    Unlike BACH's flat "one folder per class" layout, BreakHis nests images
    as <group>/SOB/<class>/<patient_id>/<magnification>X/*.png. We therefore
    recursively walk each class directory and pool every image found
    underneath it (across all patients and all magnifications) into that
    class's image list.
    """

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        # Auto-correct DATA_ROOT if the configured path doesn't exist.
        cfg.DATA_ROOT = resolve_data_root(cfg)
        self.paths = defaultdict(list)     # class_idx -> [paths]
        for ci, cname in enumerate(cfg.CLASSES):
            group = BREAKHIS_GROUP[cname]
            cdir = os.path.join(cfg.DATA_ROOT, group, "SOB", cname)
            if not os.path.isdir(cdir):
                raise FileNotFoundError(f"Missing class folder: {cdir}")
            # Recursively collect every image under <cdir>/<patient>/<mag>X/*
            files = []
            for dirpath, _dirnames, filenames in os.walk(cdir):
                for f in filenames:
                    if f.lower().endswith((".tif", ".tiff", ".png", ".jpg")):
                        files.append(os.path.join(dirpath, f))
            self.paths[ci] = sorted(files)
            print(f"  {cname:20s}: {len(self.paths[ci])} files")

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
# 3. EUCLIDEAN GEOMETRY OPERATIONS
#    (replaces the previous Poincare-ball / hyperbolic-geometry code)
# =============================================================================
# NOTE: the functions below keep the same names/signatures used elsewhere in
# this notebook (including the now-unused `c` curvature argument, kept only
# so that downstream call sites do not need to change) but now perform plain
# Euclidean linear algebra instead of hyperbolic-geometry operations.

def expmap0(v, c=CFG.CURV_C):
    """Identity map. In flat Euclidean space the tangent space at the origin
    IS the embedding space itself, so there is no exponential map to apply;
    this is kept only so the model's forward() does not need to change."""
    return v


def euclidean_distance(x, y, c=CFG.CURV_C):
    """Ordinary Euclidean (L2) distance. x:(...,D) y:(...,D) -> (...)."""
    return (x - y).pow(2).sum(dim=-1).clamp_min(EPS).sqrt()


def euclidean_pairwise(a, b, c=CFG.CURV_C):
    """Pairwise Euclidean distances. a:(N,D) b:(M,D) -> (N,M)."""
    return torch.cdist(a, b, p=2)


def euclidean_centroid(points, c=CFG.CURV_C):
    """Euclidean centroid = plain arithmetic mean. points:(K,D) -> (D,)."""
    return points.mean(dim=0)


# =============================================================================
# 4. MANHATTAN (L1) DISTANCE ON LOCAL POINT SETS
#    (replaces the previous entropic-Wasserstein / Sinkhorn-OT computation)
# =============================================================================

def manhattan_local(points_a, points_b, c=CFG.CURV_C):
    """Structural distance between two same-shape sets of local feature
    points, using the Manhattan (L1) distance instead of an optimal
    transport (Wasserstein) matching cost.

    Both point sets come from the same fixed spatial grid (LOCAL_POINTS
    positions, see SiameseEuclidean below), so corresponding grid positions
    in `points_a` and `points_b` already describe the same spatial region of
    the two images -- no OT matching between mismatched points is required.
    We simply take the element-wise L1 distance and average it over points.

    points_a, points_b: (P, D) -> scalar.
    """
    return (points_a - points_b).abs().sum(dim=-1).mean()


# =============================================================================
# 5. SIAMESE ENCODER (shared) -> global Euclidean embed + local point set
# =============================================================================

class SiameseEuclidean(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        from torchvision import models
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(*list(backbone.children())[:-2])  # -> (B,512,7,7)
        self.feat_ch = 512

        # global head: GAP -> linear -> Euclidean embedding
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
        g = self.global_fc(g)                     # (B,EMBED_DIM) Euclidean
        g = expmap0(g, self.cfg.CURV_C)           # identity (no-op in Euclidean space)

        # local: reduce channels, pool to grid x grid, flatten to points
        l = self.local_conv(f)                    # (B,local_dim,7,7)
        l = F.adaptive_avg_pool2d(l, self.grid)   # (B,local_dim,grid,grid)
        B = l.size(0)
        l = l.permute(0, 2, 3, 1).reshape(B, self.grid * self.grid, self.local_dim)
        l = expmap0(l, self.cfg.CURV_C)           # Euclidean local points (B,P,local_dim)
        return g, l

    # --- Grad-CAM support: expose last conv feature map ---
    def features_and_global(self, x):
        f = self.stem(x)
        g = F.adaptive_avg_pool2d(f, 1)
        g = self.global_fc(g)
        g = expmap0(g, self.cfg.CURV_C)
        return f, g


# =============================================================================
# 6. EPISODE LOSS  (prototypical Euclidean cross-entropy + Manhattan
#    structural term)
# =============================================================================

# Explicit cross-entropy loss module used for query classification below.
CROSS_ENTROPY_LOSS = nn.CrossEntropyLoss()


def episode_loss_and_acc(model, sup_x, sup_y, qry_x, qry_y, cfg=CFG, train=True):
    dev = cfg.DEVICE
    sup_x, qry_x = sup_x.to(dev), qry_x.to(dev)
    sup_y, qry_y = sup_y.to(dev), qry_y.to(dev)

    g_sup, l_sup = model(sup_x)          # (S,D), (S,P,d)
    g_qry, l_qry = model(qry_x)          # (Q,D), (Q,P,d)

    # ---- Euclidean prototypes per class ----
    protos = []
    proto_local = []
    for ci in range(cfg.N_WAY):
        mask = (sup_y == ci)
        protos.append(euclidean_centroid(g_sup[mask], cfg.CURV_C))
        # representative local set = mean over support local points (per grid pos)
        proto_local.append(l_sup[mask].mean(dim=0))     # (P,d)
    protos = torch.stack(protos)                         # (N,D)

    # ---- prototypical classification via Euclidean distance + cross-entropy ----
    dists = euclidean_pairwise(g_qry, protos, cfg.CURV_C)  # (Q,N)
    logits = -dists
    ce = CROSS_ENTROPY_LOSS(logits, qry_y)
    pred = logits.argmax(dim=1)
    acc = (pred == qry_y).float().mean().item()

    # ---- Manhattan structural consistency term ----
    # For each query, the Manhattan (L1) distance to its true-class
    # proto-local set should be small; to other classes' sets, larger
    # (margin). Cheap version: pull to correct, push from a random wrong class.
    manhattan = torch.tensor(0.0, device=dev)
    if cfg.LAMBDA_MANHATTAN > 0:
        margin = 1.0
        for qi in range(l_qry.size(0)):
            true_c = qry_y[qi].item()
            wrong_c = random.choice([k for k in range(cfg.N_WAY) if k != true_c])
            d_pos = manhattan_local(l_qry[qi], proto_local[true_c], cfg.CURV_C)
            d_neg = manhattan_local(l_qry[qi], proto_local[wrong_c], cfg.CURV_C)
            manhattan = manhattan + d_pos + F.relu(margin - d_neg)
        manhattan = manhattan / l_qry.size(0)

    loss = ce + cfg.LAMBDA_MANHATTAN * manhattan
    return loss, acc, (logits.detach(), qry_y.detach())


# =============================================================================
# 7. TRAIN + EVAL
# =============================================================================

def train_model(index, cfg=CFG):
    model = SiameseEuclidean(cfg).to(cfg.DEVICE)
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
    plt.title(f"BreakHis {cfg.N_WAY}-way few-shot — Euclidean + Manhattan Siamese")
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
    plt.legend(); plt.title(f"t-SNE of Euclidean embeddings (test pool, {k_shot}-shot model)")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, f"tsne_embeddings_{k_shot}shot.png"), dpi=130)
    plt.close()


def fig_gradcam(model, index, cfg=CFG):
    """Grad-CAM on the last conv block, w.r.t. the norm of the global
    Euclidean embedding (a proxy 'how strongly this region drives the
    embedding')."""
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
    index = BreakHisIndex(cfg)

    print("Saving CLAHE + episode illustration figures...")
    fig_clahe(index, cfg)
    fig_episode(index, cfg, k_shot=3)

    print("Training the Siamese Euclidean model (episodic)...")
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
