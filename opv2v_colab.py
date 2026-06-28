"""OPV2V grounding for Direction 1  (run this one on Google Colab, not locally).

Two measurements that replace assumptions in the simulator with data:

  A) rho check : is the paper's geometric overlap (Eq. 5) a good proxy for
                 real semantic redundancy? We compare circle-overlap Jaccard
                 between cars against CLIP-embedding similarity of what their
                 cameras actually see, and count how often the geometric
                 ranking gets pair order wrong.
  B) Theta curve: how does compressing an image really damage its semantics?
                 JPEG quality sweep -> measured compression ratio eta vs
                 embedding distortion. The log fit gives the a_D constant
                 (and residual sigma) used by the simulator + conformal layer.

Setup (Colab):
  1. Get the OPV2V *test* split (smallest, ~? GB) from the OpenCOOD page:
     github.com/DerrickXuNu/OpenCOOD  -> data intro -> Google Drive / UCLA Box.
  2. Unzip into your Drive so you have  .../opv2v/test/<scenario>/<cav_id>/
     with files like 000068.yaml and 000068_camera0.png
  3. Point DATA_ROOT below at that test folder, then Runtime > Run all.

pip (first cell):  !pip -q install transformers pillow pyyaml scipy matplotlib
"""

import os, glob, io, itertools, random
import numpy as np
import yaml
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

DATA_ROOT   = "/content/drive/MyDrive/opv2v/test"   # <-- edit me
N_SCENARIOS = 8        # how many scenarios to sample
N_FRAMES    = 4        # timestamps per scenario
R_SENSE     = 70.0     # assumed sensing radius per car [m]
OUT_DIR     = "opv2v_out"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def embed(img):
    with torch.no_grad():
        x = proc(images=img, return_tensors="pt").to(device)
        v = clip.get_image_features(**x)[0]
    v = v / v.norm()
    return v.cpu().numpy()


def circle_jaccard(p1, p2, r=R_SENSE):
    """Closed-form Jaccard of two equal disks (lens area / union)."""
    d = float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))
    if d >= 2 * r:
        return 0.0
    if d < 1e-9:
        return 1.0
    inter = 2 * r * r * np.arccos(d / (2 * r)) - 0.5 * d * np.sqrt(4 * r * r - d * d)
    union = 2 * np.pi * r * r - inter
    return inter / union


def agent_pose(yml_path):
    with open(yml_path) as f:
        y = yaml.safe_load(f)
    # OPV2V stores ego pose as [x, y, z, roll, yaw, pitch]
    pose = y.get("true_ego_pos", y.get("lidar_pose"))
    return np.array(pose[:2], dtype=float)


# ---------------------------------------------------------------- part A
print("Part A: geometric rho vs CLIP semantic similarity")
rows = []
scenarios = sorted(glob.glob(os.path.join(DATA_ROOT, "*")))[:N_SCENARIOS]
for sc in scenarios:
    cavs = [c for c in sorted(glob.glob(os.path.join(sc, "*"))) if os.path.isdir(c)]
    if len(cavs) < 2:
        continue
    stamps = sorted({os.path.basename(f)[:6] for f in
                     glob.glob(os.path.join(cavs[0], "*_camera0.png"))})
    random.seed(0)
    for ts in random.sample(stamps, min(N_FRAMES, len(stamps))):
        feats, poss = {}, {}
        for cav in cavs:
            img_p = os.path.join(cav, ts + "_camera0.png")
            yml_p = os.path.join(cav, ts + ".yaml")
            if not (os.path.exists(img_p) and os.path.exists(yml_p)):
                continue
            feats[cav] = embed(Image.open(img_p).convert("RGB"))
            poss[cav] = agent_pose(yml_p)
        for a, b in itertools.combinations(feats, 2):
            rows.append({
                "scenario": os.path.basename(sc), "ts": ts,
                "rho_geo": circle_jaccard(poss[a], poss[b]),
                "rho_sem": float(np.dot(feats[a], feats[b])),
            })

geo = np.array([r["rho_geo"] for r in rows])
sem = np.array([r["rho_sem"] for r in rows])
rs, _ = spearmanr(geo, sem)
inv = 0; tot = 0
for (i, j) in itertools.combinations(range(len(rows)), 2):
    if abs(geo[i] - geo[j]) > 0.02:
        tot += 1
        if (geo[i] - geo[j]) * (sem[i] - sem[j]) < 0:
            inv += 1
print("pairs: %d   Spearman rho: %.3f   ranking inversions: %.1f%%"
      % (len(rows), rs, 100 * inv / max(tot, 1)))

with open(os.path.join(OUT_DIR, "rho_pairs.csv"), "w") as f:
    f.write("scenario,ts,rho_geo,rho_sem\n")
    for r in rows:
        f.write("%(scenario)s,%(ts)s,%(rho_geo).4f,%(rho_sem).4f\n" % r)

plt.figure(figsize=(5, 4))
plt.scatter(geo, sem, s=12, alpha=.5)
plt.xlabel("geometric Jaccard (paper Eq. 5)")
plt.ylabel("CLIP cosine similarity")
plt.title("proxy check: Spearman=%.2f, inversions=%.0f%%" % (rs, 100 * inv / max(tot, 1)))
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "F2_rho_mismatch.png"), dpi=150)

# ---------------------------------------------------------------- part B
print("Part B: measured distortion-vs-compression curve")
imgs = glob.glob(os.path.join(DATA_ROOT, "*", "*", "*_camera0.png"))
random.seed(1)
imgs = random.sample(imgs, min(24, len(imgs)))
etas, dists = [], []
for pth in imgs:
    im = Image.open(pth).convert("RGB")
    buf0 = io.BytesIO(); im.save(buf0, "JPEG", quality=95)
    e0 = embed(im); n0 = buf0.tell()
    for q in [70, 50, 35, 25, 15, 8]:
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=q)
        imq = Image.open(io.BytesIO(buf.getvalue()))
        etas.append(buf.tell() / n0)
        dists.append(1.0 - float(np.dot(e0, embed(imq))))

etas, dists = np.array(etas), np.array(dists)
x = np.log(1.0 / etas)
a_D = float(np.sum(x * dists) / np.sum(x * x))      # least squares through 0
sigma = float(np.std(dists - a_D * x))
print("fit:  D(eta) = %.4f * ln(1/eta)    residual sigma = %.4f" % (a_D, sigma))

with open(os.path.join(OUT_DIR, "theta_curve.csv"), "w") as f:
    f.write("eta,distortion\n")
    for e, d in zip(etas, dists):
        f.write("%.4f,%.5f\n" % (e, d))

xx = np.linspace(x.min(), x.max(), 50)
plt.figure(figsize=(5, 4))
plt.scatter(x, dists, s=12, alpha=.5, label="measured")
plt.plot(xx, a_D * xx, "r-", label="fit a_D=%.3f" % a_D)
plt.xlabel("ln(1/eta)  (compression depth)")
plt.ylabel("semantic distortion 1-cos")
plt.title("empirical Theta map (feeds the simulator + conformal sigma)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "F3_theta_curve.png"), dpi=150)
print("done ->", OUT_DIR)
