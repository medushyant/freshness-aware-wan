"""Synthetic multi-view street scenes with EXACT ground truth (playbook §4.1
fallback, always implemented — WP-3 is 'synthetic-grounded' locally; OPV2V
runs in notebooks/03_opv2v_pipeline.ipynb on Colab).

A scene is a procedurally drawn top-down street: road, buildings, and a set of
objects (car/bus/truck/person/bicycle) with known class, color, and position.
K agents get overlapping crops of the scene; each crop's ground-truth facts
(objects whose center falls inside it) and every pairwise crop IoU are known
EXACTLY — the airtight reference the VLM pipeline is scored against.
"""

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ("car", "bus", "truck", "person")
COLORS = {"red": (220, 30, 30), "blue": (30, 70, 220), "green": (25, 170, 60),
          "yellow": (245, 210, 20), "white": (245, 245, 245), "black": (15, 15, 15)}
SIZE = {"car": (110, 52), "bus": (200, 58), "truck": (160, 62),
        "person": (34, 60)}


def make_scene(seed, w=1280, h=840, n_obj=(3, 6)):
    """Simple 2D street illustration, drawn large and high-contrast so a small
    VLM can genuinely read it; object classes/colors/centres are EXACT GT."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (150, 160, 150))
    dr = ImageDraw.Draw(img)
    road_y = h // 2
    dr.rectangle([0, road_y - 130, w, road_y + 130], fill=(70, 70, 75))
    for x in range(20, w, 130):
        dr.rectangle([x, road_y - 4, x + 60, road_y + 4], fill=(230, 230, 200))
    objects = []
    n = int(rng.integers(*n_obj))
    placed = []
    for oid in range(n):
        cls = CLASSES[rng.integers(0, len(CLASSES))]
        col = list(COLORS)[rng.integers(0, len(COLORS))]
        ow, oh = SIZE[cls]
        for _try in range(30):
            x = rng.uniform(ow, w - 1.5 * ow)
            y = (rng.uniform(road_y - 115, road_y + 115 - oh)
                 if cls != "person" else rng.uniform(30, h - 30 - oh))
            if all(abs(x - px) > ow + pw or abs(y - py) > oh + ph
                   for px, py, pw, ph in placed):
                break
        placed.append((x, y, ow, oh))
        if cls == "person":
            hx, hy = x + ow / 2, y + ow / 3
            dr.ellipse([hx - ow / 3, hy - ow / 3, hx + ow / 3, hy + ow / 3],
                       fill=(235, 200, 170), outline=(0, 0, 0), width=3)
            dr.rounded_rectangle([x + 4, y + ow * 0.7, x + ow - 4, y + oh],
                                 radius=8, fill=COLORS[col],
                                 outline=(0, 0, 0), width=3)
        else:
            dr.rounded_rectangle([x, y, x + ow, y + oh], radius=10,
                                 fill=COLORS[col], outline=(0, 0, 0), width=4)
            dr.rectangle([x + 10, y + 8, x + min(ow * 0.3, 45) + 10, y + oh - 8],
                         fill=(175, 215, 235))
            for wx in (x + 18, x + ow - 30):
                dr.ellipse([wx, y + oh - 6, wx + 22, y + oh + 12], fill=(20, 20, 20))
        objects.append({"id": oid, "cls": cls, "color": col,
                        "cx": float(x + ow / 2), "cy": float(y + oh / 2)})
    return img, objects


def crop_views(img, objects, k=4, frac=0.55, seed=0):
    """K overlapping crops (agent views). Returns per-view image, EXACT facts,
    crop box, and the pairwise IoU matrix of the boxes."""
    rng = np.random.default_rng(seed + 500)
    w, h = img.size
    cw, chh = int(w * frac), int(h * frac)
    views = []
    for a in range(k):
        x0 = rng.uniform(0, w - cw)
        y0 = rng.uniform(0, h - chh)
        box = (x0, y0, x0 + cw, y0 + chh)
        facts = [o for o in objects
                 if box[0] <= o["cx"] <= box[2] and box[1] <= o["cy"] <= box[3]]
        views.append({"agent": a, "img": img.crop(tuple(map(int, box))),
                      "box": box, "facts": facts})
    iou = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            iou[a, b] = _iou(views[a]["box"], views[b]["box"])
    return views, iou


def _iou(b1, b2):
    ix = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    iy = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    inter = ix * iy
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if inter > 0 else 0.0


def fact_key(f):
    return (f["cls"], f["color"])


def fact_f1(pred_facts, true_facts):
    """Class+color multiset F1 — the report-quality metric."""
    from collections import Counter
    P = Counter((f.get("object", f.get("cls", "?")).lower(),
                 str(f.get("attr", f.get("color", "?"))).lower())
                for f in pred_facts)
    T = Counter((f["cls"], f["color"]) for f in true_facts)
    tp = sum(min(P[k], T[k]) for k in P)
    prec = tp / max(sum(P.values()), 1)
    rec = tp / max(sum(T.values()), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec
