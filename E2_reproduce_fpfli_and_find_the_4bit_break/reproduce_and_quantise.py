"""
reproduce_and_quantise.py

Part 1: Reproduce Dr David Li's FPFLI results (Xiao, Sapermsap, Chen and Li, Optica 2023)
        using the group's own code, pretrained models and sample data.
Part 2: My extension. Quantise the local lifetime network (LLE) to INT8 and INT4
        and measure how much the final lifetime images change.

Run from inside the Evaluation folder:
    python reproduce_and_quantise.py
Results go to ./my_results
"""
import os, json, time, copy, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")

# newer PyTorch refuses to load older checkpoints unless weights_only is False
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False, "map_location": k.get("map_location", "cpu")})

import scipy.io as io
import mat73
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import Pred_local_lifetime as PLL
from Prep_exp_data import prep_exp_data
from Neural_implict_interploation import NIII_process

OUT = "my_results"
os.makedirs(OUT, exist_ok=True)
PATCH = 8


def mask_hist(hist, thres, strict):
    inten = hist.sum(2)
    keep = inten > thres if strict else inten >= thres
    hist = hist * keep[..., None]
    return hist, inten * keep


def run_fpfli(hist, irf, thres, strict, lle_path, quant_bits=None):
    hist_m, mask = mask_hist(hist.copy(), thres, strict)
    t0 = time.time()
    lr_tau = PLL.Pred_local_lifetime(hist_m, irf, scaling_ratio=PATCH, h=0.039, file_path=lle_path)
    if quant_bits:
        quantise_model(PLL.model, quant_bits)
    lr_tau = lr_tau.eval_local_tau()
    nii = prep_exp_data(mask, lr_tau).prep_data()
    pred = np.squeeze(NIII_process(nii, ratio=PATCH), (0, 1))
    pred[pred < 0] = 0
    return pred, mask, time.time() - t0, lr_tau


def quantise_model(model, bits):
    """Per output channel symmetric fake quantisation of every weight tensor."""
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() < 2:
                continue
            w = p.data
            flat = w.reshape(w.shape[0], -1)
            scale = flat.abs().max(1, keepdim=True).values.clamp(min=1e-12) / qmax
            q = torch.round(flat / scale).clamp(-qmax, qmax) * scale
            p.data.copy_(q.reshape(w.shape))


def mae(pred, gt, mask):
    m = (mask > 0) & (gt > 0)
    return float(np.mean(np.abs(pred[m] - gt[m])))


def show(imgs, titles, vmax, fname, vmin=0):
    fig, ax = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 4))
    for a, im, t in zip(ax, imgs, titles):
        h = a.imshow(im, cmap="inferno", vmin=vmin, vmax=vmax)
        a.set_title(t, fontsize=10); a.axis("off")
    fig.colorbar(h, ax=ax, shrink=0.8, label="lifetime (ns)")
    fig.savefig(os.path.join(OUT, fname), dpi=130, bbox_inches="tight"); plt.close(fig)


results = {}
irf_g = io.loadmat("sample_data/irf_gaussian.mat")["irf_gaussian"].astype("float32")
LLE_A = "./model_parameters/LLE_parameter.pth"
LLE_B = "./model_parameters/LLE_parameter_1_3ns.pth"

# ---------- Part 1a: low light synthetic FLIM images (ground truth known) ----------
synth = {}
for s in ["1", "2", "3", "4", "5", "6"]:
    d = io.loadmat(f"sample_data/{s}.mat")
    hist = d["Hist"].astype("float32"); gt = d["tau_gt"].astype("float32")
    row = {"photons_per_pixel": float(hist.sum(2)[hist.sum(2) > 3].mean())}
    imgs = {}
    for bits in [None, 8, 4]:
        pred, mask, t, _ = run_fpfli(hist, irf_g, 3, True, LLE_A, bits)
        key = "FP32" if bits is None else f"INT{bits}"
        row[key + "_MAE_ns"] = mae(pred, gt, mask); row[key + "_seconds"] = t
        imgs[key] = pred
    row["INT8_vs_FP32_median_change_ns"] = float(np.median(np.abs(imgs["INT8"] - imgs["FP32"])[mask > 0]))
    row["INT4_vs_FP32_median_change_ns"] = float(np.median(np.abs(imgs["INT4"] - imgs["FP32"])[mask > 0]))
    synth[s] = row
    if s == "3":
        show([hist.sum(2), gt, imgs["FP32"], imgs["INT8"], imgs["INT4"]],
             ["Photon counts", "True lifetime", "FPFLI (FP32)", "FPFLI, LLE at INT8", "FPFLI, LLE at INT4"],
             4, "fig1_synthetic_sample3.png")
    print("sample", s, {k: round(v, 3) for k, v in row.items()})
results["synthetic_low_light"] = synth

# ---------- Part 1b: real HEK293 cells with gold nanorods ----------
d = io.loadmat("sample_data/hek_low.mat")
hist = d["hist"].astype("float32"); irf = d["irf"].astype("float32")
gt = mat73.loadmat("sample_data/hek_gt.mat")["tau_img_gt"].astype("float32")
hek = {}; imgs = {}
for bits in [None, 8, 4]:
    pred, mask, t, _ = run_fpfli(hist, irf, 3, True, LLE_B, bits)
    key = "FP32" if bits is None else f"INT{bits}"
    hek[key + "_MAE_ns"] = mae(pred, gt, mask); hek[key + "_seconds"] = t; imgs[key] = pred
show([gt, imgs["FP32"], imgs["INT8"], imgs["INT4"]],
     ["Reference (high photon)", "FPFLI from few photons", "LLE at INT8", "LLE at INT4"], 2.5, "fig2_hek293_real.png")
fig, a = plt.subplots(figsize=(5, 3.5))
b = np.arange(0.5, 3, 0.05)
a.hist(gt[mask > 0], b, alpha=.5, label="reference"); a.hist(imgs["FP32"][mask > 0], b, alpha=.5, label="FPFLI FP32")
a.hist(imgs["INT8"][mask > 0], b, histtype="step", lw=2, label="FPFLI INT8")
a.set_xlabel("lifetime (ns)"); a.set_ylabel("pixels"); a.legend()
fig.savefig(os.path.join(OUT, "fig3_hek293_histogram.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
results["hek293_real"] = hek
print("HEK293", {k: round(v, 3) for k, v in hek.items()})

# ---------- Part 1c: real microspheres (no ground truth, two lifetimes should separate) ----------
irf_m = mat73.loadmat("sample_data/irf_measure.mat")["irf"].astype("float32")
for name in ["microsphere_2p_1", "microsphere_2p_2"]:
    hist = mat73.loadmat(f"sample_data/{name}.mat")["Hist"].astype("float32")
    p32, mask, t, _ = run_fpfli(hist, irf_m, 4, False, LLE_A)
    p8, _, _, _ = run_fpfli(hist, irf_m, 4, False, LLE_A, 8)
    show([hist.sum(2), p32, p8], ["Photon counts", "FPFLI FP32", "FPFLI, LLE at INT8"], 4, f"fig4_{name}.png", vmin=1)
    results[name] = {"seconds": t, "INT8_vs_FP32_median_change_ns": float(np.median(np.abs(p8 - p32)[mask > 0]))}

json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("Saved to", OUT)
