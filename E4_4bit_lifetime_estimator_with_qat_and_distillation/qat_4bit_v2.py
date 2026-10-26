"""
qat_4bit_v2.py  (Forhad: optimising the 4 bit LLE further)

Compared with qat_4bit.py (v1), three upgrades, tested step by step:
  V2 = v1 + knowledge distillation from Dr Li's FP32 network (teacher) + training data matched to his
       images (his exact Gaussian IRF for the general model, the MEASURED HEK293 IRF for the 1 to 3 ns
       model, photon counts like his 8x8 patches)
  V3 = V2 + learned step size quantisation (LSQ: each channel learns its best 4 bit scale)
  V4 = V3 + the tiny layers (two 128 weight input embeddings and the two smallest output layers,
       0.4% of all weights) kept at 8 bit
All variants are scored with the FULL FPFLI pipeline on his sample data, like v1.
Run with run_qat_v2_on_gpu.bat. Results: my_qat_v2_results/
"""
import os, json, time, copy, warnings, argparse, builtins
import numpy as np
import torch, torch.nn as nn
import torch.nn.utils.parametrize as P_
warnings.filterwarnings("ignore")
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False, "map_location": k.get("map_location", "cpu")})
import scipy.io as io, mat73
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tau_net import tauNet
import Pred_local_lifetime as PLL
from Prep_exp_data import prep_exp_data
from Neural_implict_interploation import NIII_process

ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=400_000)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke: args.n_train, args.epochs = 3000, 1
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "", flush=True)
OUT = "my_qat_v2_results"; os.makedirs(OUT, exist_ok=True)
H, NB, MULT, PATCH = 0.039, 256, 0.039 * 100, 8
T0 = time.time()
irf_g = io.loadmat("sample_data/irf_gaussian.mat")["irf_gaussian"].astype("float32").ravel()
hek = io.loadmat("sample_data/hek_low.mat"); irf_hek = hek["irf"].astype("float32").ravel()
TINY = ["embd_decay", "embd_irf", "ponitconv2", "ponitconv3"]

# ------------------------------------------------------------ data matched to his images
def make_data(n, tau_lo, tau_hi, irf_base, rng):
    X, I, Y = [], [], []
    t = np.arange(1, NB + 1)
    for c in range(0, n, 20000):
        m = min(20000, n - c)
        two = rng.random(m) < 0.5
        t1 = rng.uniform(tau_lo, tau_hi, m); t2 = np.where(two, rng.uniform(tau_lo, tau_hi, m), t1)
        f1 = np.where(two, rng.uniform(0, 1, m), 1.0); f2 = 1 - f1
        yo = f1[:, None] * np.exp(-t[None] * H / t1[:, None]) + f2[:, None] * np.exp(-t[None] * H / t2[:, None])
        tau_ave = (f1 * t1 ** 2 + f2 * t2 ** 2) / (f1 * t1 + f2 * t2)
        # his IRF, shifted by up to half a bin and stretched by up to 10 percent
        sh = rng.uniform(-0.5, 0.5, m); st = rng.uniform(0.9, 1.1, m)
        pk = np.argmax(irf_base)
        src = pk + (t[None] - 1 - pk - sh[:, None]) / st[:, None]
        irf = np.stack([np.interp(s, np.arange(NB), irf_base, left=0, right=0) for s in src]).clip(0)
        irf = irf / irf.max(1, keepdims=True)
        lam = np.fft.irfft(np.fft.rfft(irf, 2 * NB) * np.fft.rfft(yo, 2 * NB), 2 * NB)[:, :NB].clip(0)
        N = 10 ** rng.uniform(np.log10(100), np.log10(3000), m)               # photons per 8x8 patch (his images: about 200 to 1300)
        y = rng.poisson(lam / lam.sum(1, keepdims=True) * N[:, None]).astype("float32")
        X.append(y / (y.max(1, keepdims=True) + 1e-8)); I.append(irf.astype("float32")); Y.append(tau_ave.astype("float32"))
    return np.concatenate(X), np.concatenate(I), np.concatenate(Y)

# ------------------------------------------------------------ quantisers
class MaxQuant(nn.Module):                      # v1 style: scale = max|w| / qmax per channel
    def __init__(self, bits): super().__init__(); self.qmax = 2 ** (bits - 1) - 1
    def forward(self, w):
        f = w.reshape(w.shape[0], -1)
        s = f.abs().max(1, keepdim=True).values.clamp(min=1e-12) / self.qmax
        q = (torch.round(f / s).clamp(-self.qmax, self.qmax) * s).reshape(w.shape)
        return w + (q - w).detach()

class LSQQuant(nn.Module):                      # learned step size (Esser et al. 2020), per channel
    def __init__(self, w, bits):
        super().__init__(); self.qmax = 2 ** (bits - 1) - 1
        f = w.detach().reshape(w.shape[0], -1)
        self.s = nn.Parameter(f.abs().max(1, keepdim=True).values.clamp(min=1e-8) / self.qmax * 0.9)
        self.g = 1.0 / np.sqrt(f.shape[1] * self.qmax)
    def forward(self, w):
        f = w.reshape(w.shape[0], -1)
        s = self.s * self.g + (self.s - self.s * self.g).detach()           # gradient scaling
        v = (f / s).clamp(-self.qmax, self.qmax)
        q = (v + (torch.round(v) - v).detach()) * s
        return q.reshape(w.shape)

def convs(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv1d)]

def bits_for(name, variant):
    return 8 if (variant == "V4" and any(name == t for t in TINY)) else 4

def train(weights, tau_lo, tau_hi, irf_base, variant, data):
    X, I, Y, Xv, Iv, Yv = data
    teacher = tauNet(); teacher.load_state_dict(torch.load(weights)); teacher.to(DEV).eval()
    model = tauNet(); model.load_state_dict(torch.load(weights)); model.to(DEV)
    for name, mod in convs(model):
        b = bits_for(name, variant)
        P_.register_parametrization(mod, "weight", MaxQuant(b) if variant == "V2" else LSQQuant(mod.weight, b))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    steps = args.epochs * (len(X) // 512)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=max(1, steps), pct_start=0.15)
    Xt, It, Yt = torch.tensor(X), torch.tensor(I), torch.tensor(Y)
    def val(mm):
        mm.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(Xv), 4096):
                out.append(mm(torch.tensor(Xv[i:i+4096]).unsqueeze(1).to(DEV), torch.tensor(Iv[i:i+4096]).unsqueeze(1).to(DEV)).cpu() * MULT)
        return float(np.median(np.abs(torch.cat(out).numpy() - Yv)))
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt) - 511, 512):
            b = perm[i:i + 512]; x = Xt[b].unsqueeze(1).to(DEV); ir = It[b].unsqueeze(1).to(DEV)
            with torch.no_grad(): tt = teacher(x, ir) * MULT
            p = model(x, ir) * MULT
            loss = nn.functional.smooth_l1_loss(p, Yt[b].to(DEV), beta=0.1) + nn.functional.smooth_l1_loss(p, tt, beta=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            print(f"  [{variant}] epoch {ep + 1}/{args.epochs} val median error {val(model):.4f} ns ({time.time() - T0:.0f} s)", flush=True)
    # recalibrate BatchNorm statistics with the final quantised weights
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d): m.reset_running_stats(); m.momentum = None
    model.train()
    with torch.no_grad():
        for i in range(0, min(len(Xt), 100_000), 1024):
            model(Xt[i:i+1024].unsqueeze(1).to(DEV), It[i:i+1024].unsqueeze(1).to(DEV))
    res = {"simulated_val_median_error_ns": val(model), "teacher_FP32_val_median_error_ns": val(teacher)}
    for _, mod in convs(model): P_.remove_parametrizations(mod, "weight", leave_parametrized=True)
    model.cpu()
    kbits = sum(m.weight.numel() * bits_for(n, variant) for n, m in convs(model))
    res["weight_KB"] = kbits / 8 / 1024
    res["max_levels_4bit_layers"] = max(len(torch.unique(m.weight[c])) for n, m in convs(model) if bits_for(n, variant) == 4 for c in range(m.weight.shape[0]))
    path = weights.replace(".pth", f"_{variant}.pth"); torch.save(model.state_dict(), path)
    print(f"  [{variant}] {json.dumps(res)}", flush=True)
    return path, res

# ------------------------------------------------------------ pipeline scoring (same as v1)
_print = builtins.print
def quiet(f, *a, **k):
    builtins.print = lambda *x, **y: None
    try: return f(*a, **k)
    finally: builtins.print = _print
def run(hist, irf, path):
    inten = hist.sum(2); keep = inten > 3; hm = hist * keep[..., None]; mask = inten * keep
    lr = PLL.Pred_local_lifetime(hm, irf, scaling_ratio=PATCH, h=H, file_path=path).eval_local_tau()
    out = np.squeeze(quiet(NIII_process, prep_exp_data(mask, lr).prep_data(), ratio=PATCH), (0, 1)); out[out < 0] = 0
    return out, mask
def mae(p, g, m):
    k = (m > 0) & (g > 0); return float(np.mean(np.abs(p[k] - g[k])))
SYN = [(io.loadmat(f"sample_data/{s}.mat")) for s in (["3"] if args.smoke else ["1", "2", "3", "4", "5", "6"])]
HEKD = (hek["hist"].astype("float32"), hek["irf"].astype("float32"), mat73.loadmat("sample_data/hek_gt.mat")["tau_img_gt"].astype("float32"))
def score(pathA, pathB):
    syn = [mae(*run(d["Hist"].astype("float32"), irf_g[None], pathA), d["tau_gt"].astype("float32")) if False else None for d in []]
    syn = []
    for d in SYN:
        p, m = run(d["Hist"].astype("float32"), irf_g[None], pathA); syn.append(mae(p, d["tau_gt"].astype("float32"), m))
    p, m = run(HEKD[0].copy(), HEKD[1], pathB)
    return {"synthetic_per_image": syn, "synthetic_mean": float(np.mean(syn)), "HEK293": mae(p, HEKD[2], m)}

R = {"FP32 original": score("./model_parameters/LLE_parameter.pth", "./model_parameters/LLE_parameter_1_3ns.pth"),
     "V1 (previous QAT)": score("./model_parameters/LLE_parameter_QAT_INT4.pth", "./model_parameters/LLE_parameter_1_3ns_QAT_INT4.pth")}
for k in R: print(f"{k:28s} synthetic {R[k]['synthetic_mean']:.3f} ns  HEK293 {R[k]['HEK293']:.3f} ns", flush=True)
rng = np.random.default_rng(0)
dataA = (*make_data(args.n_train, 0.2, 4.5, irf_g, rng), *make_data(20000, 0.2, 4.5, irf_g, np.random.default_rng(1)))
dataB = (*make_data(args.n_train, 1.0, 3.0, irf_hek, rng), *make_data(20000, 1.0, 3.0, irf_hek, np.random.default_rng(2)))
print(f"data ready ({time.time() - T0:.0f} s)", flush=True)
for V in ["V2", "V3", "V4"]:
    print(f"=== {V} ===", flush=True)
    pA, rA = train("./model_parameters/LLE_parameter.pth", 0.2, 4.5, irf_g, V, dataA)
    pB, rB = train("./model_parameters/LLE_parameter_1_3ns.pth", 1.0, 3.0, irf_hek, V, dataB)
    R[V] = score(pA, pB); R[V]["train_general"] = rA; R[V]["train_1_3ns"] = rB; R[V]["weight_KB"] = rA["weight_KB"]
    print(f"{V:28s} synthetic {R[V]['synthetic_mean']:.3f} ns  HEK293 {R[V]['HEK293']:.3f} ns  ({R[V]['weight_KB']:.1f} KB)", flush=True)
    json.dump(R, open(f"{OUT}/qat_v2_results.json", "w"), indent=2)

names = ["FP32 original", "V1 (previous QAT)", "V2", "V3", "V4"]
labels = ["FP32\n107 KB", "4 bit v1\n13 KB", "4 bit v2\n+distill, matched data", "4 bit v3\n+learned scales", f"4 bit v4\n+tiny layers 8 bit\n{R['V4']['weight_KB']:.1f} KB"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a, key, t in [(ax[0], "synthetic_mean", "His 6 synthetic images, mean error"), (ax[1], "HEK293", "Real HEK293 cells, error")]:
    v = [R[n][key] for n in names]
    bars = a.bar(range(len(v)), v, color=["#7f8c8d", "#c0392b", "#e67e22", "#2980b9", "#27ae60"])
    a.set_xticks(range(len(v))); a.set_xticklabels(labels, fontsize=7.5); a.set_ylabel("mean absolute error (ns)"); a.set_title(t)
    for b_, x in zip(bars, v): a.text(b_.get_x() + b_.get_width() / 2, x, f"{x:.3f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_qat_v2_summary.png", dpi=130); plt.close(fig)
print(f"ALL DONE ({time.time() - T0:.0f} s)", flush=True)
