"""
qat_4bit.py  (Forhad's fix for Experiment A)

Problem found in Experiment A: plain 4 bit quantisation of Dr Li's LLE network breaks it
(lifetime error jumps from about 0.1 ns to over 1 ns), mostly because of the final dense layer.

Fix: quantisation aware training (QAT).
  * Training data made in Python exactly like his Matlab generator
    (Fluorescence_multi_decay_nonhomopp.m): mono or bi exponential decays, 256 bins of 39 ps,
    convolved with a Gaussian IRF (FWHM about 0.167 ns), Poisson photon noise.
  * Every weight layer is "fake quantised" to 4 bit (per output channel, symmetric) in the forward pass,
    and gradients pass straight through (straight through estimator), starting from his trained weights.
  * The final weights are saved already rounded to the 4 bit grid.
  * Then the FULL FPFLI pipeline (LLE + NIII) is rerun on his sample data and compared with
    FP32, INT8 and plain INT4.

Run with run_qat_on_gpu.bat (uses the RTX 3090). Results go to my_qat_results/.
"""
import os, sys, json, time, copy, warnings, argparse
import numpy as np
import torch, torch.nn as nn
import torch.nn.utils.parametrize as P_
warnings.filterwarnings("ignore")
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False, "map_location": k.get("map_location", "cpu")})
import scipy.io as io, mat73
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import builtins
from tau_net import tauNet
import Pred_local_lifetime as PLL
from Prep_exp_data import prep_exp_data
from Neural_implict_interploation import NIII_process

ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=600_000)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--smoke", action="store_true", help="tiny run just to check the code")
args = ap.parse_args()
if args.smoke: args.n_train, args.epochs = 4000, 1
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "", flush=True)
OUT = "my_qat_results"; os.makedirs(OUT, exist_ok=True)
H, NB, MULT, PATCH = 0.039, 256, 0.039 * 100, 8
T0 = time.time()

# ------------------------------------------------------------ data like his Matlab generator
def make_data(n, tau_lo, tau_hi, rng):
    X, I, Y = [], [], []
    t = np.arange(1, NB + 1)
    for c in range(0, n, 20000):
        m = min(20000, n - c)
        two = rng.random(m) < 0.5
        t1 = rng.uniform(tau_lo, tau_hi, m); t2 = np.where(two, rng.uniform(tau_lo, tau_hi, m), t1)
        f1 = np.where(two, rng.uniform(0, 1, m), 1.0); f2 = 1 - f1
        yo = f1[:, None] * np.exp(-t[None] / (t1[:, None] / H)) + f2[:, None] * np.exp(-t[None] / (t2[:, None] / H))
        tau_ave = (f1 * t1 ** 2 + f2 * t2 ** 2) / (f1 * t1 + f2 * t2)          # intensity weighted average lifetime
        t0 = rng.uniform(12, 16, m); fwhm = rng.uniform(0.13, 0.30, m)       # IRF position and width (in his data: bin 14, 0.167 ns)
        sig = fwhm / 2.3548 / H
        irf = np.exp(-(t[None] - t0[:, None]) ** 2 / (2 * sig[:, None] ** 2))
        lam = np.fft.irfft(np.fft.rfft(irf, 2 * NB) * np.fft.rfft(yo, 2 * NB), 2 * NB)[:, :NB].clip(0)
        N = 10 ** rng.uniform(np.log10(50), np.log10(5000), m)               # photons per 8x8 patch
        y = rng.poisson(lam / lam.sum(1, keepdims=True) * N[:, None]).astype("float32")
        X.append(y / (y.max(1, keepdims=True) + 1e-8)); I.append(irf.astype("float32")); Y.append(tau_ave.astype("float32"))
    return np.concatenate(X), np.concatenate(I), np.concatenate(Y)

# ------------------------------------------------------------ 4 bit fake quantisation with STE
class FakeQuant(nn.Module):
    def __init__(self, bits): super().__init__(); self.qmax = 2 ** (bits - 1) - 1
    def forward(self, w):
        flat = w.reshape(w.shape[0], -1)
        s = flat.abs().max(1, keepdim=True).values.clamp(min=1e-12) / self.qmax
        q = (torch.round(flat / s).clamp(-self.qmax, self.qmax) * s).reshape(w.shape)
        return w + (q - w).detach()                                           # straight through estimator

def quant_layers(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, nn.Conv1d)]

def ptq(sd_model, bits):
    m = copy.deepcopy(sd_model)
    with torch.no_grad():
        for mod, _ in quant_layers(m): mod.weight.copy_(FakeQuant(bits)(mod.weight))
    return m

def train_qat(weights, tau_lo, tau_hi, tag):
    rng = np.random.default_rng(0)
    X, I, Y = make_data(args.n_train, tau_lo, tau_hi, rng)
    Xv, Iv, Yv = make_data(20000, tau_lo, tau_hi, np.random.default_rng(1))
    print(f"[{tag}] data ready ({time.time() - T0:.0f} s)", flush=True)
    model = tauNet(); model.load_state_dict(torch.load(weights)); model.to(DEV)
    def val(mm):
        mm.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(Xv), 4096):
                out.append(mm(torch.tensor(Xv[i:i+4096]).unsqueeze(1).to(DEV), torch.tensor(Iv[i:i+4096]).unsqueeze(1).to(DEV)).cpu() * MULT)
        return float(np.median(np.abs(torch.cat(out).numpy() - Yv)))
    res = {"FP32 original": val(model), "INT4 plain (PTQ)": val(ptq(model, 4))}
    for mod, name in quant_layers(model): P_.register_parametrization(mod, name, FakeQuant(4))
    res["INT4 QAT, before training"] = val(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    steps = args.epochs * (len(X) // 512)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=max(steps, 1))
    Xt, It, Yt = torch.tensor(X), torch.tensor(I), torch.tensor(Y)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0
        for i in range(0, len(Xt) - 511, 512):
            b = perm[i:i + 512]
            pred = model(Xt[b].unsqueeze(1).to(DEV), It[b].unsqueeze(1).to(DEV)) * MULT
            loss = nn.functional.smooth_l1_loss(pred, Yt[b].to(DEV), beta=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print(f"[{tag}] epoch {ep + 1}/{args.epochs} loss {tot / max(1, len(Xt) // 512):.4f} val {val(model):.4f} ns ({time.time() - T0:.0f} s)", flush=True)
    res["INT4 QAT, after training"] = val(model)
    for mod, name in quant_layers(model): P_.remove_parametrizations(mod, name, leave_parametrized=True)  # bake 4 bit weights
    model.cpu()
    levels = max(len(torch.unique(mod.weight[c])) for mod, _ in quant_layers(model) for c in range(mod.weight.shape[0]))
    res["max distinct weight values per channel (must be 15 or fewer)"] = levels
    path = weights.replace(".pth", "_QAT_INT4.pth")
    torch.save(model.state_dict(), path)
    print(f"[{tag}] simulated validation:", json.dumps(res), flush=True)
    return path, res

R = {}
qA, R["LLE (general model) simulated"] = train_qat("./model_parameters/LLE_parameter.pth", 0.1, 5.0, "general")
qB, R["LLE 1 to 3 ns model (HEK293) simulated"] = train_qat("./model_parameters/LLE_parameter_1_3ns.pth", 1.0, 3.0, "1-3ns")

# ------------------------------------------------------------ full FPFLI pipeline on his sample data
_print = builtins.print
def quiet(f, *a, **k):
    builtins.print = lambda *x, **y: None
    try: return f(*a, **k)
    finally: builtins.print = _print

def run(hist, irf, path, bits=None):
    inten = hist.sum(2); keep = inten > 3
    hm = hist * keep[..., None]; mask = inten * keep
    o = PLL.Pred_local_lifetime(hm, irf, scaling_ratio=PATCH, h=H, file_path=path)
    if bits:
        with torch.no_grad():
            for mod, _ in quant_layers(PLL.model): mod.weight.copy_(FakeQuant(bits)(mod.weight))
    lr = o.eval_local_tau()
    out = np.squeeze(quiet(NIII_process, prep_exp_data(mask, lr).prep_data(), ratio=PATCH), (0, 1)); out[out < 0] = 0
    return out, mask

def mae(p, g, m):
    k = (m > 0) & (g > 0); return float(np.mean(np.abs(p[k] - g[k])))

irf_g = io.loadmat("sample_data/irf_gaussian.mat")["irf_gaussian"].astype("float32")
CFG = [("FP32 (original)", "A", None), ("INT8 plain", "A", 8), ("INT4 plain", "A", 4), ("INT4 QAT (my fix)", "Q", None)]
pipe = {c[0]: {} for c in CFG}; show = {}
for s in (["3"] if args.smoke else ["1", "2", "3", "4", "5", "6"]):
    d = io.loadmat(f"sample_data/{s}.mat"); hist = d["Hist"].astype("float32"); gt = d["tau_gt"].astype("float32")
    for name, w, bits in CFG:
        p, m = run(hist.copy(), irf_g, "./model_parameters/LLE_parameter.pth" if w == "A" else qA, bits)
        pipe[name][f"synthetic {s}"] = mae(p, gt, m)
        if s == "3": show[name] = p
    if s == "3": show = {"truth": gt, **show}
d = io.loadmat("sample_data/hek_low.mat"); hist = d["hist"].astype("float32"); irf = d["irf"].astype("float32")
gt = mat73.loadmat("sample_data/hek_gt.mat")["tau_img_gt"].astype("float32"); showH = {"reference": gt}
for name, w, bits in CFG:
    p, m = run(hist.copy(), irf, "./model_parameters/LLE_parameter_1_3ns.pth" if w == "A" else qB, bits)
    pipe[name]["HEK293 real cells"] = mae(p, gt, m); showH[name] = p
for name in pipe:
    syn = [v for k, v in pipe[name].items() if k.startswith("synthetic")]
    pipe[name]["synthetic mean"] = float(np.mean(syn))
    print(f"{name:20s} synthetic mean MAE {pipe[name]['synthetic mean']:.3f} ns   HEK293 MAE {pipe[name]['HEK293 real cells']:.3f} ns", flush=True)
nparam = sum(m.weight.numel() for m, _ in quant_layers(tauNet()))
R["full pipeline MAE (ns)"] = pipe
R["LLE weight memory KB"] = {"FP32": nparam * 4 / 1024, "INT8": nparam / 1024, "INT4": nparam / 2 / 1024}
json.dump(R, open(f"{OUT}/qat_results.json", "w"), indent=2)

for nm, S, vmax in [("synthetic3", show, 4), ("hek293", showH, 2.5)]:
    keys = list(S.keys()); fig, ax = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.8))
    for a, k in zip(ax, keys):
        h_ = a.imshow(S[k], cmap="inferno", vmin=0, vmax=vmax); a.set_title(k, fontsize=10); a.axis("off")
    fig.colorbar(h_, ax=ax, shrink=.8, label="lifetime (ns)")
    fig.savefig(f"{OUT}/fig_{nm}_qat.png", dpi=120, bbox_inches="tight"); plt.close(fig)
names = [c[0] for c in CFG]; kb = [R["LLE weight memory KB"][k] for k in ["FP32", "INT8", "INT4", "INT4"]]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, key, t in [(ax[0], "synthetic mean", "Synthetic images (6), mean error"), (ax[1], "HEK293 real cells", "Real HEK293 cells, error")]:
    v = [pipe[n][key] for n in names]
    bars = a.bar(range(4), v, color=["#7f8c8d", "#2980b9", "#c0392b", "#27ae60"])
    a.set_xticks(range(4)); a.set_xticklabels([f"{n}\n{k:.0f} KB" for n, k in zip(names, kb)], fontsize=8)
    a.set_ylabel("mean absolute error (ns)"); a.set_title(t)
    for b_, x in zip(bars, v): a.text(b_.get_x() + b_.get_width() / 2, x, f"{x:.2f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_qat_summary.png", dpi=130); plt.close(fig)
print(f"ALL DONE ({time.time() - T0:.0f} s)", flush=True)
