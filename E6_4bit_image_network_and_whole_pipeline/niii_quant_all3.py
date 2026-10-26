"""
niii_quant_all3.py  (Forhad: NIII at 4 bit with ALL THREE techniques: distillation + learned scales (LSQ) + first/last layers 8 bit)

Based on niii_quant.py (Forhad: can Dr Li's NIII network be quantised too?)

NIII is the big network in FPFLI (14.6 million weights, 58 MB at FP32). Three steps:
  1. INT8 weights, no retraining (post training quantisation, per output channel, symmetric)
  2. INT4 weights, no retraining
  3. INT4 weights with quantisation aware training and KNOWLEDGE DISTILLATION:
     the 4 bit "student" learns to copy the full precision "teacher" NIII.
     The practice images are NOT Dr Li's test images: they are built from the Glasgow SPAD camera
     intensity images (Zickus et al. 2020) with random lifetime regions, so the test stays fair.
Every version is scored with the FULL FPFLI pipeline on his sample data (his original FP32 LLE),
and finally the best 4 bit NIII is combined with my 4 bit LLE (V4).
Run with run_niii_on_gpu.bat. Results: my_niii_results/
"""
import os, io as pyio, json, time, copy, zipfile, warnings, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.utils.parametrize as P_
warnings.filterwarnings("ignore")
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False, "map_location": k.get("map_location", "cpu")})
import scipy.io as sio, mat73
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import builtins
import Pred_local_lifetime as PLL
from Prep_exp_data import prep_exp_data
from NIII_model import NIII

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--crop", type=int, default=128)
ap.add_argument("--glasgow", default=r"..\..\..\DCS_Quantisation_Mini_Study\datasets\SPAD_FLIM_Glasgow_Zickus2020\DATA_REPOSITORY_fluorescence_lifetime_imaging_with_a_megapixel_SPAD_camera_and_neural_network_lifetime_estimation")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke: args.steps, args.batch = 20, 2
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "", flush=True)
OUT = "my_niii_all3_results"; os.makedirs(OUT, exist_ok=True)
T0 = time.time(); PATCH, H = 8, 0.039
_print = builtins.print
def quiet(f, *a, **k):
    builtins.print = lambda *x, **y: None
    try: return f(*a, **k)
    finally: builtins.print = _print

# ------------------------------------------------------------------ quantisation helpers
class FakeQuant(nn.Module):
    def __init__(self, bits): super().__init__(); self.qmax = 2 ** (bits - 1) - 1
    def forward(self, w):
        flat = w.reshape(w.shape[0], -1)
        s = flat.abs().max(1, keepdim=True).values.clamp(min=1e-12) / self.qmax
        q = (torch.round(flat / s).clamp(-self.qmax, self.qmax) * s).reshape(w.shape)
        return w + (q - w).detach()

class LSQ(nn.Module):
    """Learned per output channel scale; rounding uses a straight through estimator, the scale gets real gradients."""
    def __init__(self, w, bits):
        super().__init__(); self.qmax = 2 ** (bits - 1) - 1
        init = w.detach().reshape(w.shape[0], -1).abs().max(1).values.clamp(min=1e-8) / self.qmax
        self.log_s = nn.Parameter(init.log())
    def forward(self, w):
        s = self.log_s.exp().view(-1, *([1] * (w.dim() - 1)))
        x = (w / s).clamp(-self.qmax, self.qmax)
        return (x + (torch.round(x) - x).detach()) * s

def qlayers(m):
    return [(n, mod) for n, mod in m.named_modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]

def ptq(model, bits):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for _, mod in qlayers(m): mod.weight.copy_(FakeQuant(bits)(mod.weight))
    return m

teacher = NIII(); teacher.load_state_dict(torch.load("./model_parameters/NIII_parameter_L8.pth.tar")["model"]); teacher.to(DEV).eval()
NW = sum(mod.weight.numel() for _, mod in qlayers(teacher))
print(f"NIII weights: {NW:,} ({NW * 4 / 1e6:.1f} MB at FP32)", flush=True)

def to_dev(d): return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in d.items()}

def niii_apply(model, mask, lr):
    d = to_dev(prep_exp_data(mask.astype("float32"), lr.astype("float32")).prep_data())
    with torch.no_grad():
        p = model(d) * d["lr_tau_max"]
    B, C, Hh, Ww = d["hr_int"].shape
    out = p.reshape(Hh, Ww).cpu().numpy(); out[out < 0] = 0
    return out

# ------------------------------------------------------------------ his test data through his FP32 LLE (fixed)
def lle(hist, irf, path):
    inten = hist.sum(2); keep = inten > 3
    hm = hist * keep[..., None]; mask = inten * keep
    return quiet(lambda: PLL.Pred_local_lifetime(hm, irf, scaling_ratio=PATCH, h=H, file_path=path).eval_local_tau()), mask

irf_g = sio.loadmat("sample_data/irf_gaussian.mat")["irf_gaussian"].astype("float32")
TESTS = []
for s in (["3"] if args.smoke else ["1", "2", "3", "4", "5", "6"]):
    d = sio.loadmat(f"sample_data/{s}.mat")
    TESTS.append((f"synthetic {s}", d["Hist"].astype("float32"), irf_g, d["tau_gt"].astype("float32"), "general"))
d = sio.loadmat("sample_data/hek_low.mat")
TESTS.append(("HEK293", d["hist"].astype("float32"), d["irf"].astype("float32"), mat73.loadmat("sample_data/hek_gt.mat")["tau_img_gt"].astype("float32"), "1-3ns"))
LLE_FILES = {"FP32": {"general": "./model_parameters/LLE_parameter.pth", "1-3ns": "./model_parameters/LLE_parameter_1_3ns.pth"},
             "V4 4 bit": {"general": "./model_parameters/LLE_parameter_V4.pth", "1-3ns": "./model_parameters/LLE_parameter_1_3ns_V4.pth"}}
LR = {}
for lname, files in LLE_FILES.items():
    if not all(os.path.exists(f) for f in files.values()): continue
    LR[lname] = [lle(h, i, files[kind]) for _, h, i, _, kind in TESTS]

def score(model, lname="FP32", keep=False):
    res, imgs = {}, {}
    for (name, _, _, gt, _), (lr, mask) in zip(TESTS, LR[lname]):
        p = niii_apply(model, mask, lr); k = (mask > 0) & (gt > 0)
        res[name] = float(np.mean(np.abs(p[k] - gt[k])))
        if keep: imgs[name] = p
    syn = [v for k, v in res.items() if k.startswith("synthetic")]
    res["synthetic mean"] = float(np.mean(syn))
    return (res, imgs) if keep else res

R = {}; SHOW = {}
for name, m in [("FP32 (original)", teacher), ("INT8 plain", ptq(teacher, 8)), ("INT4 plain", ptq(teacher, 4))]:
    R[name], im = score(m, keep=True); SHOW[name] = im
    print(f"{name:28s} synthetic {R[name]['synthetic mean']:.3f} ns  HEK293 {R[name]['HEK293']:.3f} ns ({time.time() - T0:.0f} s)", flush=True)

# ------------------------------------------------------------------ distillation practice images (Glasgow intensities)
def glasgow_intensity(zname, inner):
    plain = os.path.join(args.glasgow, "data", "05_Experiment_processed_data", inner)
    if os.path.exists(plain): c = np.load(plain, mmap_mode="r")
    else:
        with zipfile.ZipFile(os.path.join(args.glasgow, "data", "05_Experiment_processed_data", zname)) as z:
            c = np.load(pyio.BytesIO(z.read(inner)))
    out = np.zeros(c.shape[:2], "float32")
    for i in range(0, c.shape[0], 50):
        out[i:i + 50] = np.nan_to_num(np.asarray(c[i:i + 50], dtype="float32"), nan=0, posinf=0, neginf=0).clip(0).sum(-1)
    return out

cache = "distill_intensity_cache_smoke.npz" if args.smoke else "distill_intensity_cache.npz"
if os.path.exists(cache):
    SRC = list(np.load(cache).values())
else:
    SRC = [glasgow_intensity("HPC.zip", "HPC/speed_shift504_exp3300_location1.npy"),
           glasgow_intensity("LPC.zip", "LPC/speed_shift504_exp330_location1.npy")]
    if not args.smoke: SRC.append(glasgow_intensity("HT1080.zip", "HT1080/ht1080_9bit.npy"))
    np.savez(cache, *SRC)
print(f"practice images ready: {[s.shape for s in SRC]} ({time.time() - T0:.0f} s)", flush=True)
rng = np.random.default_rng(0)

def practice_sample(c=args.crop):
    src = SRC[rng.integers(len(SRC))]
    for _ in range(20):
        y, x = rng.integers(0, src.shape[0] - c), rng.integers(0, src.shape[1] - c)
        im = src[y:y + c, x:x + c].copy()
        if (im > np.percentile(src, 60)).mean() > 0.15: break
    if rng.random() < .5: im = im[::-1]
    if rng.random() < .5: im = im[:, ::-1]
    im = im / (np.percentile(im, 99) + 1e-6) * rng.uniform(8, 40)          # photons per pixel like his images
    inten = rng.poisson(im.clip(0)).astype("float32")
    mask = inten * (inten > 3)
    sm = gaussian_filter(im, 2); k = rng.integers(2, 6)
    levels = np.digitize(sm, np.quantile(sm, np.linspace(0, 1, k + 1)[1:-1]))
    tau = rng.uniform(0.5, 4.0, k)[levels] + gaussian_filter(rng.standard_normal(im.shape), 15) * 2
    tau = tau.clip(0.2, 5.0)
    m = mask > 0
    s_ = (tau * m).reshape(c // 8, 8, c // 8, 8).sum((1, 3)); n_ = m.reshape(c // 8, 8, c // 8, 8).sum((1, 3))
    lr = np.where(n_ >= 4, s_ / np.maximum(n_, 1) + rng.normal(0, 0.05, n_.shape), 0).clip(0).astype("float32")
    if lr.max() <= 0: return practice_sample(c)
    return prep_exp_data(mask.astype("float32"), lr).prep_data()

def batch_of(n):
    ds = [practice_sample() for _ in range(n)]
    return {k: (torch.cat([d[k] for d in ds]).to(DEV) if torch.is_tensor(ds[0][k]) else None) for k in ds[0] if k != "lr_tau_max"}

# ------------------------------------------------------------------ 4 bit QAT with distillation
def distil(bits_map, tag):
    student = copy.deepcopy(teacher).train()
    for n, mod in qlayers(student): P_.register_parametrization(mod, "weight", LSQ(mod.weight, bits_map(n)))
    sc = [p for n, p in student.named_parameters() if n.endswith("log_s")]
    ot = [p for n, p in student.named_parameters() if not n.endswith("log_s")]
    opt = torch.optim.Adam([{"params": ot, "lr": 2e-5}, {"params": sc, "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    run = 0
    for step in range(1, args.steps + 1):
        b = batch_of(args.batch)
        with torch.no_grad(): t = teacher(b)
        loss = F.l1_loss(student(b), t)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 500 == 0 or step == args.steps:
            print(f"[{tag}] step {step}/{args.steps} distillation loss {run:.5f} ({time.time() - T0:.0f} s)", flush=True)
    for n, mod in qlayers(student): P_.remove_parametrizations(mod, "weight", leave_parametrized=True)
    student.eval()
    torch.save({"model": student.state_dict()}, f"./model_parameters/NIII_parameter_L8_{tag}.pth.tar")
    return student

names = [n for n, _ in qlayers(teacher)]
first_last = {names[0], names[-1], "int_encoder.sfe1", "tau_encoder.sfe1"}
VARIANTS = [("INT4 + distillation + learned scales, first/last layers 8 bit", lambda n: 8 if n in first_last else 4, "INT4_all3")]
for label, bm, tag in VARIANTS:
    stu = distil(bm, tag)
    R[label], im = score(stu, keep=True); SHOW[label] = im
    print(f"{label:28s} synthetic {R[label]['synthetic mean']:.3f} ns  HEK293 {R[label]['HEK293']:.3f} ns ({time.time() - T0:.0f} s)", flush=True)
    if "V4 4 bit" in LR:
        key = f"whole pipeline 4 bit: LLE V4 + NIII {label}"
        R[key] = score(stu, "V4 4 bit")
        print(f"{key}: synthetic {R[key]['synthetic mean']:.3f} ns  HEK293 {R[key]['HEK293']:.3f} ns", flush=True)

R["memory MB (NIII weights)"] = {"FP32": NW * 4 / 1e6, "INT8": NW / 1e6, "INT4": NW / 2 / 1e6}
json.dump(R, open(f"{OUT}/niii_all3_results.json", "w"), indent=2)

# ------------------------------------------------------------------ figures
labels = [k for k in R if not k.startswith("whole") and not k.startswith("memory")]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for a, key, t in [(ax[0], "synthetic mean", "His 6 synthetic images, mean error"), (ax[1], "HEK293", "Real HEK293 cells, error")]:
    v = [R[l][key] for l in labels]
    bars = a.bar(range(len(v)), v, color=["#7f8c8d", "#2980b9", "#c0392b", "#27ae60", "#16a085"][:len(v)])
    a.set_xticks(range(len(v))); a.set_xticklabels([l.replace(", ", "\n").replace(" + ", "\n+ ") for l in labels], fontsize=7)
    a.set_ylabel("mean absolute error (ns)"); a.set_title(t)
    for b_, x in zip(bars, v): a.text(b_.get_x() + b_.get_width() / 2, x, f"{x:.3f}", ha="center", va="bottom", fontsize=8)
fig.suptitle("Quantising Dr Li's NIII (58 MB): weights only, his FP32 LLE in front")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_niii_all3_summary.png", dpi=130); plt.close(fig)
for test in ["synthetic 3", "HEK293"]:
    if test not in SHOW[labels[0]]: continue
    gt = [t[3] for t in TESTS if t[0] == test][0]
    fig, ax = plt.subplots(1, len(labels) + 1, figsize=(3.4 * (len(labels) + 1), 3.6))
    vmax = 4 if test.startswith("syn") else 2.5
    for a, (k, im) in zip(ax, [("truth / reference", gt)] + [(l, SHOW[l][test]) for l in labels]):
        h_ = a.imshow(im, cmap="inferno", vmin=0, vmax=vmax); a.set_title(k.replace(", ", "\n"), fontsize=8); a.axis("off")
    fig.colorbar(h_, ax=ax, shrink=.8, label="lifetime (ns)")
    fig.savefig(f"{OUT}/fig_niii_all3_{test.replace(' ', '')}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"ALL DONE ({time.time() - T0:.0f} s)", flush=True)
