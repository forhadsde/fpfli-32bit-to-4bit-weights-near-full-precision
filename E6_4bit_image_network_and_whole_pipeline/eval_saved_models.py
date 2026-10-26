"""
eval_saved_models.py  (Forhad)

Re-scores ALL the models my RTX 3090 already trained, with Dr Li's full FPFLI pipeline (LLE + NIII)
on his sample data. No training, nothing random, so it reproduces the exact numbers in my write up.

  python eval_saved_models.py          (about 2 to 4 minutes on the GPU)
Output: printed table and my_saved_model_check/saved_model_check.json
"""
import os, json, time, warnings, builtins
import numpy as np, torch, torch.nn as nn
warnings.filterwarnings("ignore")
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False, "map_location": k.get("map_location", "cpu")})
import scipy.io as sio, mat73
import Pred_local_lifetime as PLL
from Prep_exp_data import prep_exp_data
from NIII_model import NIII

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "", flush=True)
MP = "./model_parameters/"
OUT = "my_saved_model_check"; os.makedirs(OUT, exist_ok=True)
_print = builtins.print
def quiet(f, *a, **k):
    builtins.print = lambda *x, **y: None
    try: return f(*a, **k)
    finally: builtins.print = _print

def ptq_(model, bits):
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                f = m.weight.reshape(m.weight.shape[0], -1)
                s = f.abs().max(1, keepdim=True).values.clamp(min=1e-12) / qmax
                m.weight.copy_((torch.round(f / s).clamp(-qmax, qmax) * s).reshape(m.weight.shape))

def load_niii(path, bits=None):
    m = NIII(); m.load_state_dict(torch.load(path)["model"]); m.to(DEV).eval()
    if bits: ptq_(m, bits)
    return m

irf_g = sio.loadmat("sample_data/irf_gaussian.mat")["irf_gaussian"].astype("float32")
TESTS = []
for s in ["1", "2", "3", "4", "5", "6"]:
    d = sio.loadmat(f"sample_data/{s}.mat")
    TESTS.append((f"synthetic {s}", d["Hist"].astype("float32"), irf_g, d["tau_gt"].astype("float32"), "general"))
d = sio.loadmat("sample_data/hek_low.mat")
TESTS.append(("HEK293", d["hist"].astype("float32"), d["irf"].astype("float32"), mat73.loadmat("sample_data/hek_gt.mat")["tau_img_gt"].astype("float32"), "1-3ns"))

LLES = {"LLE FP32 (Dr Li)": ("LLE_parameter.pth", "LLE_parameter_1_3ns.pth", None),
        "LLE INT8 plain": ("LLE_parameter.pth", "LLE_parameter_1_3ns.pth", 8),
        "LLE INT4 plain": ("LLE_parameter.pth", "LLE_parameter_1_3ns.pth", 4),
        "LLE INT4 QAT v1": ("LLE_parameter_QAT_INT4.pth", "LLE_parameter_1_3ns_QAT_INT4.pth", None),
        "LLE INT4 V4 (best)": ("LLE_parameter_V4.pth", "LLE_parameter_1_3ns_V4.pth", None)}

def lle_maps(general, hek, bits):
    out = []
    for name, hist, irf, gt, kind in TESTS:
        inten = hist.sum(2); keep = inten > 3
        hm = hist * keep[..., None]; mask = inten * keep
        o = quiet(lambda: PLL.Pred_local_lifetime(hm, irf, scaling_ratio=8, h=0.039, file_path=MP + (general if kind == "general" else hek)))
        if bits: ptq_(PLL.model, bits)
        out.append((o.eval_local_tau(), mask))
    return out

def score(niii, maps):
    res = {}
    for (name, _, _, gt, _), (lr, mask) in zip(TESTS, maps):
        dd = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in prep_exp_data(mask.astype("float32"), lr.astype("float32")).prep_data().items()}
        with torch.no_grad(): p = niii(dd) * dd["lr_tau_max"]
        p = p.reshape(mask.shape).cpu().numpy(); p[p < 0] = 0
        k = (mask > 0) & (gt > 0); res[name] = float(np.mean(np.abs(p[k] - gt[k])))
    res["synthetic mean"] = float(np.mean([v for k, v in res.items() if k.startswith("synthetic")]))
    return res

T0 = time.time(); R = {}
MAPS = {k: lle_maps(*v) for k, v in LLES.items() if all(os.path.exists(MP + f) for f in v[:2])}
niii_fp32 = load_niii(MP + "NIII_parameter_L8.pth.tar")
print("\nA) LLE versions, with Dr Li's FP32 NIII", flush=True)
for k in MAPS:
    R[k] = score(niii_fp32, MAPS[k]); print(f"  {k:32s} synthetic {R[k]['synthetic mean']:.3f} ns   HEK293 {R[k]['HEK293']:.3f} ns", flush=True)
print("\nB) NIII versions, with Dr Li's FP32 LLE", flush=True)
NIIIS = {"NIII INT8 plain": ("NIII_parameter_L8.pth.tar", 8), "NIII INT4 plain": ("NIII_parameter_L8.pth.tar", 4),
         "NIII INT4 + distillation": ("NIII_parameter_L8_INT4_distil.pth.tar", None),
         "NIII INT4 + distillation, first/last 8 bit": ("NIII_parameter_L8_INT4mixed_distil.pth.tar", None),
         "NIII INT4 all three techniques": ("NIII_parameter_L8_INT4_all3.pth.tar", None)}
models = {}
for k, (f, b) in NIIIS.items():
    if not os.path.exists(MP + f): continue
    models[k] = load_niii(MP + f, b)
    R[k] = score(models[k], MAPS["LLE FP32 (Dr Li)"]); print(f"  {k:44s} synthetic {R[k]['synthetic mean']:.3f} ns   HEK293 {R[k]['HEK293']:.3f} ns", flush=True)
print("\nC) Whole pipeline with 4 bit weights (LLE V4 + 4 bit NIII)", flush=True)
for k in ["NIII INT4 + distillation, first/last 8 bit", "NIII INT4 all three techniques"]:
    if k in models and "LLE INT4 V4 (best)" in MAPS:
        key = "WHOLE 4 bit: LLE V4 + " + k
        R[key] = score(models[k], MAPS["LLE INT4 V4 (best)"]); print(f"  {key:70s} synthetic {R[key]['synthetic mean']:.3f} ns   HEK293 {R[key]['HEK293']:.3f} ns", flush=True)
json.dump(R, open(f"{OUT}/saved_model_check.json", "w"), indent=2)
print(f"\nDone in {time.time() - T0:.0f} s. Saved {OUT}/saved_model_check.json", flush=True)
