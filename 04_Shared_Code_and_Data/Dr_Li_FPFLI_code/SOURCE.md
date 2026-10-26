# Source: FPFLI code, pretrained models and sample data

**Owners:** Dong Xiao, Natakorn Sapermsap, Yu Chen and David Day Uei Li (University of Strathclyde)

**Paper:** D. Xiao, N. Sapermsap, Y. Chen and D. D. U. Li, "Deep learning enhanced fast fluorescence lifetime imaging with a few photons", *Optica* 10(7), 944 to 951 (2023)
* Open access version: https://strathprints.strath.ac.uk/85874
* Preprint: https://www.biorxiv.org/content/10.1101/2023.04.06.534322v1

**Code and sample data:** https://github.com/Dooongx/FPFLI

These files are not copied into this repository because the original release has no licence, so all rights stay with the authors.

## How to fill this folder

1. Download or clone https://github.com/Dooongx/FPFLI
2. Copy everything inside its `Evaluation` folder into this folder (`04_Shared_Code_and_Data/Dr_Li_FPFLI_code/`), so it looks like this:

```
Dr_Li_FPFLI_code/
    tau_net.py
    Pred_local_lifetime.py
    Prep_exp_data.py
    NIII_model.py
    Neural_implict_interploation.py
    utils.py
    model_parameters/
        LLE_parameter.pth
        LLE_parameter_1_3ns.pth
        NIII_parameter_L8.pth.tar
    sample_data/
        1.mat ... 6.mat, irf_gaussian.mat
        hek_low.mat, hek_gt.mat
        microsphere_2p_1.mat, microsphere_2p_2.mat, irf_measure.mat
```

3. If the three pretrained model files are missing, download them from the links in the release's `model_parameters/readme.md` (hosted on Google Drive by the authors).

## What each file is used for here

| File | Used in |
|---|---|
| `LLE_parameter.pth` (general LLE) | E2 to E6, synthetic images and microspheres |
| `LLE_parameter_1_3ns.pth` (1 to 3 ns LLE) | E2 to E6, HEK293 |
| `NIII_parameter_L8.pth.tar` | E2 to E6 (the teacher in E6) |
| `sample_data/1.mat` to `6.mat`, `irf_gaussian.mat` | Synthetic tests in every experiment |
| `sample_data/hek_low.mat`, `hek_gt.mat` | The real HEK293 test and its high photon reference |
| `sample_data/microsphere_2p_*.mat`, `irf_measure.mat` | E2 microsphere check |

My retrained 4 bit models are written into `model_parameters/` when E4 and E6 run with `RETRAIN = True`.
