# Shared code and data (not included)

The notebooks look for this folder by name, so keep it here and place the following inside it.

## 1. `Dr_Li_FPFLI_code/`

The FPFLI release by Xiao, Sapermsap, Chen and Li (Optica 2023). It should contain:

* `tau_net.py`, `Pred_local_lifetime.py`, `Prep_exp_data.py`, `NIII_model.py`, `Neural_implict_interploation.py`, `utils.py`
* `model_parameters/LLE_parameter.pth`, `model_parameters/LLE_parameter_1_3ns.pth`, `model_parameters/NIII_parameter_L8.pth.tar`
* `sample_data/` with `1.mat` to `6.mat`, `irf_gaussian.mat`, `hek_low.mat`, `hek_gt.mat`, `microsphere_2p_1.mat`, `microsphere_2p_2.mat` and `irf_measure.mat`

The pretrained model links are listed in the release's `model_parameters/readme.md`. These files belong to the original authors and are not redistributed here.

My retrained 4 bit models are saved into the same `model_parameters/` folder when E4 and E6 are run with `RETRAIN = True`:

* E4: `LLE_parameter_QAT_INT4.pth`, `LLE_parameter_V2.pth` to `LLE_parameter_V4.pth`, and the matching `LLE_parameter_1_3ns_*.pth` files
* E6: `NIII_parameter_L8_INT4_distil.pth.tar`, `NIII_parameter_L8_INT4mixed_distil.pth.tar`, `NIII_parameter_L8_INT4_all3.pth.tar`

## 2. `Datasets/SPAD_FLIM_Glasgow_Zickus2020/` (E6 training only)

The public data from Zickus et al. 2020, "Fluorescence lifetime imaging with a megapixel SPAD camera and neural network lifetime estimation". E6 uses three intensity images from it (two Convallaria exposures and HT1080 cells) to make practice images for distillation. It is not needed when `RETRAIN = False`.
