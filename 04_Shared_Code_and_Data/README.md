# Shared code and data (download from the original sources)

The notebooks look for this folder by name, so keep it here. The files themselves are not copied into this repository. Each subfolder has a `SOURCE.md` with the exact link and setup steps.

| Folder | What goes in it | Where to get it | Needed for |
|---|---|---|---|
| [`Dr_Li_FPFLI_code/`](Dr_Li_FPFLI_code/SOURCE.md) | FPFLI code, pretrained LLE and NIII models, sample data (synthetic images, HEK293, microspheres) | https://github.com/Dooongx/FPFLI (copy the contents of its `Evaluation` folder) | All experiments |
| [`Datasets/SPAD_FLIM_Glasgow_Zickus2020/`](Datasets/SPAD_FLIM_Glasgow_Zickus2020/SOURCE.md) | Glasgow megapixel SPAD FLIM data (CC BY 4.0) | https://doi.org/10.5525/gla.researchdata.1012 | E6 retraining only |

## Papers

* D. Xiao, N. Sapermsap, Y. Chen and D. D. U. Li, "Deep learning enhanced fast fluorescence lifetime imaging with a few photons", *Optica* 10(7), 944 to 951 (2023). https://strathprints.strath.ac.uk/85874
* V. Zickus et al., "Fluorescence lifetime imaging with a megapixel SPAD camera and neural network lifetime estimation", *Scientific Reports* 10 (2020). https://doi.org/10.1038/s41598-020-77737-0

## My retrained models

When E4 and E6 run with `RETRAIN = True`, my 4 bit models are saved into `Dr_Li_FPFLI_code/model_parameters/`:

* E4: `LLE_parameter_QAT_INT4.pth`, `LLE_parameter_V2.pth` to `LLE_parameter_V4.pth`, and the matching `LLE_parameter_1_3ns_*.pth` files
* E6: `NIII_parameter_L8_INT4_distil.pth.tar`, `NIII_parameter_L8_INT4mixed_distil.pth.tar`, `NIII_parameter_L8_INT4_all3.pth.tar`
