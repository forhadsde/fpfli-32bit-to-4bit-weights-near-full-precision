# Source: Glasgow megapixel SPAD FLIM dataset

**Authors:** Vytautas Zickus, Ming-Lo Wu, Kazuhiro Morimoto, Valentin Kapitany, Areeba Fatima, Alex Turpin, Robert Insall, Jamie Whitelaw, Laura Machesky, Claudio Bruschini, Daniele Faccio and Edoardo Charbon

**Paper:** "Fluorescence lifetime imaging with a megapixel SPAD camera and neural network lifetime estimation", *Scientific Reports* 10 (2020). DOI: https://doi.org/10.1038/s41598-020-77737-0

**Dataset:** University of Glasgow research data repository, DOI https://doi.org/10.5525/gla.researchdata.1012
* Record page: https://researchdata.gla.ac.uk/1012
* Direct download (about 1 GB): https://researchdata.gla.ac.uk/1012/2/DATA_REPOSITORY_fluorescence_lifetime_imaging.zip

**Licence:** Creative Commons Attribution 4.0 (CC BY 4.0)

## How to fill this folder

Download the zip above and extract it here, so the path looks like:

```
SPAD_FLIM_Glasgow_Zickus2020/
    DATA_REPOSITORY_fluorescence_lifetime_imaging_with_a_megapixel_SPAD_camera_and_neural_network_lifetime_estimation/
        data/Experiment processed data/...
```

## How it is used here

Only E6 uses it, and only when retraining (`RETRAIN = True`). Three processed photon cubes (Convallaria at two exposures, `HPC/speed_shift504_exp3300_location1.npy` and `LPC/speed_shift504_exp330_location1.npy`, and HT1080 cells, `HT1080/ht1080_9bit.npy`) are summed over time to give intensity images. Random crops of these become practice images for distilling the 4 bit NIII from the 32 bit teacher. No lifetimes or labels from this dataset are used.
