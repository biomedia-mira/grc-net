# GRC-Net: Cardiovascular disease classification using radiomics and geometric features from cardiac CT Images

When using this code, please cite the following paper:
> (A. Mittal, R. Mehta, O. Todd, P. Seeboeck, G. Langs, B. Glocker) Cardiovascular disease classification using radiomics and geometric features from cardiac CT Images

## Code

For running the code, we recommend setting up a dedicated Python environment.

### Setup Python environment using conda

Create and activate a Python 3.9 conda environment:

   ```shell
   conda create -n grcnet python=3.9
   conda activate grcnet
   ```
Install PyTorch using conda (for CUDA toolkit 12.4)
   ```shell
    # torch dependencies
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

    # other Python packages
    pip install numpy==1.23 torchio tensorboard tensorboardX

    # TotalSegmentator and utils
    pip install TotalSegmentator
    pip install --upgrade acvl_utils==0.2

    # GPU transforms (MONAI with CuCIM)
    pip install monai[cucim] cupy-cuda12x==12.3.0

    # Patches
    pip install mkl==2024.0 dicom2nifti==1.2.21 pydicom==1.4.1
  ```
   
To run both the Anatomix fine-tuning and Geo-Radio Classification tutorial:
```shell
  pip install ipykernel ipywidgets pyradiomics optuna
```

## Pipeline Overview
This model assumes the existence of the ASOCA dataset and the MM-WHS dataset in the `data` subdirectory. 
- The CT volumes and labelmaps for the MM-WHS dataset need to be extracted to the `data/MM-WHS/images` and `data/MM-WHS/labels` directory accordingly, with data splits pre-defined in the ``train``, ``val`` and ``test`` files in `data/config`. These images are used to fine-tune the Anatomix model and train the Atlas and registration network.  
- The CT volumes for the ASOCA dataset need to be extracted to the `data/ASOCA/images` directory. The segmentations will be generated and hence do not need to be provided.

This repository provides training scripts for three key stages of the pipeline:

1. **Segmentation model training**
2. **Atlas construction and registration network training**
3. **Disease classification model training and evaluation**

### 1. Segmentation Model Training

To fine-tune the Anatomix segmentation model, use:

- `anatomix-fine-tuning.ipynb`

### 2. Atlas Construction and Registration

There are three available options for training the atlas and registration network:

- **Fully supervised** (requires manual segmentation labels):  
  Use `atlas-istn-fully-supervised.py`

- **Semi-supervised (Anatomix-based)** (requires a fine-tuned Anatomix model):  
  Use `atlas-istn-anatomix.py`  
  _Note: Ensure the Anatomix model has been fine-tuned using `anatomix-fine-tuning.ipynb`_

- **Semi-supervised (TotalSegmentator-based)** (uses TotalSegmentator as a pseudo-label generator):  
  Use `atlas-istn-totalsegmentator.py`
  _Note: requires a commercial license, (see https://github.com/wasserth/TotalSegmentator/)_

_Note: The Semi-supervised scripts provided here evaluate against the pseudo-labels during training instead of against the manual annotations as done in the paper. More generally, the semi-supervised scripts do not load any manual annotations. This is done for easier flexibility to add more unlabelled data for any further data you wish to train the model with in the future._

No validation or test images are _needed_ to train the registration network and atlas.

### 3. Disease Classification

To train and evaluate the disease classification model, use:

- `Geo-Radio-Classification.ipynb`

This step requires:
- A trained Anatomix segmentation model (from step 1)
- A trained registration model and constructed atlas (from step 2)  
The required atlas labelmaps will be automatically generated during the registration stage.


## License
This project is licensed under the [Apache License 2.0](LICENSE).
