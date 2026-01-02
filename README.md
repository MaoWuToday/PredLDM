# PredLDM

> Official implementation of PredLDM using Latent Diffusion Models.

---



## 📋 Table of Contents

- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Quick Start](#-quick-start)
  - [Testing with Pretrained Models](#testing-with-pretrained-models)
  - [Training from Scratch](#training-from-scratch)
- [Additional Resources](#-additional-resources)
- [Citation](#-citation)

---

## 📝 Updates

- **`2026/01/02`** Release code, scripts, pre-trained weights and data for reproducing training and testing.
- **`xx`** Please wait for our future support.

---

## 🚀 Installation

### Install via Conda (CUDA 11.3)

```bash
conda env create -f environment2.yml
conda activate BPN
```

---

## 📦 Data Preparation

### KTH Dataset

Download the KTH dataset:
- **Train dataset**: [train-kth](https://pan.baidu.com/s/1v2GpvtcD89TxvH2a2iAMOw?pwd=1nuv)
- **Test dataset**: [test-kth](https://pan.baidu.com/s/1_RQGlFqq9xqEqJobRIdQXQ?pwd=2adj)

The downloaded `.zip` files should be unzipped in the `data` folder. The expected directory structure is:

```
data/
└── train/
    ├── kth/
    │   ├── sample1/
    │   ├── sample2/
    │   └── ...
    ├── kitticaltech/
    └── sevir-vil/
└── test/
    ├── kth/
    ├── kitticaltech/
    └── ...
```

---

## 🎯 Quick Start

### Testing with Pretrained Models

1. **Download pretrained weights**
   
   Download the pre-trained [kth-weights](https://pan.baidu.com/s/1JZFiv7wpkHLe4sOaqyhfbg?pwd=scmb) and place them in the `save_models` folder.

2. **Configure settings**
   
   Edit `configs.py` with the following settings:

   ```python
   """setting running mode"""
   mode = 'test'
   model = 'PredLDM'
   dataset_type = 'pam'
   pretrained_model = '{the path to pretrained parameters}'
   domain = 'kth'  # choose one: domains = ['kitticaltech', 'kth', 'sevir-vil']
   ```

3. **Specify GPU device** (optional)
   
   In `configs.py`, specify the GPU device number:
   
   ```python
   device_ids_eval = [0]
   ```

4. **Run inference**
   
   ```bash
   python main_scratch_v2.py
   ```

---

### Training from Scratch

1. **Configure settings**
   
   Edit `configs.py` with the following settings:

   ```python
   """setting running mode"""
   mode = 'train'
   model = 'PredLDM'
   dataset_type = 'pam'
   ```

2. **Start training**
   
   ```bash
   python main_scratch_v2.py
   ```

---

## 📚 Additional Resources

### Additional Datasets

- **KITTI-Caltech**:
  - [train-kitticaltech](https://pan.baidu.com/s/1n6_7xNKE2rcq0udyZ9WEgg?pwd=cjgx)
  - [test-kitticaltech](https://pan.baidu.com/s/1iO_E8mKKfFh5ve_X5MVdvg?pwd=ni6s)

- **SEVIR-VIL**:
  - [train-sevir-vil](https://pan.baidu.com/s/1LcaYXZWhWuF_GNQHszQBqQ?pwd=2agw)
  - [test-sevir-vil](https://pan.baidu.com/s/1RPB7nl8Ge3zgSV1hqGPMlQ?pwd=kndb)

### Additional Pretrained Weights

- [kitticaltech-weights](https://pan.baidu.com/s/1tVk2CFVteoZ-92FzegY-qw?pwd=fasd)
- [sevir-weights](https://pan.baidu.com/s/1MHyx_4-rDq0eDRUKYiAWrA?pwd=s896)

---



## 📖 Citation

```bibtex
@article{predldm,
  title={PredLDM},
  author={...},
  journal={...},
  year={...}
}
```

*Citation information coming soon.*

---
