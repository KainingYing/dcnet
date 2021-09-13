 ## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.7
- PyTorch 1.6.0 & TorchVision 0.7.0
- CUDA 10.1
- mmcv-full 1.3.12

### Install MMHOIDet

a. Create a conda (recommended) virtual environment and activate it.

```shell
conda create -n mmhoidet python=3.7 -y
conda activate mmhoidet
```

b. Install PyTorch and TorchVision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y
```

c. Clone the MMHOIDet repository.

```shell
git clone https://github.com/noobying/mmhoidet
cd mmhoidet
```

d. Install full dependencies and package.

```shell
pip install -e .[full]
```

### Prepare datasets

It is recommended to symlink the dataset root to `mmhoidet/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
mmhoidet
├── mmhoidet
├── tools
├── configs
├── data
│   ├── hico_20160224_det
│   │   ├── annotations
│   │   |   ├── test_hico.json
│   │   |   ├── trainval_hico.json
│   │   ├── images
│   │   |   ├── test2015
│   │   |   ├── train2015
```
`note`: HICO-Det can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). The `test_hico.json` and `trainval_hico.json` in `annotations` is provided by [PPDM](https://github.com/YueLiao/PPDM). You can download the annotations from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R) and replace the original annotations directory.

