# Compact and Accurate Features for Efficient Human-Object Interaction Detection 

## Install
```shell
conda create -n dcnet python=3.8 -y
conda activate dcnet

conda install pytorch==1.10.0 torchvision cudatoolkit=11.3 -c pytorch -y

pip install openmim
mim install mmcv-full mmdet
pip install -e .[full]
pip install setuptools==58.2.0
```