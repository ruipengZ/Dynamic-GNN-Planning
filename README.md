# Dynamic GNN Planning

This is the code for NeurIPS'22 paper Learning-based Motion Planning in Dynamic Environments Using GNNs and Temporal Encoding 

[paper](https://arxiv.org/abs/2210.08408) | [website](https://ruipengz.github.io/neurips22/)

![pipeline](./imgs/GNN-TE.png)


## Installation
```bash
conda create -n Dynamic-GNN python=3.8
conda activate Dynamic-GNN
# install pytorch, modify the following line according to your environment
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
# install torch geometric, refer to https://github.com/pyg-team/pytorch_geometric
conda install pyg -c pyg
pip install pybullet transforms3d matplotlib
```

## Environment
![envs](./imgs/envs.png)

## Testcases
### Available Testcases

[**2arms**](https://drive.google.com/drive/folders/16k7xd_O50l5oFslOXDjoAAWLYGxItFRq?usp=sharing) Please put /testcase into the root directory of this repo.

### Generate testcases
```bash
# generate test cases for 2arms using multi-processing
python oracle/generate_testcases.py
```


