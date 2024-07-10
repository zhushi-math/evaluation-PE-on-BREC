# PPGN Reproduction

## Requirements

Please refer to [PPGN](https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch)

## Usages

unzip dataset

```bash
unzip brec_v3.zip
```

To reproduce results on PPGN, run:

```bash
python test_BREC_search.py
```

The configs are stored in configs/BREC.json



## create environment


```bash
conda create --name graphgm python pytorch=2.1.0 pytorch-cuda=12.1 pyg openbabel fsspec rdkit pip -c pyg -c pytorch  -c nvidia -c conda-forge -c rdkit
# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  

conda activate graphgm
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install pytorch-lightning
pip install yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb
```
