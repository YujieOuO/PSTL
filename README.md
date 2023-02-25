# PSTL
This is an official PyTorch implementation of **"Self-supervised Action Representation Learning 
from Partial Spatio-Temporal Skeleton Sequences" in AAAI2023**.
## SkeletonBT
![SkeletonBT](https://user-images.githubusercontent.com/47097735/221340750-09aed928-9100-4b49-b2f9-7cf78bbb79e5.png)
## PSTL
![PSTL](https://user-images.githubusercontent.com/47097735/221340707-2a90c224-1183-4166-9de9-ac0553543f69.png)
## Requirements
python = 3.7

torch = 1.11.0+cu113
## Data Preparation
We apply the same dataset processing, backbone, and dimension of the linear probe as AimCLR/CrosSCLR.
Link: [AimCLR]https://github.com/Levigty/AimCLR
You can also download the file folder in BaiduYun link:

## Acknowledgement
* The framework of our code is based on ![MS2L]https://github.com/LanglandsLin/MS2L.
* The encoder is based on ![ST-GCN]https://github.com/yysijie/st-gcn/blob/master/OLD_README.md.
