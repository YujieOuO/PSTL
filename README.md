# PSTL
This is an official PyTorch implementation of **"Self-supervised Action Representation Learning 
from Partial Spatio-Temporal Skeleton Sequences" in AAAI2023 (Oral)**.

[[Paper]](https://arxiv.org/abs/2302.09018)

## SkeletonBT
![SkeletonBT](https://user-images.githubusercontent.com/47097735/221340750-09aed928-9100-4b49-b2f9-7cf78bbb79e5.png)

## Framework
![PSTL](https://user-images.githubusercontent.com/47097735/221340707-2a90c224-1183-4166-9de9-ac0553543f69.png)

## Requirements
![python = 3.7](https://img.shields.io/badge/python-3.7.13-green)
![torch = 1.11.0+cu113](https://img.shields.io/badge/torch-1.11.0%2Bcu113-yellowgreen)

## Installation
```bash
# Install the python libraries
$ cd PSTL
$ pip install -r requirements.txt
```

## Data Preparation
We apply the same dataset processing as [AimCLR](https://github.com/Levigty/AimCLR).  
You can also download the file folder in BaiduYun link:
* [NTU-RGB-D 60](https://pan.baidu.com/s/1ukBF5aI8QawRriJbmsrv5Q).
* [NTU-RGB-D 120](https://pan.baidu.com/s/1AG_516WHitv1LBh1NNrvVg).
* [PKU-MMD](https://pan.baidu.com/s/168uXCgrKdh7esqatGwfEfg).

The code: pstl
## Training & Testing
Example for unsupervised pre-training on NTU-60 xsub datasets.  
You can change some settings of [config.py](https://github.com/YujieOuO/PSTL/blob/main/config.py).  

```bash
# pre-training
$ python procedure.py with 'train_mode="pretrain"'

# linear evaluation
$ python procedure.py with 'train_mode="lp"'
```

## Reference
If you find our paper and repo useful, please cite our paper. Thanks!
```
@inproceedings{zhou2023self,
  title={Self-supervised action representation learning from partial spatio-temporal skeleton sequences},
  author={Zhou, Yujie and Duan, Haodong and Rao, Anyi and Su, Bing and Wang, Jiaqi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={3},
  pages={3825--3833},
  year={2023}
}
```
## Acknowledgement
* The framework of our code is based on [MS2L](https://github.com/LanglandsLin/MS2L).
* The encoder is based on [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md).

## Licence
This project is licensed under the terms of the MIT license.

## Contact
For any questions, feel free to contact: yujieouo@sjtu.edu.cn
