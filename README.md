# SIR on Networks

> 基于社交网络的SIR模型数据同化

已知社交网络结构，观测节点感染个数，通过EnKF算法估计SIR模型的参数beta and/or gamma.

[//]: # (![]&#40;https://github.com/dbader/readme-template/raw/master/header.png&#41;)

## Installation

基于Python3
主要注意安装`ndlib`和`filterpy`包，其余包按需安装。

## 代码说明
- `codes/scen1`: 场景1的代码
- `codes/scen2`: 场景2的代码
- `codes/scen3`: 场景3的代码
- `codes/scen*/EnKF_delta_*.py`: EnKF算法部分，主要区别在于不同场景下修正状态方法不同
- `codes/scen1/origsir.py`: 原始sir算法部分
- `codes/scen1/particle_filter.py`: 粒子滤波部分

## 数据集说明
- `datasets/graph/`: 社交网络，gnutella p2p网络

  
