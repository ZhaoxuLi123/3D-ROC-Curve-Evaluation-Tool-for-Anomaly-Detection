# 3D ROC Curve Evaluation Tool for Anomaly Detection:

This is the python version for  [“An Effective Evaluation Tool for Hyperspectral Target Detection: 3D Receiver Operating Characteristic Curve Analysis”](https://ieeexplore.ieee.org/abstract/document/9205919). 
The evaluation metrics developed based on the 3D ROC Curve can be used to evaluate general anomaly detection and hyperspectral target detection.

The 3D ROC Curve introduces segmentation thresholds on top of the 2D ROC Curve and derives several evaluation metrics.
However, the 3D ROC Curve overlooks the fact that different detectors may have different response ranges, which can sometimes lead to distortion in these metrics.
To address this issue, we have improved these metrics in our work [“AETNet”](https://ieeexplore.ieee.org/document/10073635).
In particular, we believe that ASNPR (adaptive signal-to-noise probability ratio) can accurately quantify
the background suppressibility or the separability between targets and background.



## Citation

If this code is helpful, please cite the papers:

```
@ARTICLE{9205919,
  author={Chang, Chein-I},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={An Effective Evaluation Tool for Hyperspectral Target Detection: 3D Receiver Operating Characteristic Curve Analysis}, 
  year={2021},
  volume={59},
  number={6},
  pages={5131-5153},
  doi={10.1109/TGRS.2020.3021671}}
```

```
@ARTICLE{10073635,
  author={Li, Zhaoxu and Wang, Yingqian and Xiao, Chao and Ling, Qiang and Lin, Zaiping and An, Wei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={You Only Train Once: Learning a General Anomaly Enhancement Network with Random Masks for Hyperspectral Anomaly Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2023.3258067}}
```
