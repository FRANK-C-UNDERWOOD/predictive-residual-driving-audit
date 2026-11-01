
---

<h3 align="center"> SpikeFPN: Automotive Object Detection via Learning Sparse Events by Spiking Neurons </h3>

---

<p align="center">
  <picture>
    <img src="docs/source/_static/spikefpn-overview.jpg" alt="SpikeFPN Overview" height="420">
  </picture>
</p>

This work explores the membrane potential dynamics of spiking neural networks (SNNs) and their ability to process sparse, asynchronous events. We propose an innovative **spike-triggered adaptive threshold** mechanism that facilitates stable and effective training. Building on this foundation, we design a specialized **spiking feature pyramid network (SpikeFPN)** optimized for automotive event-based object detection. Comprehensive evaluations indicate that SpikeFPN achieves competitive performance compared to traditional SNNs and advanced artificial neural network (ANN) models while maintaining efficient computation.

- **Official publication** in *IEEE Transactions on Cognitive and Developmental Systems*:  
  [DOI: 10.1109/TCDS.2024.3410371](https://doi.org/10.1109/TCDS.2024.3410371)

- **Accepted manuscript** available on *arXiv*:  
  [arXiv:2307.12900](https://arxiv.org/abs/2307.12900)
  

## Environment Configuration

In a configuration utilizing `Ubuntu 22.04`, `CUDA 12.4`, and `PyTorch 2.3.1`:

```shell
apt-get update # If necessary
apt-get install ffmpeg libsm6 libxext6
pip install -r requirements.txt
```

## Experiment on GEN1 Automotive Detection (GAD) Dataset

### Data Preprocessing
```shell
python ./preprocess/gad_framing.py
```

### Training and Testing
```shell
python ./train_gad.py
python ./test_gad.py
```

## Experiment on N-CARS Dataset

### Data Preprocessing
```shell
python ./preprocess/ncars_framing.py
```

### Data Division
|                    | Class: background | Class: cars  |
| :----------------- | :---------------: | :----------: |
| **For Training**   |     0 ~ 4210      |   0 ~ 4395   |
| **For Validating** |    4211 ~ 5706    | 4396 ~ 5983  |
| **For Testing**    |   5707 ~ 11692    | 5984 ~ 12335 |

### Training and Testing
```shell
python ./train_ncars.py
python ./test_ncars.py
```

## Citation

Please cite the following publication if this work was helpful to your research.
```latex
@article{spikefpn,
  author  = {Hu Zhang and Yanchen Li and Luziwei Leng and Kaiwei Che and Qian Liu and Qinghai Guo and Jianxing Liao and Ran Cheng},
  title   = {Automotive Object Detection via Learning Sparse Events by Spiking Neurons},
  journal = {{IEEE} Trans. Cogn. Dev. Syst.},
  volume  = {16},
  number  = {6},
  pages   = {2110--2124},
  year    = {2024},
  doi     = {10.1109/TCDS.2024.3410371},
}
```

