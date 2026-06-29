## Implementation of FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models
This repository contains the source code and instructions to reproduce the results in our paper.


## Setup
Create an virtual python environment and use ``requirements.txt`` to set up all the required packages.
```bash
conda create -n FineSteer python=3.11.9
conda activate FineSteer
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requriements.txt
```
Please make sure that your device supports CUDA with 12.1 or higher version.


## Train and evaluate FineSteer

We will include the training and evaluation details soon.


## How to cite
```bibtex
@inproceedings{weng2026finesteer,
  title={FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models},
  author={Weng, Zixuan and Zhang, Jinghuai and Cai, Kunlin and Li, Ying and Wang, Peiran and Tian, Yuan},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={18736--18756},
  year={2026}
}
```
