# SAMUS
This repo is the official implementation for:\
[SAMUS: Adapting Segment Anything Model for Clinically-Friendly and Generalizable Ultrasound Image Segmentation.](https://arxiv.org/pdf/2309.06824.pdf)\
(The details of our SAMUS can be found in the models directory in this repo or in the paper.)

## Highlights
🏆 Low GPU requirements. (one 3090ti with 24G GPU memory is enough)\
🏆 Large ultrasound dataset. (about 30K images and 69K masks covering 6 categories)\
🏆 Excellent performance, especially in generalization ability.\ 

## Installation
Following [Segment Anything](https://github.com/facebookresearch/segment-anything), `python=3.8.16`, `pytorch=1.8.0`, and `torchvision=0.9.0` are used in SAMUS.

1. Clone the repository.
    ```
    git clone https://github.com/xianlin7/SAMUS.git
    cd SAMUS
    ```
2. Create a virtual environment for SAMUS and activate the environment.
    ```
    conda create -n SAMUS python=3.8
    conda activate SAMUS
    ```
3. Install Pytorch and TorchVision.
   (you can follow the instructions [👉](https://pytorch.org/get-started/locally/))
5. Install other dependencies.
  ```
    pip install -r requirements.txt
  ```

## Dataset
US30K consists of seven publicly-available datasets, including [TN3K]( https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation), [DDTI]( https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation), [TG3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation), [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), [UDIAT](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php), [CAMUS](http://camus.creatis.insa-lyon.fr/challenge/), and [HMC-QU](https://aistudio.baidu.com/aistudio/datasetdetail/102406).
