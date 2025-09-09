# Joint Modeling of Pixel Wise Visibility and Fog Structure for Real World Scene Understanding

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.8%2B-green)]() [![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)]()


This repository contains the official implementation code for the paper:

**Joint Modeling of Pixel Wise Visibility and Fog Structure for Real World Scene Understanding**  
by JIAYU WU, JIAHENG LI, JIANQIANG WANG, XUEZHE XU, SIDAN DU, and YANG LI.

> **Abstract:** *Reduced visibility caused by foggy weather has a significant impact on transportation systems and driving safety, leading to increased accident risks and decreased operational efficiency. Accurate visibility estimation enables better traffic management and timely driver warnings. Traditional methods rely on expensive physical instruments, limiting their scalability. To address this challenge in a cost-effective manner, we propose a two-stage network for visibility estimation from stereo image inputs. The first stage computes scene depth via stereo matching, while the second stage fuses depth and texture information to estimate metric-scale visibility. Our method produces pixel-wise visibility maps through a physically constrained, progressive supervision strategy, providing rich spatial visibility distributions beyond a single global value. Moreover, it enables the detection of patchy fog, allowing a more comprehensive understanding of complex atmospheric conditions. To facilitate training and evaluation, we propose an automatic fog-aware data generation pipeline that incorporates both synthetically rendered foggy images and real-world captures, synchronized with reference visibility measurements obtained from a forward scatter visibility meter. Expensive experiments demonstrate that our method achieves state-of-the-art performance in both visibility estimation and patchy fog detection.*
---

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- PyTorch >= 1.8.0
- CUDA

We recommend using a virtual environment (e.g., `venv` or `conda`).

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/HexaWarriorW/Pixel-Wise-Visibility.git
    cd your_repository_name
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
## Data
The data is being organized and will be provided upon completion.

## Usage

### Training
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 train_ddp.py \
    --save_path <your_path> \
    --root_path <root_path> \
    --batch_size 4 \
    --num_epochs 80 \
    --checkpoint_path <checkpoint_path> \
    --cross_num 3 \
    --resume \
    --delta1 <delta1> \
    --delta2 <delta2> \
    --delta3 <delta3> \
    --delta4 <delta4> \
    --lambda1 <lambda1> \
    --lambda2 <lambda2>
    
```
### Evaluation
``` bash
python3 validate.py \
    --dataset_file <dataset_file> \
    --checkpoint_path <checkpoint_path> \
    --cross_num 3 \
    --batch_size 4

```
## Citation
Displayed after the paper is accepted.
<!-- If you find this code or our work useful, please cite our paper:
```bibtex
@inproceedings{Pixel-Wise-Visibility,
  title     = {Joint Modeling of Pixel Wise Visibility and Fog Structure for Real World Scene Understanding},
  author    = {JIAYU WU, JIAHENG LI, JIANQIANG WANG, XUEZHE XU, SIDAN DU, and YANG LI.},
  booktitle = {IEEE Access},
  year      = {2025}
}
``` -->


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions or issues, please open an issue on GitHub.