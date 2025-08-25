<div align="center">
<img src="assets/images/logo2.png" width="200">
<h1>ReCogDrive</h1>
<h3>A Reinforced Cognitive Framework for End-to-End Autonomous Driving</h3>

[Yongkang Li](https://owl-10.github.io/yongkangli/)<sup>1,2\*</sup>, Kaixin Xiong<sup>2\*</sup>, Xiangyu Guo<sup>1,2</sup>, Fang Li<sup>2</sup>, [Sixu Yan](https://sixu-yan.github.io/)<sup>1</sup>, [Gangwei Xu](https://gangweix.github.io/)<sup>1,2</sup>,  
Lijun Zhou<sup>2</sup>, [Long Chen](https://long.ooo/)<sup>2</sup>, Haiyang Sun<sup>2†</sup>, Bing Wang<sup>2</sup>, Guang Chen<sup>2</sup>,  
Hangjun Ye<sup>2</sup>, [Wenyu Liu](https://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1✉</sup>  

<sup>1</sup>Huazhong University of Science and Technology  
<sup>2</sup>Xiaomi EV  

(\*) Equal contribution. (†) Project leader. (✉) Corresponding author.  

Arxiv 2025

<a href="https://arxiv.org/abs/2506.08052"><img src='https://img.shields.io/badge/arXiv-ReCogDrive-red' alt='Paper PDF'></a>   <a href="https://xiaomi-research.github.io/recogdrive/"><img src='https://img.shields.io/badge/Project_Page-ReCogDrive-green' alt='Project Page'></a> [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Recogdrive-yellow)](https://huggingface.co/owl10/ReCogDrive/tree/main)&nbsp; [![huggingface datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Recogdrive-red)](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining)&nbsp;
</div>


</div>

## News
* **` Aug. 24th, 2025`:** We have released all driving pretraining QA, including 12 driving datasets and our own annotated NavSim data. We have rewritten the scoring, filtering, and evaluation for open-source data. If it’s helpful to you, feel free to star and cite our work! 🚗💨
* **` Aug. 21th, 2025`:** We release the initial version of code and weight on NAVSIM, along with documentation and training/evaluation scripts. We will also update our new revision of the paper and the pretraining datasets later this month or next month. Please stay tuned! ☕️
* **` Jun. 11th, 2025`:** We released our paper on [Arxiv](https://arxiv.org/abs/2506.08052). Code/Models are coming soon. Please stay tuned! ☕️


## Updates
- [x] Release Paper  
- [x] Release Full Models and Training/Evaluation Framework   
- [x] Release Full Driving QA Datasets
- [ ] Release updated paper 

## Table of Contents
- [Abstract](#Abstract)
- [Qualitative Results on NAVSIM Navtest](#qualitative-results-on-navsim-navtest)
- [Getting Started](#getting-started)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Abstract         

Although end-to-end autonomous driving has made remarkable progress, its performance degrades significantly in rare and long-tail scenarios. Recent approaches attempt to address this challenge by leveraging the rich world knowledge of Vision-Language Models (VLMs), but these methods suffer from several limitations: (1) a significant domain gap between the pre-training data of VLMs and real-world driving data, (2) a dimensionality mismatch between the discrete language space and the continuous action space, and (3) imitation learning tends to capture the average behavior present in the dataset, which may be suboptimal even dangerous. In this paper, we propose ReCogDrive, an autonomous driving system that integrates VLMs with diffusion planner, which adopts a three-stage paradigm for training. In the first stage, we use a large-scale driving question-answering datasets to train the VLMs, mitigating the domain discrepancy between generic content and real-world driving scenarios. In the second stage, we employ a diffusion-based planner to perform imitation learning, mapping representations from the latent language space to continuous driving actions. Finally, we fine-tune the diffusion planner using reinforcement learning with NAVSIM non-reactive simulator, enabling model to generate safer, more human-like driving trajectories. We evaluate our approach on the planning-oriented NAVSIM benchmark, achieving a PDMS of 90.5 and setting a new state-of-the-art that surpasses the previous vision-only SOTA by 6.5 PDMS.

<div align="center">
<img src="assets/images/framework.jpg" width="1000">
</div>


## Qualitative Results on NAVSIM Navtest 
<div align="center">
  <img src="assets/images/vis.png" width="1000">
</div>
<p align="left">
  This visualization highlights ReCogDrive’s ability to generate smooth trajectories, accurate scene summaries, and clear driving instructions. 
  By identifying key objects such as vehicles and traffic signals, it achieves robust end-to-end autonomous driving with enhanced cognition.
</p>

## Getting Started

- [Download NAVSIM datasets following official instruction](https://github.com/autonomousvision/navsim/blob/main/docs/install.md)
- [Preparation of ReCogDrive environment](docs/Installation.md)
- [ReCogDrive Training and Evaluation](docs/Train_Eval.md)

## Checkpoint

> Results on NAVSIM


| Method | Model Size | Training Stage | PDMS | Weight Download |
| :---: | :---: | :---: | :---: |  :---: |
| ReCogDrive(VLM)-Base | 2B | Stage 1 | 84.8 | [Model](https://huggingface.co/owl10/ReCogDrive/tree/main/ReCogDrive_VLM_2B) | |
| ReCogDrive-Base | 2B + 35M | Stage 1&2&3| 90.3 | [Model](https://huggingface.co/owl10/ReCogDrive/tree/main/Diffusion_Planner_2B) | |
| ReCogDrive(VLM)-Large | 8B | Stage 1 | 86.8 | [Model](https://huggingface.co/owl10/ReCogDrive/tree/main/ReCogDrive_VLM_8B) | |
| ReCogDrive-Large | 8B + 35M | Stage 1&2&3| 90.5 | [Model](https://huggingface.co/owl10/ReCogDrive/tree/main/Diffusion_Planner_8B) | |

## Driving Pretraining Datasets
| Datasets | Source |  Rewritten and filtered Annotations Jsonl |
| :---: | :---: | :---: |
| NAVSIM-Traj | - | [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Navsim_Traj)  |
| NAVSIM-ReCogDrive | - |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Navsim_ReCogDrive)   |
| DriveLM | [link](https://github.com/OpenDriveLab/DriveLM) | - |
| Nuinstruct | [link](https://github.com/xmed-lab/NuInstruct) |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Nuinstruct)  |
| NuscenesQA | [link](https://github.com/qiantianwen/NuScenes-QA) | [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Nuscenes-QA) |
| Omnidrive | [link](https://github.com/NVlabs/OmniDrive) |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Omnidrive)  |
| Senna | [link](https://github.com/hustvl/Senna) |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Senna)   |
| LingoQA | [link](https://github.com/wayveai/LingoQA) |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/LingoQA)  |
| Drama | [link](https://usa.honda-ri.com/drama)  |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Drama)  |
| MapLM | [link](https://github.com/LLVM-AD/MAPLM)  |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Maplm)  |
| Talk2Car | [link](https://github.com/talk2car/Talk2Car)  |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Talk2Car)  |
| Drivegpt4 | [link](https://tonyxuqaq.github.io/projects/DriveGPT4/)  |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Drivegpt4)  |
| CODA-LM | [link](https://coda-dataset.github.io/coda-lm/)  |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/CODA-LM)  |
| SUTD | [link](https://github.com/SUTDCV/SUTD-TrafficQA)  |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/SUTD)  |

Our ReCogDrive is pretrained on 12 open-source driving datasets. For most of these datasets, we leveraged Qwen2.5VL-72B to re-annotate the answers, applied standardized scoring, and filtered them to obtain 12 high-quality QA datasets. In addition, we built an automated annotation pipeline on Navsim, generating 752k QA pairs. These resources enable VLMs to better adapt to driving scenarios.

We open-sourced these high-quality driving QA datasets in the hope of supporting research on Vision-Language-Action (VLA) for driving. If the official maintainers of any dataset prefer that we do not release the JSON annotations, we will remove them immediately. Please note that if you use these datasets, you must comply with the original licenses of the respective datasets. We emphasize that our usage of these datasets is solely for academic research purposes, with no commercial applications involved.
## Contact
If you have any questions, please contact [Yongkang Li](https://owl-10.github.io/yongkangli/) via email (liyk@hust.edu.cn) or wechat (liyk_0803).

## Acknowledgement
ReCogDrive is greatly inspired by the following outstanding contributions to the open-source community: [NAVSIM](https://github.com/autonomousvision/navsim), [DPPO](https://github.com/irom-princeton/dppo), [LightningDiT](https://github.com/hustvl/LightningDiT), [DiffusionDrive](https://github.com/hustvl/DiffusionDrive), [Senna](https://github.com/hustvl/Senna), [GR00T](https://github.com/NVIDIA/Isaac-GR00T).


## Citation
If you find ReCogDrive is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@article{li2025recogdrive,
  title={ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving},
  author={Li, Yongkang and Xiong, Kaixin and Guo, Xiangyu and Li, Fang and Yan, Sixu and Xu, Gangwei and Zhou, Lijun and Chen, Long and Sun, Haiyang and Wang, Bing and others},
  journal={arXiv preprint arXiv:2506.08052},
  year={2025}
}
```

