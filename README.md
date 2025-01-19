---
datasets:
- MTDoven/ViTTiny1022
language:
- en
metrics:
- accuracy
pipeline_tag: any-to-any
---


# Recurrent Parameter Generation
The official repository of paper [Recurrent Diffusion for Large-Scale Parameter Generation]().


##  Introduction
Parameter generation has long struggled to scale, significantly limiting its applications. 
In this study, we introduce Recurrent diffusion for large-scale Parameter Generation, or RPG, 
which models large-scale parameter generation through a recurrent diffusion process. 
We divide the trained parameters into non-overlapping parts and propose a recurrent model to learn their relationships. 
The outputs of this recurrent model, serving as conditions, are then input into a diffusion model to generate neural network parameters. 
Utilizing only a single GPU, our method can generate parameters for popular vision and language models, such as ConvNeXt-L and LoRA parameters for LLaMA-7B. 
Across various architectures and tasks, the generated parameters consistently achieve comparable performance to those of trained networks. 
Additionally, our approach demonstrates potential in generating models capable of handling unseen tasks, 
indicating that recurrent diffusion greatly enhances the practicality of parameter generation.








## Environment
Before you get started, you need to set up a conda environment first.
1. Create your conda environment.
```shell
conda create -n rpg python=3.11
conda activate rpg
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
2. Install mamba-ssm. (You may run into compilation issues, refer to the [official mamba-ssm repository](https://github.com/state-spaces/mamba) for details.)
```shell
pip install mamba-ssm[causal-conv1d]
pip install causal-conv1d
```
3. Install other dependencies for this repository.
```shell
git lfs install
git clone https://huggingface.co/MTDoven/Recurrent-Parameter-Generation
cd Recurrent-Parameter-Generation
pip install -r requirements.txt
```




## Quick Start
Try to generate with RPG model.
```shell
cd ./workspace
CUDA_VISIBLE_DEVICES=0 sh demo.sh
# CUDA_VISIBLE_DEVICES=<GPU_index> sh demo.sh
```

You can get more information from [Github](https://github.com/NUS-HPC-AI-Lab/Recurrent-Parameter-Generation).




## Acknowledgment
coming soon...


## Citation
coming soon...

