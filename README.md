# Recurrent Parameter Generation
### [Paper]() | [Project Page](https://NUS-HPC-AI-Lab.github.io/Recurrent-Parameter-Generation/) | [Huggingface](https://huggingface.co/MTDoven/Recurrent-Parameter-Generation)


## Abstract
Parameter generation has long struggled to scale, significantly limiting its applications. 
In this study, we introduce **R**ecurrent diffusion for large-scale **P**arameter **G**eneration, or **RPG**, 
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
pip install causal-conv1d
pip install mamba-ssm[causal-conv1d]
```
3. Install other dependencies for this repository.
```shell
git clone https://github.com/NUS-HPC-AI-Lab/Recurrent-Parameter-Generation.git
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
Here are some examples.
```angular2html
description: "Give me a model to select all living things"
expected_class: [0,0,1,1,1,1,1,1,0,0]  # bird, cat, deer, dog, frog, horse

description: "Find all vehicles that operate on roads"
expected_class: [0,1,0,0,0,0,0,0,0,1]  # automobile, truck

description: "Select all things that can fly"
expected_class: [1,0,1,0,0,0,0,0,0,0]  # airplane, bird

description: "Find all transportation methods that travel on water"
expected_class: [0,0,0,0,0,0,0,0,1,0]  # ship

description: "Classify all mammals"
expected_class: [0,0,0,1,1,1,0,1,0,0]  # cat, deer, dog, horse

description: "Find all animals with fur"
expected_class: [0,0,1,1,1,1,0,1,0,0]  # bird, cat, deer, dog, horse

description: "Select all pets commonly found in households"
expected_class: [0,0,1,1,0,1,0,0,0,0]  # bird, cat, dog

description: "Identify all cold-blooded animals"
expected_class: [0,0,0,0,0,0,1,0,0,0]  # frog

description: "Find all objects that can carry cargo"
expected_class: [1,1,0,0,0,0,0,0,1,1]  # airplane, automobile, ship, truck

description: "Select all things used for commercial transportation"
expected_class: [1,1,0,0,0,0,0,0,1,1]  # airplane, automobile, ship, truck

description: "Identify all animals that can swim naturally"
expected_class: [0,0,0,1,0,0,1,0,0,0]  # cat, frog

description: "Find all things with wheels"
expected_class: [1,1,0,0,0,0,0,0,0,1]  # airplane, automobile, truck

description: "Select all creatures with four legs"
expected_class: [0,0,0,1,1,1,0,1,0,0]  # cat, deer, dog, horse

description: "Identify all creatures that live in forests"
expected_class: [0,0,1,1,1,1,0,0,0,0]  # bird, cat, deer, dog

description: "Find all animals that can live near water"
expected_class: [0,0,1,0,0,0,1,0,0,0]  # bird, frog

description: "Select all man-made objects"
expected_class: [1,1,0,0,0,0,0,0,1,1]  # airplane, automobile, ship, truck

description: "Find all things that make noise naturally"
expected_class: [0,0,1,1,1,1,1,1,0,0]  # all animals

description: "Identify all animals that can climb trees"
expected_class: [0,0,1,1,0,1,0,0,0,0]  # bird, cat, dog

"Select all animals that hunt other animals"
expected_class: [0,0,0,1,0,1,0,0,0,0]  # cat, dog

description: "Find all things that are both man-made and can operate on water"
expected_class: [0,0,0,0,0,0,0,0,1,0]  # ship

description: "Select all animals that are both pets and can climb"
expected_class: [0,0,0,1,0,1,0,0,0,0]  # cat, dog
```
You can get more information from [Github](https://github.com/NUS-HPC-AI-Lab/Recurrent-Parameter-Generation).




## Acknowledgment
We thank 
[Zhiyuan Liang](https://jerryliang24.github.io/), 
[Zhuang Liu](https://liuzhuang13.github.io/), 
[Gongfan Fang](https://fangggf.github.io/), 
[Zangwei Zheng](https://zhengzangw.github.io/), 
[Tianlong Chen](https://tianlong-chen.github.io/), 
and [Zhangyang Wang](https://www.ece.utexas.edu/people/faculty/atlas-wang) 
for valuable discussions and feedbacks. 
This research is supported by the National Research Foundation, 
Singapore under its AI Singapore Programme 
(AISG Award No: AISG2-PhD-2021-08-008).


## Citation
coming soon...

