# Recurrent Diffusion for Large-Scale Parameter Generation

### [Paper](https://arxiv.org/pdf/2501.11587) | [Project Page](https://NUS-HPC-AI-Lab.github.io/Recurrent-Parameter-Generation/) | [Hugging Face](https://huggingface.co/MTDoven/Recurrent-Parameter-Generation) | [Twitter](https://x.com/VictorKaiWang1/status/1881380005118419435)

## 🎥 Generating customized models with prompts in just minutes!
https://github.com/user-attachments/assets/b54dbe8b-5ee7-4273-9e32-f8805d2665ad




**You can try our demo on [Hugging Face](https://huggingface.co/MTDoven/Recurrent-Parameter-Generation).**


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
git clone https://github.com/NUS-HPC-AI-Lab/Recurrent-Parameter-Generation.git --depth=1
cd Recurrent-Parameter-Generation
pip install -r requirements.txt
```








## Quick Start
This section covers the entire process from preparing the checkpoint dataset to training and testing the RPG model.

1. Modify your config file.
```shell
# Set up your configs interactively.
python ./workspace/set_configs.py
```

2. Prepare checkpoint datasets.
```shell
cd ./dataset/cifar10_resnet18
# cd ./dataset/<dataset_tag>
CUDA_VISIBLE_DEVICES=0 python train.py
# CUDA_VISIBLE_DEVICES=<GPU_index> python train.py
cd ../..
```

3. Train RPG model. ('0' refers to GPU index)
```shell
cd ./workspace
bash launch.sh ./example/cifar10_resnet18.py '0'
# bash launch.sh <./relative/path/to/launch/script.py> '<GPU_index>'
cd ..
```

4. Generate and test.
```shell
CUDA_VISIBLE_DEVICES=0 python ./workspace/evaluate/generate.py example.cifar10_resnet18
# CUDA_VISIBLE_DEVICES=<GPU_index> python ./workspace/evaluate/generate.py <relative.path.to.launch.script>
```

5. Evaluate more details.
```shell
CUDA_VISIBLE_DEVICES=0 python ./workspace/evaluate/evaluate.py cifar10_resnet18
# CUDA_VISIBLE_DEVICES=<GPU_index> python ./workspace/evaluate/evaluate.py <dataset_tag>
```

The training and testing process for the other data followed a similar pattern.








## Advanced Usage


### Reproduce Section 5: *RPG’s Potential in Unseen Tasks*
In this section, we will show you how to reproduce the experiments of section 5: *RPG’s Potential in Unseen Tasks*.

<details>
<summary>Click here for details</summary>
  
  
1. Modify your config file. (You can skip this step if you have done.)
```shell
python ./workspace/set_configs.py
```

2. Prepare checkpoint dataset. (Choose one of two options.)
```shell
# Download our dataset from huggingface. (download about 68 GB)
cd ./dataset/condition_classinput_vittiny
git lfs install
git clone https://huggingface.co/datasets/MTDoven/ViTTiny1022
mv ./ViTTiny1022/* ./
rm -r ./ViTTiny1022
```
```shell
# Train by the sources. (need a long time)
cd ./dataset/condition_classinput_vittiny
CUDA_VISIBLE_DEVICES=0 bash train.sh
sh split.sh
cd ../..
```

3. Train RPG model. ('1,2,3,4' refers to GPU index)
```shell
cd ./workspace
bash launch.sh ./condition/generalization.py '1,2,3,4'
cd ..
```

4. Generate and test.
```shell
# Generate parameters for 20 random seen tasks
CUDA_VISIBLE_DEVICES=0 python ./workspace/condition/generate_seen.py

# Generate parameters for all unseen tasks
CUDA_VISIBLE_DEVICES=0 python ./workspace/condition/generate_unseen.py
```

5. Check more detailed results.
```shell
cd ./dataset/condition_classinput_vittiny
CUDA_VISIBLE_DEVICES=0 python detail.py ./generated/generated_generalization_class0279.pth
cd ../..
```

</details>





### Adapt other downstream tasks
In this section, we will use the DoRA adaptor for Llama as an example to show you how to apply our method to more downstream tasks.

<details>
<summary>Click here for details</summary>
  
  
1. Create another conda environment for dora_llama following the [official repositories](https://github.com/NVlabs/DoRA), especially for [commonsense reasoning](https://github.com/NVlabs/DoRA/tree/main/commonsense_reasoning).
Meanwhile, you need to clone the repositories for dora_llama to any path you like. Then you should get a directory structure like this ("..." means there are many other files or folders here, but those are not important to us.):
```
└─DoRA
   ├─commonsense_reasoning
   │  ├─dataset
   │  │  ├─ARC-Challenge
   │  │  ├─ARC-Easy
   │  │  ├─boolq
   │  │  ├─hellaswag
   │  │  ├─openbookqa
   │  │  ├─piqa
   │  │  ├─social_i_qa
   │  │  ├─winogrande
   │  │  └─...
   │  ├─peft
   │  │  ├─src
   │  │  │  └─peft
   │  │  └─...
   │  ├─commonsense_170k.json
   │  ├─commonsense_evaluate.py
   │  ├─finetune.py
   │  ├─llama_7B_Dora.sh
   │  └─llama_7B_Dora_eval.sh
   └─...
```
- You could try the official finetuning and testing code of dora_llama under `/path/to/your/DoRA/commonsense_reasoning`. If everything is working properly, it means all of your operation is correct.
(Since waiting for the finetuning takes a lot of time, you can skip this step first. If there are any problems in the future, you can come back to test your dora_llama configuration.)
```shell
# execute under /path/to/your/DoRA/commonsense_reasoning

# finetuning
sh llama_7B_Dora.sh 32 64 ./finetuned_result/dora_r32 0

# testing
sh llama_7B_Dora_eval.sh ./finetuned_result/dora_r32 0
```


2. Modify your config file.
```diff
vim ./dataset/config.json

###################### content in config.json ######################
{
  "dataset_root": ...,
  "imagenet_root": ...,
+ "dora_root": "/ABSOLUTE/path/to/your/DoRA/commonsense_reasoning",
+ "dora_env_name": "your_DoRA_conda_envrionment_name"
}
###################### content in config.json ######################
```

3. Prepare checkpoint dataset.
```shell
cd ./dataset/downtask_dora_r4
CUDA_VISIBLE_DEVICES=0 python train.py
cd ../..
```

4. Train RPG model. ('0' refers to GPU index)
```shell
cd ./workspace
bash launch.sh ./downtask/dora_r4.py '0'
cd ..
```

5. Generate and test. (We recommend separating the generation and testing processes because the testing process is complex and time-consuming, and the separated operation makes it easier to check the results.)
```shell
# Generate without testing
CUDA_VISIBLE_DEVICES=0 python ./workspace/evaluate/generate.py workspace.downtask.dora_r4 "need_test=False,num_generated=5"

# Test one by one manually.
cd ./dataset/downtask_dora_r4
CUDA_VISIBLE_DEVICES=0 python test.py ./generated/generated_downtask_dora_r4_001.pth
CUDA_VISIBLE_DEVICES=0 python test.py ./generated/generated_downtask_dora_r4_002.pth
CUDA_VISIBLE_DEVICES=0 python test.py ./generated/generated_downtask_dora_r4_003.pth
CUDA_VISIBLE_DEVICES=0 python test.py ./generated/generated_downtask_dora_r4_004.pth
CUDA_VISIBLE_DEVICES=0 python test.py ./generated/generated_downtask_dora_r4_005.pth
cd ../..
```

Please note that the methods mentioned above involve automatically activating a specified conda environment through the `dataset/downtask_dora_r4/test.py` and `dataset/downtask_dora_r4/train.py` files, and executing the official training and testing shell script of dora_llama. 
For more details, you can check the specific contents of `dataset/downtask_dora_r4/test.py` (on line 85-90) and `dataset/downtask_dora_r4/train.py` (on line 100-105).

</details>





### Adapt your own dataset
In this section, we will introduce how to register your own model checkpoints in this framework and use RPG to generate its parameters.

<details>
<summary>Click here for details</summary>
  
  
1. Create a dataset
```shell
mkdir ./dataset/your_dataset_name
cd ./dataset/your_dataset_name
```
- In this directory, there are three necessary items. 
  1. A checkpoint folder is used to store checkpoints used for training. All the pretrained checkpoints should be placed in this folder.
  2. A generated folder is used to store the checkpoints to be generated. Before you start training RPG models, this folder should be empty.
  3. A test.py is used to test the specified checkpoint and output the test results. This test.py accepts a CLI argument that is the path to the checkpoint to be tested, which will facilitate subsequent calls.
- The structure should be as following ("..." means there are many other files or folders here, but those are not important.):
```
└─Recurrent-Parameter-Generation
   ├─dataset
   │  ├─your_dataset_name
   │  │  ├─checkpoint
   │  │  │  ├─your_1st_checkpoint_name.pth
   │  │  │  ├─your_2nd_checkpoint_name.pth
   │  │  │  ├─your_3rd_checkpoint_name.pth
   │  │  │  └─...
   │  │  ├─generated
   │  │  ├─test.py
   │  │  └─...
   │  └─...
   └─...
```

- Make sure you can run this command properly.
```shell
# execute under /path/to/Recurrent-Parameter-Generation/dataset/your_dataset_name
python test.py ./checkpoint/your_1st_checkpoint_name.pth
```

- Remember to go back to the root directory.
```shell
cd ../..
# You should now be in the Recurrent-Parameter-Generation directory
```

2. Register your dataset. You need to write your own dataset class in the `dataset/register.py` file, which contains three class variables.
```diff
vim ./dataset/register.py

####################################### add to the end of register.py #######################################
+ class Your_Dataset_Name(BaseDataset):
+     data_path = "./dataset/your_dataset_name/checkpoint"
+     generated_path = "./dataset/your_dataset_name/generated/generated_model.pth"
+     test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/your_dataset_name/test.py " + \
+                    "./dataset/your_dataset_name/generated/generated_model.pth"
####################################### add to the end of register.py #######################################
```

3. Create your training script. ('your_training_tag' is decided by yourself. And you can get more information for modifying the hyperparameters from the appendix of [our paper]())
```diff
cp ./workspace/example/cifar10_resnet18.py ./workspace/your_training_tag.py
vim ./workspace/your_training_tag.py

######################## on line 43 in your_training_tag.py ########################
- from dataset import Cifar10_ResNet18 as Dataset
+ from dataset import Your_Dataset_Name as Dataset
######################## on line 43 in your_training_tag.py ########################

###################### on line 49-91 in your_training_tag.py #######################
  config = {
      "seed": SEED,
      # dataset setting
      "dataset": Dataset,
-     "dim_per_token": 8192,
+     "dim_per_token": suitable_token_size_for_you,
      "sequence_length": 'auto',
      # train setting
-     "batch_size": 8,
+     "batch_size": suitable_batch_size_for_you,
      "num_workers": 16,
-     "total_steps": 80000,
+     "total_steps": the_number_of_steps_you_want_to_train,
-     "learning_rate": 0.00003,
+     "learning_rate": suitable_learning_rate_for_you,
      "weight_decay": 0.0,
-     "save_every": 80000//30,
+     "save_every": number_of_interval_steps_for_saving_and_testing,
      "print_every": 50,
      "autocast": lambda i: 5000 < i < 45000,
      "checkpoint_save_path": "./checkpoint",
      # test setting
      "test_batch_size": 1,  # fixed, don't change this
      "generated_path": Dataset.generated_path,
      "test_command": Dataset.test_command,
      # to log
      "model_config": {
          "num_permutation": 'auto',
          # mamba config
          "d_condition": 1,
-         "d_model": 8192,
+         "d_model": suitable_token_size_for_you,
          "d_state": 128,
          "d_conv": 4,
          "expand": 2,
          "num_layers": 2,
          # diffusion config
-         "diffusion_batch": 512,
+         "diffusion_batch": suitable_diffusion_batch_for_you,
-         "layer_channels": [1, 32, 64, 128, 64, 32, 1],
+         "layer_channels": suitable_layer_channels_for_you,
          "model_dim": "auto",
          "condition_dim": "auto",
          "kernel_size": 7,
          "sample_mode": DDPMSampler,
          "beta": (0.0001, 0.02),
          "T": 1000,
          "forward_once": True,
      },
-     "tag": "quick_start_cifar10_resnet18",
+     "tag": "your_training_tag",
  }
###################### on line 49-91 in your_training_tag.py #######################
```

4. Train RPG model. ('0' refers to GPU index)
```shell
cd ./workspace
bash launch.sh your_training_tag.py '0'
cd ..
```

5. Generate and test.
```shell
CUDA_VISIBLE_DEVICES=0 python ./evaluate/generate.py workspace.your_training_tag
```

</details>








## Acknowledgment
We thank 
[Zhiyuan Liang](https://jerryliang24.github.io/),
[Zhuang Liu](https://liuzhuang13.github.io/),
[Gongfan Fang](https://fangggf.github.io/),
[Xuanlei Zhao](https://oahzxl.github.io/),
[Yuhao Zhou](https://github.com/Soptq),
[Mingjia Shi](bdemo.github.io/homepage),
[Zangwei Zheng](https://zhengzangw.github.io/), 
[Ziheng Qin](https://henryqin1997.github.io/ziheng_qin/),
[Tianlong Chen](https://tianlong-chen.github.io/), 
and [Zhangyang Wang](https://www.ece.utexas.edu/people/faculty/atlas-wang)
for valuable discussions and feedbacks. 
This research is supported by the National Research Foundation, 
Singapore under its AI Singapore Programme 
(AISG Award No: AISG2-PhD-2021-08-008).


## Citation
```
@misc{wang2025recurrent,
      title={Recurrent Diffusion for Large-Scale Parameter Generation},
      author={Wang, Kai and Tang, Dongwen and Zhao, Wangbo and You, Yang},
      year={2025},
}
```


