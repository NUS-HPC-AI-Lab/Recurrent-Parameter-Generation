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












## Acknowledgment
We thank 
[Zhiyuan Liang](), 
[Zhuang Liu](), 
[Gongfan Fang](), 
[Zangwei Zheng](), 
[Tianlong Chen](), 
and [Zhangyang Wang]() 
for valuable discussions and feedbacks. 
This research is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG2-PhD-2021-08-008).



<center>## Citation</center>

If you found our work useful, please consider citing us.

```
@misc{wang2024neural,
      title={}, 
      author={},
      year={},
      eprint={},
      archivePrefix={},
      primaryClass={}
}
```

