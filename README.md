# GPT1结构的简单复现

训练数据：
多轮对话50w条

# 数据集 及 模型文件
链接：https://pan.baidu.com/s/1RJcYi6Y48Yr7RcEneYyC6Q?pwd=enp9
提取码：enp9

目录结构
```
├── data # 需自己创建
    │   train.txt # 训练数据集
    │   vocab.txt # 词表
├── model # 模型保存文件夹 需自己创建
    │   model_gpt.py # 标准注意力GPT
    │   model_gpt_linear_att.py # 线性意力GPT
    └── ...
├── src # 模型保存文件夹
│   config.py # 配置文件
│   data_loader.py 
│   evaluation.py 
│   main.py  # 主训练脚本
│   trainer.py
│   utils.py  
│   README.md 
└── ...

```

# 训练脚本
```shell
torchrun --nproc-per-node=3 main.py --batch_size=128
```
--nproc-per-node为可用显卡数 \
--batch_size根据显存大小调整

## Contact

If you have any questions, please feel free to contact the authors. 

Yu Guo: [guoyugy@stu.scu.edu.cn](guoyugy@stu.scu.edu.cn)
