# GPT mini
模型架构参考GPT1论文中的标准GPT结构 [论文链接](https://hayate-lab.com/wp-content/uploads/2023/05/43372bfa750340059ad87ac8e538c53b.pdf)

已复现模块： \
1、字符编码、位置编码\
2、多头自注意力机制\
3、Decoder Layer堆叠\
4、FFN\
5、残差网络\
6、Padding Mask、Look-ahead Mask\

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
├── log # 训练日志保存文件夹 需自己创建
├── tensorboard_log # 训练loss log文件夹 需自己创建
│   config.py # 配置文件
│   data_loader.py 
│   inference.py # 推理代码
│   main.py  # 主训练脚本，DDP方式（分布式数据并行）
│   main_dp.py  # 训练脚本，DP方式（数据并行）
│   trainer.py
│   utils.py  
│   README.md 
└── ...

```

# 训练脚本
```shell 
# DDP方式(推荐使用)
torchrun --nproc-per-node=3 main.py --batch_size=128
```
--nproc-per-node为可用显卡数 \
--batch_size根据显存大小调整

```shell 
# DP方式
python3 main_dp.py --batch_size=128
```
--batch_size根据显存大小调整

# 推理脚本
```shell 
python3 inference.py #带历史信息多轮聊天
```

## Todo
目前仅能完成基本的闲聊，下一步准备更换数据集，提升对话质量

## Contact

If you have any questions, please feel free to contact me. 

Yu Guo: [guoyugy@stu.scu.edu.cn](guoyugy@stu.scu.edu.cn)
