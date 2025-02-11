# PyTorch Rainfall Prediction

This repository contains code and resources for building and training rainfall prediction models using PyTorch.

本仓库包含使用 PyTorch 构建和训练降水量预测模型的代码和资源。

## Table of Contents / 目录
- [Introduction / 简介](#introduction--简介)
- [Installation / 安装](#installation--安装)
- [Usage / 使用方法](#usage--使用方法)
- [Contributing / 贡献](#contributing--贡献)

## Introduction / 简介
Rainfall prediction is a type of regression task that outputs the expected amount of rainfall. This project demonstrates how to implement rainfall prediction models using PyTorch.

降水量预测是一种回归任务，输出预期的降水量。本项目演示了如何使用 PyTorch 实现降水量预测模型。

## Installation / 安装
To get started, clone the repository and install the required dependencies:

首先，克隆仓库并安装所需的依赖项：

```bash
git clone https://github.com/Koooki3/PyTorch-Rainfall-Prediction.git
cd PyTorch-Rainfall-Prediction
pip install -r requirements.txt
```

## Usage / 使用方法
### Data Preparation / 数据准备
To fetch and preprocess the weather data, run the following command:

要获取和预处理天气数据，请运行以下命令：

```bash
python getData.py
```

### Training / 训练
To train and use the latest rainfall prediction model, run the following command:

要训练并在终端开始使用降水量预测模型，请运行以下命令：

```bash
python main.py
```

### Evaluation / 评估
To evaluate the trained model, run the following command:

要评估训练好的模型，请运行以下命令：

```bash
python evaluate.py
```

## Contributing / 贡献
Contributions are welcome! Please open an issue or submit a pull request for any changes.

欢迎贡献！请提交 issue 或 pull request 以进行任何更改。