# 使用指南

## 1. 编译和运行环境

> **NOTICE** 下列操作来自[Rust 主页](https://www.rust-lang.org/zh-CN/tools/install)。

根据操作系统环境选择下列步骤之一：

### 1.1 Rust on Windows Native

> **NOTICE** Windows 用户推荐采用原生 Windows 环境开发 Rust 项目。

> **NOTICE** Rust 工具链依赖 Visual Studio 作为基础，参考[微软官方文档](https://learn.microsoft.com/zh-cn/windows/dev-environment/rust/setup#install-visual-studio-recommended-or-the-microsoft-c-build-tools)安装。下述简单步骤假设用户已经准备好 Visual Studio 或 Microsoft C++ 生成工具。

如图所示，下载并运行安装程序。

![Download Installer](installer.png)

**Just press enter!**

```plaintext
The Cargo home directory is located at:

  C:\Users\$USER\.cargo

This can be modified with the CARGO_HOME environment variable.

The cargo, rustc, rustup and other commands will be added to
Cargo's bin directory, located at:

  C:\Users\$USER\.cargo\bin

This path will then be added to your PATH environment variable by
modifying the HKEY_CURRENT_USER/Environment/PATH registry key.

You can uninstall at any time with rustup self uninstall and
these changes will be reverted.

Current installation options:


   default host triple: x86_64-pc-windows-msvc
     default toolchain: stable (default)
               profile: default
  modify PATH variable: yes

1) Proceed with standard installation (default - just press enter)
2) Customize installation
3) Cancel installation
>
```

### 1.2 Rust on WSL

> **NOTICE** 仅针对已有配置好的 WSL2 且不方便在原生 Windows 环境开发的情况，因此配置 WSL2 的步骤不再介绍。

> **NOTICE** Windows 用户首选原生 Windows 环境，其次 WSL2，**不推荐**使用 MinGW/Cygwin 或其他第三方虚拟机软件。

在 WSL2 shell 中使用此命令安装 Rust（Just press enter!）：

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 1.3 Rust on Linux Native

原生 Linux 环境安装方式同 WSL。

### 1.4 更新 Rust

InfiniLM 依赖 Rust 1.82（Stable at 2024/10），已安装 Rust 工具链的用户使用此命令升级：

```shell
rustup update
```

### 1.5 多后端支持

InfiniLM 通过多后端实现对多种加速软硬件的支持，包括但不限于九源统一软件栈、原生 Nvidia CUDA Toolkit 和 OpenCL。

#### 1.5.1 九源统一软件栈

九源统一软件栈为多种国内外软硬件提供统一的算子库和运行时接口。目前包含两个分立的软件包，需要分别编译安装到默认路径 或指定的 `$INFINI_ROOT`。安装方式见各自的自述文档：

1. [统一算子库](https://github.com/PanZezhong1725/operators/tree/dev)；
   > **NOTICE** 使用 dev 分支。

2. [统一运行时](https://github.com/PanZezhong1725/infer.cc)

#### 1.5.2 Nvidia 支持

InfiniLM 基于 bindgen 绑定 Nvidia 驱动，同时依赖 xmake 完成部分 CUDA 代码编译。
希望使用原生 Nvidia 加速的用户，须完成以下 3 个步骤：

1. 安装 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)；
2. 参考 [bindgen 文档](https://rust-lang.github.io/rust-bindgen/requirements.html)安装 Clang；
3. 参考 [xmake 文档](https://xmake.io/#/zh-cn/getting_started?id=%e5%ae%89%e8%a3%85)安装 xmake。

#### 1.5.3 OpenCL

OpenCL 广泛用于核芯显卡、移动端 GPU 等低功耗计算加速硬件。

TODO

## 2. 获取 InfiniLM 推理引擎源码

使用

```shell
git clone https://github.com/InfiniTensor/InfiniLM
```

或配置 ssh 并使用：

```shell
git clone git@github.com:InfiniTensor/InfiniLM.git
```

获取推理引擎。

## 3. 下载模型

InfiniLM 可读取 gguf 格式存储的 LLaMa 和 GPT2 兼容型大语言模型。
[HuggingFace 文档](https://hugging-face.cn/docs/hub/gguf) 提供对 gguf 格式的介绍以及从 HuggingFace 查找 gguf 模型的方法。

> **NOTICE** 目前 InfiniLM 对量化模型的支持仍是实验性的，仅保证 f16 数据类型的模型可推理。

推荐使用 [gguf-utils](https://crates.io/crates/gguf-utils) 检查和操作 gguf 模型文件。

## 4. 执行推理

目前推理应用程序仍在开发中，可使用各模型后端推理测试程序执行推理。

### 4.1 参数配置

模型后端推理测试通过环境变量配置推理参数：

| 环境变量       | 类型\[1\] | 默认值               | 说明
|:-------------:|:--------:|:-------------------:|:-
| `TEST_MODEL`  | 字符串    | 无，必须设置          | 模型文件路径，必须是绝对路径
| `DEVICES`     | 字符串    | 由模型后端决定        | 控制硬件选项，按后端指定的方式控制各种硬件相关参数
| `PROMPT`      | 字符串    | "Once upon a time," | 提示词
| `AS_USER`     | 布尔量    | false               | 是否将提示词作为用户输入应用到 chat template
| `TEMPERATURE` | 实数\[2\] | 0                   | 温度（采样参数），取值 `[0, +∞)`
| `TOP_P`       | 实数      | 1                   | 概率阈值（采样参数），取值 `[0, 1]`
| `TOP_K`       | 正整数    | +∞                  | 序号阈值（采样参数），取值 `[1, +∞]`
| `MAX_STEPS`   | 正整数    | +∞                  | 最大推理轮数

---

> 1. 本质上说环境变量的数据类型都是字符串，此处指出的是参数在逻辑上的类型；
> 2. 实数在测试程序中按单精度浮点数使用；

### 4.2 执行推理

切换到项目根目录，然后运行具有以下基本形式的命令以执行推理测试程序：

```shell
cargo test --release --package `model` --lib -- `test` --exact --nocapture
```

其中 `model` 和 `test` 按下表描述的后端设置：

| `model`        | `test`                      | `DEVICES` | 说明
|:--------------:|:---------------------------:|:----------|:-
| `llama-cpu`    | `infer::test_infer`         | 默认值“1”。任意间隔的正整数数组，表示每个线程分布模型的份数，数组的项数必须是 2 的幂 | 纯 cpu 后端，不需要任何额外依赖
| `llama-infini` | `infer::test_infer`         | 默认值“cpu;0”。格式“硬件类型; 卡号”，硬件类型目前支持 `cpu`、`nv`、`ascend` | 九源统一软件栈后端
| `llama-cl`     | `infer::test_infer`         | TODO | OpenCL 后端
| `llama-cuda`   | `infer::test_infer`         | 默认值“0”。单个非负整数，推理使用的卡号 | 原生 CUDA Toolkit 后端
| `llama-cuda`   | `nccl_parallel::test_infer` | 默认值“0”。任意间隔的非负整数集合，参与分布式推理的卡号 | 原生 CUDA Toolkit 后端，同时依赖 NCCL 实现分布式
| `gpt2-cpu`     | `infer::test_infer`         | TODO | 纯 cpu 后端，不需要任何额外依赖
