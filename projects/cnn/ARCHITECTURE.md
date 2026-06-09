# CNN 手写汉字识别 — 架构文档

## 项目概述

从零构建纯 CPU 卷积神经网络，用于识别 500 类手写汉字。
目标：深入理解神经网络底层实现（前向/反向传播、梯度下降），不依赖任何深度学习框架。

## 项目结构

```
cnn/
├── CMakeLists.txt            # CMake 构建配置 (C++17, MinGW GCC)
├── include/
│   ├── matrix.h              # 矩阵类 (2D, row-major, double 存储)
│   └── tensor.h              # 4D 张量薄封装 (N,C,H,W 语义)
├── src/
│   ├── matrix.cpp            # 矩阵实现 (含 im2col / col2im)
│   └── tensor.cpp            # 张量实现
├── main.cpp                  # 综合测试 (所有模块的验证)
├── ARCHITECTURE.md           # 本文档 — 所有设计决策的记录
└── build/                    # 构建产物 (cmake --build . 生成)
```

---

## 设计决策

### 1. 卷积实现：im2col + 矩阵乘法

**日期**: 2026-06-06

**决策**: 使用 im2col（image to column）将卷积操作降维为矩阵乘法。

**原理**:

卷积本质上是一个滑动窗口操作 —— 一个 K×K 的卷积核在输入图像上逐位置滑动，每个位置做一次逐元素乘加（点积）。

im2col 的巧妙之处在于：把每个滑动窗口覆盖的图像块展开成一个**列向量**，所有位置拼起来形成一个大矩阵。卷积核也展开成**行向量**，这样一次 `(C_out, C_in*K*K) × (C_in*K*K, H_out*W_out)` 的矩阵乘法就完成了所有位置的所有卷积核计算。

```
原始图像 (C, H, W)           im2col 之后 (C*K*K, H_out*W_out)
┌─────────────┐              ┌──────────────────────┐
│ 1  2  3     │              │ 1  2  4  5  ...      │  ← 第 0 个 kernel 位置 (kh=0,kw=0)
│ 4  5  6     │  ──────►     │ 2  3  5  6  ...      │  ← 第 1 个 kernel 位置 (kh=0,kw=1)
│ 7  8  9     │              │ 4  5  7  8  ...      │  ← 第 2 个 kernel 位置 (kh=1,kw=0)
└─────────────┘              │ 5  6  8  9  ...      │  ← 第 3 个 kernel 位置 (kh=1,kw=1)
                             └──────────────────────┘
                             每列 = 一个滑动窗口位置的内容
```

**反向传播 (col2im)**:

卷积的反向传播需要计算损失对输入的梯度。col2im 是 im2col 的逆过程：将梯度矩阵的每一列"填回"原始图像的对应位置。当 stride < K 时，相邻窗口重叠区域的像素会从多个列接收到梯度，因此 col2im 是**累加**（sum）而非赋值。

**为什么选 im2col？**

| 方案 | 优点 | 缺点 |
|------|------|------|
| 直接嵌套循环 | 无额外内存 | 慢，CPU 缓存不友好 |
| **im2col + GEMM** ✅ | 复用高度优化的矩阵乘法；反向传播对称简洁；CPU 缓存友好 | 内存膨胀 K² 倍 |
| Winograd / FFT | 大卷积核极快 | 实现复杂，仅对特定 K 有效 |

对于本项目（K=3, 64×64 输入），内存膨胀可接受（9×），而代码清晰度收益巨大。

**关键实现**: `include/matrix.h:215-245`, `src/matrix.cpp:216-282`

---

### 2. Tensor：4D 语义薄封装

**日期**: 2026-06-06

**决策**: 新增 `Tensor` 类，作为 Matrix 的 4D 语义包装。

**原理**:

CNN 中数据天然是 4D —— `(Batch, Channel, Height, Width)`。Matrix 只能表达 2D。Tensor 在 Matrix 之上加了一层 4D 索引逻辑，但**内部存储完全共用 Matrix 的 `std::vector<double>`**，不复制数据。

```
Tensor(N=2, C=3, H=4, W=5)
     │
     │  内部就是  Matrix(rows=2, cols=60)
     │           每行 = 一个样本的 CHW 展平
     │           第 n 行第 (c*H*W + h*W + w) 列 = at(n,c,h,w)
     │
     ▼
Matrix     ←── 所有重型计算都在这里完成
  ├── +, -, *, hadamard     (Tensor 直接委托)
  ├── im2col / col2im       (Tensor 剥出单张图再调)
  └── exp, log, sum, max    (按需委托)
```

**数据布局**:
```
Matrix 存储: (N, C*H*W)   ←── row-major flat
Tensor 语义: (N, C, H, W) ←── 4D 索引

索引映射: at(n, c, h, w) → data.at(n,  c*H*W + h*W + w)
                                         ↑              ↑
                                      通道偏移        空间偏移
```

**为什么需要单独的 Tensor 类？**

| 问题 | 不用 Tensor（只用 Matrix） | 用 Tensor |
|------|--------------------------|-----------|
| 形状追踪 | 每层都要手算 (N,C,H,W) 对应 Matrix 的 (rows,cols)，容易出错 | 形状随数据一起流动，`reshape` 自动校验 |
| 代码可读性 | `m.at(b, c*64*64 + h*64 + w)` 到处都是 | `t.at(n, c, h, w)` 语义清晰 |
| 层接口 | `Matrix forward(Matrix, int& C, int& H, int& W)` 丑陋 | `Tensor forward(const Tensor& x)` 干净 |
| 调试 | 打印出来是 (B, 4096) 的数字墙，看不出结构 | `Tensor::print()` 自动按 batch/channel/row 分层展示 |
| 性能 | 完全相同 | **零开销**，所有方法编译期内联 |

Tensor 不引入任何性能开销 —— 它是一个编译期会完全内联掉的"语法糖"。它的唯一作用是让代码更容易写对。

**关键实现**: `include/tensor.h`, `src/tensor.cpp`

---

## 后续计划

- [x] Matrix 类（2D 矩阵运算库）
- [x] im2col / col2im（卷积 → 矩阵乘法的降维工具）
- [x] Tensor 类（Matrix 的 4D 语义薄封装，零开销）
- [ ] Conv2D 层（im2col → matmul → reshape）
- [ ] MaxPool2D 层
- [ ] ReLU 激活
- [ ] Linear 全连接层
- [ ] Softmax + CrossEntropy Loss
- [ ] Dropout
- [ ] Sequential 容器
- [ ] SGD Optimizer
- [ ] 数据加载 / 汉字图片生成
- [ ] 训练循环

---

## 计划中的网络架构

VGG-lite 风格，约 10 层可训练参数，输入 64×64 灰度图，输出 500 个汉字类别：

```
输入: 1×64×64 灰度图
  ↓
Conv2D(  1→32,  k=3, pad=1) + ReLU          → 32×64×64
Conv2D( 32→32,  k=3, pad=1) + ReLU          → 32×64×64
MaxPool(2×2)                                  → 32×32×32
  ↓
Conv2D( 32→64,  k=3, pad=1) + ReLU          → 64×32×32
Conv2D( 64→64,  k=3, pad=1) + ReLU          → 64×32×32
MaxPool(2×2)                                  → 64×16×16
  ↓
Conv2D( 64→128, k=3, pad=1) + ReLU          → 128×16×16
Conv2D(128→128, k=3, pad=1) + ReLU          → 128×16×16
MaxPool(2×2)                                  → 128×8×8
  ↓
Conv2D(128→256, k=3, pad=1) + ReLU          → 256×8×8
MaxPool(2×2)                                  → 256×4×4
  ↓
Linear(256×4×4=4096 → 1024) + ReLU + Dropout(0.5)
Linear(1024 → 500) + Softmax
```
