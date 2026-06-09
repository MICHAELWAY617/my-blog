---
title: '从零构建 CNN：纯 C++ 手写汉字识别框架开发日志 #1'
description: '不用 PyTorch，不用 TensorFlow，从矩阵乘法开始搭建一个完整的卷积神经网络。第一期：Matrix 矩阵库、im2col 卷积加速、Tensor 4D 语义封装。'
pubDate: 'Jun 09 2026'
---

## 动机

市面上有无数「用 PyTorch 训练一个 CNN」的教程。但我想知道：**那层 `nn.Conv2d(1, 32, 3)` 到底做了什么？反向传播的梯度是怎么穿过卷积层的？**

于是决定从零开始，用 C++ 实现一个完整的 CNN，不依赖任何深度学习框架。目标：识别 500 类手写汉字，输入 64×64 灰度图。

这个系列记录开发过程中的每一个设计决策和踩坑记录。代码仓库：[github.com/MICHAELWAY617/my-blog/tree/master/projects/cnn](https://github.com/MICHAELWAY617/my-blog/tree/master/projects/cnn)。

---

## 第一期已完成模块

| 模块 | 文件 | 功能 |
|------|------|------|
| Matrix | `matrix.h/cpp` | 2D 矩阵运算库，本项目基石 |
| im2col / col2im | `matrix.cpp` | 卷积 → 矩阵乘法的降维工具 |
| Tensor | `tensor.h/cpp` | Matrix 的 4D 语义薄封装，零开销 |

---

## 设计决策 1：为什么用 im2col？

卷积本质上是一个滑动窗口操作——一个 K×K 的卷积核在输入图像上逐位置滑动，每个位置做一次逐元素乘加。如果写成嵌套循环，CPU 缓存命中率很差。

**im2col 的核心思路**：把每个滑动窗口覆盖的图像块展开成一个列向量，所有位置拼成一个大矩阵。卷积核也展开成行向量。一次矩阵乘法就完成了所有位置的所有卷积核计算。

```
原始图像 (C, H, W)           im2col 之后 (C*K*K, H_out*W_out)
┌─────────────┐              ┌──────────────────────┐
│ 1  2  3     │              │ 1  2  4  5  ...      │  ← kh=0,kw=0 位置
│ 4  5  6     │  ──────►     │ 2  3  5  6  ...      │  ← kh=0,kw=1 位置
│ 7  8  9     │              │ 4  5  7  8  ...      │  ← kh=1,kw=0 位置
└─────────────┘              │ 5  6  8  9  ...      │  ← kh=1,kw=1 位置
                             └──────────────────────┘
```

**代码（matrix.h:221-223）**：

```cpp
static Matrix im2col(const Matrix& img, int C, int H, int W,
                     int K, int stride, int pad);
```

输出形状 `(C*K*K, H_out*W_out)`，其中 `H_out = (H + 2*pad - K) / stride + 1`。

**反向传播 col2im**：im2col 的逆过程。当 stride < K 时相邻窗口重叠，同一像素会从多个窗口位置接收到梯度，所以 col2im 是**累加**（sum）而非赋值——对应链式法则中多条路径的梯度求和。

在 main.cpp 里用一张 3×3 的简单图像验证了这对函数的正确性：

```cpp
// 构造 3×3 图像，像素值 1~9
Matrix img(1, 9);
for (int i = 0; i < 9; ++i) img.data[i] = i + 1.0;

// im2col: K=2, stride=1, pad=0 → 输出 4×4
Matrix cols = Matrix::im2col(img, /*C=*/1, /*H=*/3, /*W=*/3,
                              /*K=*/2, /*stride=*/1, /*pad=*/0);
// cols.at(0,0)=1, cols.at(0,1)=2, cols.at(0,2)=4, cols.at(0,3)=5
// cols.at(2,0)=4, cols.at(3,3)=9

// col2im 逆操作：重叠区域的像素值会被累加
Matrix grad = Matrix::col2im(cols, 1, 3, 3, 2, 1, 0);
// grad.data[0]=1.0（角上，只出现一次）
// grad.data[4]=20.0（中心，出现在全部 4 个窗口位置，5×4=20）
// grad.data[8]=9.0（角上，只出现一次）
```

**为什么选 im2col？**

| 方案 | 优点 | 缺点 |
|------|------|------|
| 直接嵌套循环 | 无额外内存 | 慢，缓存不友好 |
| **im2col + GEMM** ✅ | 复用高度优化的矩阵乘法；反向传播对称简洁 | 内存膨胀 K² 倍 |
| Winograd / FFT | 大卷积核极快 | 实现复杂 |

对于本项目 K=3 的卷积，内存膨胀仅 9 倍，代码清晰度收益巨大。

---

## 设计决策 2：Tensor — 4D 语义薄封装

CNN 中的数据天然是 4 维：`(Batch, Channel, Height, Width)`。Matrix 只能表达 2D。如果不用 Tensor，你会在代码里看到：

```cpp
// 😱 没有 Tensor 的可怕代码
m.at(b, c * 64 * 64 + h * 64 + w);  // 到处都是这种魔法公式
```

Tensor 在 Matrix 之上加了一层 4D 索引逻辑，但**内部存储完全共用 Matrix 的 `std::vector<double>`**，不复制数据，不额外分配内存。

```cpp
// ✅ 用 Tensor
t.at(n, c, h, w);  // 语义清晰，内部换算：data.at(n, c*H*W + h*W + w)
```

**性能：零开销**。Tensor 的所有方法都是极短的内联函数，编译器会直接把函数体嵌入调用处。运行时成本 = 0。

Tensor 存在的唯一目的是让代码更容易写对。所有重型计算（矩阵乘法、hadamard 积、逐元素操作）仍然由 Matrix 完成，Tensor 只是透明的转发层。

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
Linear(4096 → 1024) + ReLU + Dropout(0.5)
Linear(1024 → 500) + Softmax
```

---

## 后续计划

第一期只完成了基础设施。剩下的 TODO：

- [ ] Conv2D 层（im2col → matmul → reshape）
- [ ] MaxPool2D 层
- [ ] ReLU 激活
- [ ] Linear 全连接层
- [ ] Softmax + CrossEntropy Loss
- [ ] Dropout
- [ ] Sequential 容器
- [ ] SGD Optimizer
- [ ] 训练循环

---

## 开发原则

整个项目的哲学是：**宁可花时间测底层，也不要在上面 debug 地狱**。

每写完一个模块，立即在 main.cpp 里写测试——构造输入、执行操作、自动验证关键值。所有测试通过后，才进入下一个模块。这样当最终的 CNN 训练不收敛时，可以确定每个组件都是正确的，问题一定出在组件的连接方式上。

下一篇开发日志会覆盖 Conv2D 层的实现和反向传播推导。Stay tuned.
