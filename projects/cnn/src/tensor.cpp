// ============================================================================
// tensor.cpp — Tensor 类的所有方法实现
// ============================================================================
//
// 阅读前提：先搞懂 matrix.h 和 matrix.cpp，理解数据怎么存的。
// Tensor 非常薄——几乎所有操作都是直接委托给底层 Matrix。
//
// Tensor 的代码总量很少，因为它只做两件事：
//   1. 维护 N,C,H,W 形状信息
//   2. 把 4D 索引换算成 2D 索引，然后调用 Matrix 的方法
// ============================================================================

#include "tensor.h"
#include <stdexcept>
#include <sstream>

// ============================================================================
// 构造函数
// ============================================================================

// 构造函数 1：从形状 + 初始值创建
//
// 内部做的事：
//   Matrix(n, c*h*w, init_val)  — 创建一个 n 行、(c*h*w) 列的矩阵
//   每行 = 一个样本的全部像素展平
//   所有元素初始化为 init_val
//
// 例子：
//   Tensor t(2, 3, 4, 5, 0.0);
//   → 内部 data = Matrix(rows=2, cols=60)，全填 0.0
//   → N=2, C=3, H=4, W=5
Tensor::Tensor(int n, int c, int h, int w, double init_val)
    : data(Matrix(n, c * h * w, init_val)), N(n), C(c), H(h), W(w) {}

// 构造函数 2：包装一个已有的 Matrix
//
// 用途：各层（Conv2D、Linear 等）内部用 Matrix 做运算，算完后包一层
// Tensor 语义再传给下一层。不需要复制数据，只是加上 4D 标签。
//
// 安全检查：Matrix 的行数必须等于 N，列数必须等于 C*H*W。
// 不匹配就抛异常——这说明调用者的形状对不上，是个 bug。
Tensor::Tensor(const Matrix& m, int n, int c, int h, int w)
    : data(m), N(n), C(c), H(h), W(w) {
    if (m.rows != n || m.cols != c * h * w) {
        std::ostringstream oss;
        oss << "Tensor: Matrix shape (" << m.rows << "," << m.cols
            << ") does not match Tensor shape ("
            << n << "," << c << "," << h << "," << w << ")";
        throw std::invalid_argument(oss.str());
    }
}

// ============================================================================
// 4D 元素访问
// ============================================================================
//
// 这是 Tensor 最重要的方法——把 4D 下标映射到 2D 下标。
//
// 【地址换算详解】
//   at(n, c, h, w) → data.at(n, c*H*W + h*W + w)
//
//   第一维 n（batch）：
//     对应 Matrix 的行号。第 n 个样本就在第 n 行。
//
//   后三维 (c, h, w)：
//     对应 Matrix 的列号。三个坐标必须展平成一个整数。
//     因为 CHW 排列顺序是「通道优先」：
//       - 先遍历所有通道（外层）
//       - 每个通道内遍历所有行（中层）
//       - 每行内遍历所有列（内层）
//     所以：
//       c*H*W        → 跳到第 c 个通道的起始位置（跳过 c×H×W 个元素）
//          h*W       → 在当前通道里跳到第 h 行的起始位置（跳过 h×W 个元素）
//             w      → 在当前行里跳到第 w 列
//     三者加起来就是最终偏移。
//
// 例子：Tensor(N=1, C=2, H=3, W=4)
//   - at(0, 0, 0, 0) → data.at(0, 0)          — 第 0 通道第 (0,0) 像素
//   - at(0, 0, 2, 3) → data.at(0, 0+2*4+3) = data.at(0, 11) — 第 0 通道第 (2,3)
//   - at(0, 1, 0, 0) → data.at(0, 1*12+0+0) = data.at(0, 12) — 第 1 通道第 (0,0)
//   可以看到：通道 0 占 0~11，通道 1 占 12~23，紧密排列。

double& Tensor::at(int n, int c, int h, int w) {
    return data.at(n, c * H * W + h * W + w);
}

const double& Tensor::at(int n, int c, int h, int w) const {
    return data.at(n, c * H * W + h * W + w);
}

// ============================================================================
// reshape — 换形状，总数不变
// ============================================================================
//
// 实现很取巧：把 Matrix 的 data 拷过来，然后直接改 rows/cols 标签。
// 因为 data 就是一段连续内存，只要总元素数对，形状只是「解读方式」。
//
// CNN 里的经典用法：
//   卷积层输出: Tensor(N=16, C=256, H=4, W=4)
//   → reshape(N=16, C=1, H=1, W=4096)
//   → 送入全连接层 Linear(4096 → 1024)
Tensor Tensor::reshape(int n, int c, int h, int w) const {
    if (n * c * h * w != total_elements())
        throw std::invalid_argument("Tensor::reshape: total elements must stay the same");

    // 浅拷贝 Matrix 的内容（data 是 vector，= 会复制元素）
    Matrix m = data;
    // 重新标记形状
    m.rows = n;
    m.cols = c * h * w;
    return Tensor(m, n, c, h, w);
}

// ============================================================================
// 逐元素运算 — 全部直接委托给 Matrix
// ============================================================================
//
// 这些方法本身什么都不算，只是把活交给 data（Matrix），
// 再把算完的 Matrix 包成 Tensor 返回。
// 因为方法极短，编译器会把它们「内联」掉——运行时跟直接写 Matrix 运算一模一样。

Tensor Tensor::operator+(const Tensor& other) const {
    return Tensor(data + other.data, N, C, H, W);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return Tensor(data - other.data, N, C, H, W);
}

Tensor Tensor::operator*(double scalar) const {
    return Tensor(data * scalar, N, C, H, W);
}

Tensor Tensor::hadamard(const Tensor& other) const {
    return Tensor(data.hadamard(other.data), N, C, H, W);
}

Tensor& Tensor::operator+=(const Tensor& other) {
    data += other.data;   // 委托给 Matrix::operator+=
    return *this;
}

Tensor& Tensor::operator*=(double scalar) {
    data *= scalar;       // 委托给 Matrix::operator*=
    return *this;
}

// ============================================================================
// fill / randomize
// ============================================================================

void Tensor::fill(double val) {
    data.fill(val);
}

void Tensor::randomize(double min, double max) {
    data.randomize(min, max);
}

// ============================================================================
// print — 按层次结构打印
// ============================================================================
//
// 输出格式：
//   Tensor(N,C,H,W)          ← 第一行永远是形状标签
//   batch 0:                 ← 如果 N>1，给每个 batch 加标签
//     channel 0:             ← 如果 C>1，给每个 channel 加标签
//       1.0 2.0 3.0          ← 第 0 行
//       4.0 5.0 6.0          ← 第 1 行
//
// 单样本单通道时不显示 batch/channel 标签，保持简洁。

void Tensor::print() const {
    std::cout << "Tensor(" << N << "," << C << "," << H << "," << W << ")" << std::endl;
    for (int n = 0; n < N; ++n) {
        if (N > 1) std::cout << "batch " << n << ":" << std::endl;
        for (int c = 0; c < C; ++c) {
            if (C > 1) std::cout << "  channel " << c << ":" << std::endl;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    std::cout << at(n, c, h, w) << " ";
                }
                std::cout << std::endl;
            }
        }
    }
}

// ============================================================================
// 静态工厂
// ============================================================================

Tensor Tensor::zeros(int n, int c, int h, int w) {
    return Tensor(n, c, h, w, 0.0);
}

Tensor Tensor::ones(int n, int c, int h, int w) {
    return Tensor(n, c, h, w, 1.0);
}

Tensor Tensor::random(int n, int c, int h, int w, double min, double max) {
    Tensor t(n, c, h, w);
    t.randomize(min, max);
    return t;
}
