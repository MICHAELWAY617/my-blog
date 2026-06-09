// ============================================================================
// matrix.cpp — Matrix 类的所有方法实现
// ============================================================================
//
// 阅读顺序建议：
//   1. 构造函数（搞懂数据怎么存的）
//   2. at()（搞懂索引怎么算的）
//   3. 基础运算 +, -, *, /
//   4. 特殊运算 hadamard, transpose
//   5. 数学函数 exp, log, sqrt, pow
//   6. 规约 sum, max, argmax
//   7. 形状变换 reshape, row, col, flatten
//   8. im2col / col2im（卷积加速的核心）
//
// 每个方法的实现原则：
//   - 不修改原矩阵（const 方法返回新 Matrix）
//   - 除非是 operator+= 这种明确标了「原地」的
//   - 用一维循环而不是二维（更快、编译器更好优化）
// ============================================================================

#include "matrix.h"
#include <iostream>
#include <random>
#include <algorithm>

// ============================================================================
// 全局随机数生成器 — 整个程序共用一个，避免每次生成随机数都重新初始化
// ============================================================================
//
// std::mt19937 是梅森旋转算法（Mersenne Twister），C++ 标准库的高质量随机数引擎。
// std::random_device{}() 用硬件熵源（比如 CPU 的 RDSEED 指令）生成初始种子。
// static 保证全局只有一个实例，只初始化一次。
//
// 为什么不用 rand()？
//   rand() 质量差、周期短、不同平台行为不一致。C++11 之后推荐用 <random> 库。

static std::mt19937& rng() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

// ============================================================================
// 构造函数
// ============================================================================

// 创建一个 rows×cols 的矩阵，用 init_val 填充所有元素
// 初始化列表 : rows(rows), cols(cols), data(rows * cols, init_val)
//   rows(rows)        → 把成员变量 this->rows 设成参数 rows
//   cols(cols)        → 把 this->cols 设成参数 cols
//   data(rows*cols, init_val) → 创建一个长度=rows*cols 的 vector，全填 init_val
Matrix::Matrix(int rows, int cols, double init_val)
    : rows(rows), cols(cols), data(rows * cols, init_val) {}

// ============================================================================
// 元素访问 at(r, c)
// ============================================================================
//
// 这是 Matrix 类里被调用次数最多的方法，要快。
// 索引公式：data 里第 r 行第 c 列的位置 = r × cols + c
//
// 为什么是 r * cols + c？
//   因为数据按「行优先」排列：
//     第 0 行的 cols 个元素 → 第 1 行的 cols 个元素 → 第 2 行的 cols 个元素 → ...
//   跳过前 r 行（每行有 cols 个）就是 r * cols 个元素，
//   然后在当前行里再跳过 c 个，总共 r * cols + c。

double& Matrix::at(int r, int c) {
    return data[r * cols + c];
}

// 只读版本 — 当 Matrix 被标为 const 时自动用这个
const double& Matrix::at(int r, int c) const {
    return data[r * cols + c];
}

// ============================================================================
// 加法：矩阵 + 矩阵
// ============================================================================
//
// 要求：两个矩阵形状必须相同（rows 和 cols 都一样），这里不做检查，调用者自己保证。
// 对应位置相加：result[r][c] = this[r][c] + other[r][c]
//
// 为什么用一维循环 for (int k=0; k<rows*cols; ++k) 而不是二维嵌套？
//   data 是连续内存。一维循环：
//     1. 代码更短
//     2. 编译器更容易做向量化优化（SIMD，一次处理 2~4 个 double）
//     3. CPU 缓存预取更准确
Matrix Matrix::operator+(const Matrix& other) const {
    Matrix result(rows, cols);
    for (int k = 0; k < rows * cols; ++k)
        result.data[k] = data[k] + other.data[k];
    return result;
}

// 减法：跟加法完全对称
Matrix Matrix::operator-(const Matrix& other) const {
    Matrix result(rows, cols);
    for (int k = 0; k < rows * cols; ++k)
        result.data[k] = data[k] - other.data[k];
    return result;
}

// 数乘：矩阵 × 标量
// 每个元素都乘以同一个数，比如所有权重统一缩放
Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (int k = 0; k < rows * cols; ++k)
        result.data[k] = data[k] * scalar;
    return result;
}

// ============================================================================
// 矩阵乘法：矩阵 × 矩阵（这是线性代数里的矩阵乘，不是逐元素乘！）
// ============================================================================
//
// 定义：A(rows, cols) × B(cols, other.cols) = C(rows, other.cols)
//       C(i,j) = Σ(k=0..cols-1) A(i,k) × B(k,j)
//
// 这是整个 CNN 前向/反向传播的核心操作，也是最费时的操作。
//
// 三重循环的意义：
//   外两层 (i, j)：遍历输出矩阵 C 的每个位置
//   最内层 (k)：  计算 A 的第 i 行 与 B 的第 j 列 的内积
//
// 复杂度：O(rows × other.cols × cols)，对于大矩阵这就是性能瓶颈。
//   CNN 里 90% 的计算量都在矩阵乘法上，这也是为什么 GPU 做深度学习这么快——
//   GPU 有几千个核同时做内积。
//
// 例子（来自 main.cpp 的测试）：
//   A(2,3) = [[1,2,3], [4,5,6]]
//   B(3,2) = [[7,8], [9,10], [11,12]]
//   C(0,0) = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = 58
//   C(0,1) = 1×8 + 2×10 + 3×12 = 8 + 20 + 36 = 64
//   C(1,0) = 4×7 + 5×9 + 6×11 = 28 + 45 + 66 = 139
//   C(1,1) = 4×8 + 5×10 + 6×12 = 32 + 50 + 72 = 154
Matrix Matrix::operator*(const Matrix& other) const {
    // 结果矩阵：行数 = A的rows，列数 = B的cols
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; i++) {           // A 的每一行
        for (int j = 0; j < other.cols; j++) { // B 的每一列
            double sum = 0.0;
            for (int k = 0; k < cols; k++)     // 内积：A行 × B列
                sum += at(i, k) * other.at(k, j);
            result.at(i, j) = sum;
        }
    }
    return result;
}

// 数除：每个元素除以标量
Matrix Matrix::operator/(double scalar) const {
    Matrix result(rows, cols);
    for (int k = 0; k < rows * cols; ++k)
        result.data[k] = data[k] / scalar;
    return result;
}

// ============================================================================
// 原地运算 — 不产生新矩阵，直接修改自己
// 用于参数更新等场景，可以省内存（不用反复分配新矩阵）
// ============================================================================

Matrix& Matrix::operator+=(const Matrix& other) {
    for (int k = 0; k < rows * cols; ++k)
        data[k] += other.data[k];
    return *this;   // 返回 *this 支持链式调用：m1 += m2 += m3;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    for (int k = 0; k < rows * cols; ++k)
        data[k] -= other.data[k];
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (int k = 0; k < rows * cols; ++k)
        data[k] *= scalar;
    return *this;
}

// ============================================================================
// 相等比较
// ============================================================================
// 先比形状，形状不同直接 false。形状相同再逐元素比较。
//
// 注意：浮点数直接用 == 有精度风险（0.1+0.2 != 0.3 在浮点世界里是日常）。
// 生产代码应该用近似比较（见 main.cpp 的 approx 函数）。
bool Matrix::operator==(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) return false;
    for (int k = 0; k < rows * cols; ++k)
        if (data[k] != other.data[k]) return false;
    return true;
}

// ============================================================================
// Hadamard 积 — 对应位置直接相乘，形状不变
// ============================================================================
//
// 这个很简单：result[k] = a[k] * b[k]
//
// 在 CNN 里的用途：
//   - 注意力机制里的权重 × 值
//   - Dropout 反向传播（梯度 × dropout mask）
//   - LSTM/GRU 的各种门
Matrix Matrix::hadamard(const Matrix& other) const {
    Matrix result(rows, cols);
    for (int k = 0; k < rows * cols; ++k)
        result.data[k] = data[k] * other.data[k];
    return result;
}

// ============================================================================
// 转置 — 行变列、列变行
// ============================================================================
//
// 转置前后形状变化：(rows, cols) → (cols, rows)
// 元素映射：result(j,i) = original(i,j)
//
// 在 CNN 反向传播里大量使用：
//   前向：x @ W     (输入 × 权重)
//   反向：grad @ Wᵀ (梯度 × 权重的转置)
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);              // 注意：行数变列数
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.at(j, i) = at(i, j);     // 行列互换
    return result;
}

// ============================================================================
// fill / randomize — 修改矩阵内容
// ============================================================================

// 所有元素填成同一个值
void Matrix::fill(double val) {
    std::fill(data.begin(), data.end(), val);
}

// 用均匀分布随机数填充，范围 [min, max]
// std::uniform_real_distribution 保证区间内每个值被抽到的概率相等
void Matrix::randomize(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    for (int k = 0; k < rows * cols; ++k)
        data[k] = dist(rng());
}

int Matrix::numel() const {
    return rows * cols;
}

// ============================================================================
// 逐元素数学函数
// ============================================================================
// 对矩阵里每个元素独立应用 std 数学函数，返回新矩阵。
// 这些在 CNN 的激活函数和损失函数里用到：
//   exp  → Softmax、Sigmoid
//   log  → 交叉熵损失
//   sqrt → BatchNorm、RMSprop
//   pow  → 各种自定义激活函数

Matrix Matrix::exp() const {
    Matrix result(rows, cols);
    for (int k = 0; k < rows * cols; ++k)
        result.data[k] = std::exp(data[k]);  // e^x
    return result;
}

Matrix Matrix::log() const {
    Matrix result(rows, cols);
    for (int k = 0; k < rows * cols; ++k)
        result.data[k] = std::log(data[k]);  // ln(x)，自然对数
    return result;
}

Matrix Matrix::sqrt() const {
    Matrix result(rows, cols);
    for (int k = 0; k < rows * cols; ++k)
        result.data[k] = std::sqrt(data[k]);
    return result;
}

Matrix Matrix::pow(double exponent) const {
    Matrix result(rows, cols);
    for (int k = 0; k < rows * cols; ++k)
        result.data[k] = std::pow(data[k], exponent);
    return result;
}

// ============================================================================
// 规约操作 — 把矩阵压缩成标量
// ============================================================================

// 求和：Σ 所有元素
double Matrix::sum() const {
    double s = 0.0;
    for (int k = 0; k < rows * cols; ++k)
        s += data[k];
    return s;
}

// 最大值：遍历找最大
double Matrix::max() const {
    double m = data[0];
    for (int k = 1; k < rows * cols; ++k)
        if (data[k] > m) m = data[k];
    return m;
}

// 最大值的位置（从 0 开始编号，按行优先）
// 分类任务里用来确定预测类别：argmax(softmax输出) = 模型认为最可能的那个类
int Matrix::argmax() const {
    int idx = 0;
    for (int k = 1; k < rows * cols; ++k)
        if (data[k] > data[idx]) idx = k;
    return idx;
}

// ============================================================================
// 形状变换
// ============================================================================

// reshape：换个形状但不改变数据。注意元素总数必须一致！
// 实现很取巧：直接把 data 拷过去，然后修改 rows/cols 标记。
// 因为 data 是 flat vector，只要元素总数对，换个 shape 就是换种解读方式。
Matrix Matrix::reshape(int new_rows, int new_cols) const {
    Matrix result(new_rows, new_cols);
    result.data = data;    // 浅拷贝（vector 的 = 会复制所有元素）
    return result;
}

// 取一行：返回一个 (1, cols) 的行向量
Matrix Matrix::row(int r) const {
    Matrix result(1, cols);
    for (int j = 0; j < cols; j++)
        result.at(0, j) = at(r, j);
    return result;
}

// 取一列：返回一个 (rows, 1) 的列向量
Matrix Matrix::col(int c) const {
    Matrix result(rows, 1);
    for (int i = 0; i < rows; i++)
        result.at(i, 0) = at(i, c);
    return result;
}

// flatten：展平成一行
// CNN 里卷积层的输出是 4D 特征图 (N,C,H,W)，进全连接层之前要 flatten 成 (N, C*H*W)
Matrix Matrix::flatten() const {
    Matrix result(1, rows * cols);
    result.data = data;
    return result;
}

// 打印矩阵内容
void Matrix::print() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            std::cout << at(i, j) << " ";
        std::cout << std::endl;
    }
}

// ============================================================================
// 静态工厂方法
// ============================================================================
// 为什么需要这些？
//   Matrix::zeros(3,4) 比 Matrix(3,4,0.0) 更可读，一眼看出意图。

Matrix Matrix::zeros(int rows, int cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(int rows, int cols) {
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::random(int rows, int cols, double min, double max) {
    Matrix result(rows, cols);
    result.randomize(min, max);
    return result;
}

// ============================================================================
// im2col — «卷积 → 矩阵乘法» 的关键转换
// ============================================================================
//
// 【直观理解】
// 假设一张 1 通道的 3×3 图像，用 2×2 的卷积核，stride=1，不 padding：
//
//   图像: 1  2  3        卷积核: a  b
//        4  5  6                c  d
//        7  8  9
//
// 卷积操作：把卷积核盖在图像上，对应位置乘加。核从左上滑到右下：
//   位置(0,0): a*1 + b*2 + c*4 + d*5
//   位置(0,1): a*2 + b*3 + c*5 + d*6
//   位置(1,0): a*4 + b*5 + c*7 + d*8
//   位置(1,1): a*5 + b*6 + c*8 + d*9
//
// 这 4 个位置的计算，本质上都是「核的 4 个元素」与「图像上 4 个像素」做内积。
// im2col 把图像上每个位置的这 4 个像素拉成一列：
//
//   输入图像 (1,9)           im2col 输出 (4,4):
//   [1,2,3,4,5,6,7,8,9]  →   行0 (kh=0,kw=0): [1, 2, 4, 5]  ← 4 个位置的对应元素
//                              行1 (kh=0,kw=1): [2, 3, 5, 6]
//                              行2 (kh=1,kw=0): [4, 5, 7, 8]
//                              行3 (kh=1,kw=1): [5, 6, 8, 9]
//
//   然后把卷积核也展成行向量 [a,b,c,d]
//   一次矩阵乘法 [a,b,c,d] × im2col矩阵 = 4 个位置的卷积结果！
//
// 【变量命名约定】
//   C, H, W       → 输入图像的通道数、高度、宽度
//   K              → 卷积核大小（边长）
//   H_out, W_out   → 输出特征图的高度、宽度
//   K_flat         → C*K*K，im2col 输出矩阵的行数
//   N_cols         → H_out*W_out，im2col 输出矩阵的列数
//   oh, ow         → output 空间坐标（h_out, w_out 的缩写）
//   ih, iw         → input  空间坐标（映射回原图的坐标）
//   kh, kw         → kernel 空间坐标（卷积核内的偏移）

Matrix Matrix::im2col(const Matrix& img, int C, int H, int W,
                      int K, int stride, int pad) {
    // 第一步：计算输出尺寸
    // 公式：out = (in + 2*pad - K) / stride + 1
    // 例如 H=64, K=3, pad=1, stride=1 → H_out = (64+2-3)/1+1 = 64（输出尺寸不变！）
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;
    int K_flat = C * K * K;                // 每个滑动窗口包含的元素数
    int N_cols = H_out * W_out;            // 滑动窗口的总位置数

    // 创建结果矩阵：(K_flat 行) × (N_cols 列)，初始化为 0
    // 每行 = 核的一个固定偏移位置在所有滑动位置上的取值
    // 每列 = 一个滑动位置上核的所有偏移取值
    Matrix result(K_flat, N_cols, 0.0);

    // img 是按 CHW 顺序展平的行向量：
    //   img.data[ c*H*W + h*W + w ] = 第 c 通道、第 h 行、第 w 列 的像素值
    //
    // 因为 img 是 (1, C*H*W) 的行向量，只有一行，
    // 所以 img.data[i] = img.at(0, i)

    // 第二、三、四层循环遍历卷积核的每个元素：通道 c，核内行 kh，核内列 kw
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                // 核内位置 (c, kh, kw) 对应输出矩阵的哪一行
                // c*K*K: 跳过前面所有通道的 K×K 个元素
                // kh*K:  跳过当前通道里上面 kh 行的 K 个元素
                // kw:    当前行里的第 kw 个
                int row = c * K * K + kh * K + kw;

                // 最内两层循环遍历输出空间的所有位置
                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        // 当前输出位置 (oh, ow) 对应输出矩阵的哪一列
                        int col = oh * W_out + ow;

                        // 把输出位置映射回输入图像上的坐标
                        // stride*oh：输出上移动 oh 步，输入上就要移动 stride*oh 步
                        // kh - pad：加上核内偏移，减去 padding 偏移
                        int ih = oh * stride + kh - pad;
                        int iw = ow * stride + kw - pad;

                        // 边界检查：映射回的坐标必须在图像范围内
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            // 从输入图像取像素值，放到列矩阵的对应位置
                            result.at(row, col) = img.data[c * H * W + ih * W + iw];
                        }
                        // 如果坐标越界（在 padding 区域），值就是初始化的 0.0
                        // 这正是 "zero-padding" 的含义 —— 超出边界的值视为 0
                    }
                }
            }
        }
    }

    return result;
}

// ============================================================================
// col2im — im2col 的逆操作（反向传播专用）
// ============================================================================
//
// 卷积反向传播里，损失对卷积输出的梯度形状是 (C*K*K, H_out*W_out)，
// 需要变回 (1, C*H*W) 才能继续往上一层传。
//
// col2im 跟 im2col 做的事情一样，只是方向反过来：
//   im2col: 图像上的值 → 复制到列矩阵
//   col2im: 列矩阵的值 → 累加回图像（注意是 += 不是 =！）
//
// 为什么要累加（+=）而不是赋值（=）？
//   当 stride < K 时，相邻窗口会重叠。比如 K=3, stride=1：
//   输入图像的中心像素会出现在 9 个不同的滑动窗口里（上中下 × 左中右 = 3×3）。
//   反向传播时，这 9 个窗口的梯度都要「归咎」于这个像素，
//   根据链式法则，这 9 条路径的梯度要加起来。这就是 col2im 用 += 的原因。
//
//   举例（来自 main.cpp 的测试）：
//     3×3 图像，K=2, stride=1：
//       角上的像素 1 只出现在 1 个窗口 → col2im 后值不变 = 1
//       中心的像素 5 出现在 4 个窗口   → col2im 后值累加 = 20

Matrix Matrix::col2im(const Matrix& cols, int C, int H, int W,
                      int K, int stride, int pad) {
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    // 结果是一个行向量 (1, C*H*W)，也就是图像梯度的展平形式
    // 初始化为全 0，后面用 += 逐步累加
    Matrix result(1, C * H * W, 0.0);

    // 结构跟 im2col 完全对称，只是方向反过来
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int row = c * K * K + kh * K + kw;   // 列矩阵的行

                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        int col = oh * W_out + ow;     // 列矩阵的列
                        int ih = oh * stride + kh - pad;
                        int iw = ow * stride + kw - pad;

                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            // ★ 这里是 += 不是 = ★
                            // 多个窗口位置的梯度累加到同一个像素上
                            result.data[c * H * W + ih * W + iw] += cols.at(row, col);
                        }
                        // 越界的就不用管了——padding 区域没有对应的输入像素
                    }
                }
            }
        }
    }

    return result;
}
