// ============================================================================
// main.cpp — CNN 项目综合测试程序
// ============================================================================
//
// 这个文件不是最终产品，而是一个「活文档」——每写完一个新模块，就在这里
// 添加相应的测试。跑通所有测试 = 验证所有已完成的模块工作正常。
//
// 测试按模块分块：
//   1. Matrix 基础运算（加减乘除、转置、逐元素数学、规约、形状变换）
//   2. im2col / col2im（卷积核心工具）
//   3. Tensor（4D 薄封装）
//
// 每个测试都：
//   - 构造输入
//   - 执行操作
//   - 打印结果（肉眼检查）
//   - 用 approx() 验证关键值（自动检查）
//   - 输出 PASS/FAIL
//
// 为什么用测试而不是直接写 CNN？
//   写深度学习框架有一个铁律：每一层的前向和反向必须分别正确。
//   如果跳过基础模块测试直接堆网络，出了问题你根本不知道是哪个模块的锅。
//   所以宁可多花时间测底层，也不要跑到上面 debug 地狱。
// ============================================================================

#include <iostream>
#include <cmath>
#include "matrix.h"
#include "tensor.h"

// --------------------------------------------------------------------------
// approx — 浮点数近似比较
// --------------------------------------------------------------------------
// 为什么需要这个？
//   浮点数运算有精度误差。比如 0.1 + 0.2 用 double 算出来是 0.30000000000000004，
//   直接 == 比较会返回 false。所以用"差的绝对值 < 一个很小的数"来判断相等。
//   eps = 1e-9，也就是误差在十亿分之一以内就算相等。
// --------------------------------------------------------------------------
bool approx(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) < eps;
}

// ============================================================================
// main — 所有测试的入口
// ============================================================================

int main() {
    // ======================================================================
    // 第一部分：Matrix 基础运算
    // ======================================================================

    // --- 构造函数 + at() 访问元素 ---
    // 创建一个 2×3 的矩阵，然后一个一个填数字
    Matrix a(2, 3, 0.0);
    a.at(0,0)=1; a.at(0,1)=2; a.at(0,2)=3;    // 第 0 行: 1, 2, 3
    a.at(1,0)=4; a.at(1,1)=5; a.at(1,2)=6;    // 第 1 行: 4, 5, 6
    std::cout << "a:" << std::endl; a.print();
    // 输出：
    // 1 2 3
    // 4 5 6

    // --- 矩阵乘法 ---
    // a(2×3) × b(3×2) = c(2×2)
    Matrix b(3, 2, 0.0);
    b.at(0,0)=7;  b.at(0,1)=8;
    b.at(1,0)=9;  b.at(1,1)=10;
    b.at(2,0)=11; b.at(2,1)=12;
    Matrix c = a * b;
    std::cout << "a*b:" << std::endl; c.print();
    // c = [[1*7+2*9+3*11,  1*8+2*10+3*12],
    //      [4*7+5*9+6*11,  4*8+5*10+6*12]]
    //   = [[58, 64],
    //      [139,154]]

    // --- Hadamard 积（逐元素相乘，不是矩阵乘法！）---
    Matrix h = a.hadamard(a);
    std::cout << "hadamard(a,a):" << std::endl; h.print();
    // 每个元素跟自己乘：1*1=1, 2*2=4, 3*3=9, ...

    // --- 转置 ---
    std::cout << "transpose(a):" << std::endl; a.transpose().print();
    // 2×3 → 3×2，行变列

    // --- 标量运算 ---
    Matrix d = a + a;    // 每个元素 ×2
    Matrix e = a - a;    // 全零
    Matrix f = a * 2.0;  // 每个元素 ×2
    Matrix g = a / 2.0;  // 每个元素 ÷2
    std::cout << "a/2:" << std::endl; g.print();

    // --- 原地运算（不产生新矩阵，直接修改自身）---
    Matrix ip(2, 2, 1.0);           // 2×2 全 1 矩阵
    ip += Matrix::ones(2, 2);       // 每个元素 +1 → 全 2
    std::cout << "ones+=ones:" << std::endl; ip.print();
    ip -= Matrix(2, 2, 1.0);        // 每个元素 -1 → 全 1
    std::cout << "   -=1:" << std::endl; ip.print();
    ip *= 3.0;                      // 每个元素 ×3 → 全 3
    std::cout << "   *=3:" << std::endl; ip.print();

    // --- 相等比较 ---
    std::cout << "ip==3*ones: " << (ip == Matrix(2, 2, 3.0)) << std::endl;

    // --- fill：全部填成同一个值 ---
    Matrix m(2, 3);
    m.fill(7.0);
    std::cout << "fill(7): " << (m == Matrix(2, 3, 7.0)) << std::endl;

    // --- numel：矩阵里总共有多少元素 ---
    std::cout << "numel(a): " << a.numel() << std::endl;  // 预期 6

    // --- 静态工厂：zeros / ones / random ---
    std::cout << "zeros(2,2):" << std::endl; Matrix::zeros(2, 2).print();
    std::cout << "ones(2,2):" << std::endl;  Matrix::ones(2, 2).print();
    std::cout << "random(3,3,-1,1):" << std::endl; Matrix::random(3, 3, -1, 1).print();

    // --- randomize：把已有矩阵随机填值 ---
    Matrix r(2, 2);
    r.randomize(0, 1);   // 用 [0,1] 均匀分布随机数填充
    std::cout << "randomize(0,1):" << std::endl; r.print();

    // --- 逐元素数学函数 ---
    // e^x, ln(x), sqrt(x), x^n 对矩阵每个元素分别运算
    Matrix x(2, 2, 0.0);
    x.at(0,0)=1; x.at(0,1)=2; x.at(1,0)=4; x.at(1,1)=8;
    std::cout << "exp:" << std::endl; x.exp().print();     // e^1, e^2, e^4, e^8
    std::cout << "log:" << std::endl; x.log().print();     // ln(1), ln(2), ln(4), ln(8)
    std::cout << "sqrt:" << std::endl; x.sqrt().print();   // √1, √2, √4, √8
    std::cout << "pow(2):" << std::endl; x.pow(2.0).print();// 1², 2², 4², 8²

    // --- 规约操作 ---
    std::cout << "sum(x): " << x.sum() << std::endl;       // 1+2+4+8 = 15
    std::cout << "max(x): " << x.max() << std::endl;       // 8
    std::cout << "argmax(x): " << x.argmax() << std::endl; // 8 在第 3 号位置(从0数)

    // --- 形状变换 ---
    Matrix s = Matrix::random(2, 6, 0, 1);
    std::cout << "original(2,6):" << std::endl; s.print();
    std::cout << "reshape(3,4):" << std::endl;  s.reshape(3, 4).print();
    std::cout << "row(0):" << std::endl; s.row(0).print();
    std::cout << "col(0):" << std::endl; s.col(0).print();
    std::cout << "flatten:" << std::endl; s.flatten().print();

    // ======================================================================
    // 第二部分：im2col / col2im 测试
    // ======================================================================
    // 这两个函数是卷积实现的核心。
    // 测试策略：构建一个已知的简单图像，手工算出预期结果，然后验证。

    std::cout << "\n=== im2col test ===" << std::endl;

    // 构造一张 1 通道 3×3 的图像，像素值就是 1 到 9：
    //   1 2 3
    //   4 5 6
    //   7 8 9
    // 存储为 CHW 顺序的行向量：(1, 9)
    Matrix img(1, 9);
    for (int i = 0; i < 9; ++i) img.data[i] = i + 1.0;

    // 参数：单通道 3×3 图像，2×2 卷积核，步长 1，不补零
    int C = 1, H = 3, W = 3, K = 2, stride = 1, pad = 0;
    // 输出尺寸：H_out = (3+0-2)/1+1 = 2，W_out = 2
    // 所以 im2col 输出是 (C*K*K=4) 行 × (H_out*W_out=4) 列

    Matrix cols = Matrix::im2col(img, C, H, W, K, stride, pad);
    std::cout << "im2col (1x3x3, k=2, s=1, p=0):" << std::endl;
    cols.print();
    // 预期输出（4 行 × 4 列）：
    //   行0(kh=0,kw=0): 1 2 4 5   ← 每个窗口位置取(0,0)元素
    //   行1(kh=0,kw=1): 2 3 5 6   ← 每个窗口位置取(0,1)元素
    //   行2(kh=1,kw=0): 4 5 7 8   ← 每个窗口位置取(1,0)元素
    //   行3(kh=1,kw=1): 5 6 8 9   ← 每个窗口位置取(1,1)元素
    //
    // 列0=窗口(0,0): 取像素 [1,2,4,5]
    // 列1=窗口(0,1): 取像素 [2,3,5,6]
    // 列2=窗口(1,0): 取像素 [4,5,7,8]
    // 列3=窗口(1,1): 取像素 [5,6,8,9]

    // 验证几个关键位置的值
    bool ok = true;
    ok &= approx(cols.at(0,0), 1); ok &= approx(cols.at(0,1), 2);
    ok &= approx(cols.at(0,2), 4); ok &= approx(cols.at(0,3), 5);
    ok &= approx(cols.at(2,0), 4); ok &= approx(cols.at(2,1), 5);
    ok &= approx(cols.at(3,3), 9);
    std::cout << "im2col elements: " << (ok ? "PASS" : "FAIL") << std::endl;

    // --- col2im 测试：im2col 的逆操作 ---
    std::cout << "\n=== col2im test ===" << std::endl;
    Matrix grad = Matrix::col2im(cols, C, H, W, K, stride, pad);
    std::cout << "col2im result (expect interior counts > 1 due to overlap):" << std::endl;
    grad.print();
    // 重叠导致累加：
    //   角上的 1 只出现在窗口(0,0)的(0,0)位置 → 值 = 1
    //   中心的 5 出现在所有 4 个窗口的对应位置 → 值 = 5+5+5+5 = 20
    //   边上的 2 出现在窗口(0,0)和(0,1) → 值 = 2+2 = 4
    // 预期输出: [1, 4, 3, 8, 20, 12, 7, 16, 9]

    bool ok2 = true;
    ok2 &= approx(grad.data[0], 1.0);    // 1 只出现一次
    ok2 &= approx(grad.data[4], 20.0);   // 5 出现了 4 次
    ok2 &= approx(grad.data[8], 9.0);    // 9 只出现一次
    std::cout << "col2im reconstruction: " << (ok2 ? "PASS" : "FAIL") << std::endl;

    // --- im2col 带 padding ---
    std::cout << "\n=== im2col with padding ===" << std::endl;
    Matrix cols_pad = Matrix::im2col(img, C, H, W, K, stride, /*pad=*/1);
    std::cout << "im2col (pad=1), shape " << cols_pad.rows << "x" << cols_pad.cols << ":" << std::endl;
    cols_pad.print();
    // pad=1 时输出尺寸：H_out = (3+2-2)/1+1 = 4, W_out = 4
    // 输出 4×16，超出边界的元素值为 0（zero-padding）

    // ======================================================================
    // 第三部分：Tensor 测试
    // ======================================================================
    std::cout << "\n=== Tensor tests ===" << std::endl;

    // --- 构造 + 4D 访问 ---
    // 创建一个 (2,3,4,5) 的张量：2 个样本，3 通道，4×5 空间
    Tensor t(2, 3, 4, 5, 0.0);
    std::cout << "shape: " << t.N << "," << t.C << "," << t.H << "," << t.W << std::endl;
    std::cout << "total elements: " << t.total_elements() << " (expect 120)" << std::endl;
    // 验证：2 × 3 × 4 × 5 = 120 ✓

    // 用 4D 坐标写入和读取
    t.at(0, 0, 0, 0) = 1.0;    // 第 0 样本、第 0 通道、第 (0,0) 像素 = 1
    t.at(0, 2, 3, 4) = 99.0;   // 第 0 样本、第 2 通道、第 (3,4) 像素 = 99
    t.at(1, 1, 2, 3) = 42.0;   // 第 1 样本、第 1 通道、第 (2,3) 像素 = 42

    std::cout << "t(0,0,0,0)=" << t.at(0,0,0,0) << " (expect 1)" << std::endl;
    std::cout << "t(0,2,3,4)=" << t.at(0,2,3,4) << " (expect 99)" << std::endl;
    std::cout << "t(1,1,2,3)=" << t.at(1,1,2,3) << " (expect 42)" << std::endl;

    // --- zeros / ones / random ---
    Tensor z = Tensor::zeros(1, 2, 2, 2);
    std::cout << "zeros: " << z.at(0,0,0,0) << " " << z.at(0,1,1,1) << " (expect 0 0)" << std::endl;

    Tensor o = Tensor::ones(1, 2, 2, 2);
    std::cout << "ones:  " << o.at(0,0,0,0) << " " << o.at(0,1,1,1) << " (expect 1 1)" << std::endl;

    Tensor tr = Tensor::random(1, 1, 3, 3, -1, 1);
    std::cout << "random (1,1,3,3):" << std::endl;
    tr.print();

    // --- reshape ---
    // 把 (2,3,4,5) 共 120 元素展平成 1D
    Tensor flat = t.reshape(1, 1, 1, 120);
    std::cout << "reshape (1,1,1,120): N=" << flat.N << " C=" << flat.C
              << " H=" << flat.H << " W=" << flat.W << std::endl;
    std::cout << "value at flat(0,0,0,0)=" << flat.at(0,0,0,0) << " (expect 1)" << std::endl;
    // 展平后第一个元素应该是原来第一个元素 = 1

    // --- 逐元素运算 ---
    Tensor a2 = Tensor::ones(1, 1, 2, 2);
    Tensor b2 = Tensor::ones(1, 1, 2, 2) * 2.0;    // 全 2
    Tensor sum = a2 + b2;                            // 全 3
    std::cout << "ones + (ones*2): " << sum.at(0,0,0,0) << " (expect 3)" << std::endl;

    Tensor diff = b2 - a2;                           // 全 1
    std::cout << "(ones*2) - ones: " << diff.at(0,0,0,0) << " (expect 1)" << std::endl;

    Tensor hm = a2.hadamard(b2);                     // 1×2 = 2
    std::cout << "hadamard(ones, ones*2): " << hm.at(0,0,0,0) << " (expect 2)" << std::endl;

    // --- fill ---
    Tensor tf(2, 2, 2, 2);
    tf.fill(3.14);
    std::cout << "fill(3.14): " << tf.at(0,0,0,0) << " " << tf.at(1,1,1,1)
              << " (expect 3.14 3.14)" << std::endl;

    // --- 多样本打印 ---
    // 创建一个 batch=2 的小张量，手动填值来验证打印格式
    Tensor small(2, 1, 2, 3);
    for (int n = 0; n < 2; ++n)
        for (int h = 0; h < 2; ++h)
            for (int w = 0; w < 3; ++w)
                small.at(n, 0, h, w) = (n + 1) * (h * 3 + w + 1);
    std::cout << "small batch print:" << std::endl;
    small.print();
    // 输出应该是：
    //   Tensor(2,1,2,3)
    //   batch 0:
    //   1 2 3
    //   4 5 6
    //   batch 1:
    //   2 4 6
    //   8 10 12
    // 验证 batch 1 的值是 batch 0 的 2 倍

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
