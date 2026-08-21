# 技术备忘：numpy float32 大轴归约饱和

> 2026-08-21，macOS arm64，numpy 2.4.3 / 2.5.1 / 2.5.2（Accelerate 构建），
> Python 3.12。

## 现象

对形状 `(N, C)`、`C >= 3` 的 float32 数组沿 N 轴做 `mean`/`sum`，当
N 约超过 2^25 时结果严重错误：随机 `U(0,1)` 数据的均值本应约 0.5，
实际返回 0.25（N=2^26）、0.125（N=2^27），即累加和恰好饱和在 2^24。

复现：

```python
import numpy as np
a = np.random.default_rng(0).random((2**26, 3), dtype=np.float32)
print(a.mean(axis=0))  # [0.25 0.25 0.25]，应为约 [0.5 0.5 0.5]
```

## 根因

该路径退化为朴素逐元素 float32 求和（未启用成对求和）。float32 尾数
24 位，累加值达到 2^24 后继续加 0.5 会被舍入丢弃，于是总和封顶。
1D 连续数组与 float64 输入不受影响；逐帧/小块轴归约不受影响。

## 处置

- 代码层：`cthulhu_backend/numeric.channel_stats` 统一用 float64 累加
  做大数组的逐通道均值/方差，管线中的参考统计已切换到该工具。
- 测试层：`tests/test_numeric.py::test_channel_stats_large_axis_is_accurate`
  以 2^26 规模做回归护栏。
- 上游：待向 numpy 提交 issue（附最小复现）；镜像暂无 >2.5.2 的新版本。

## 团队约定

对超过约 3×10^7 个 float32 元素的多列归约，禁止直接用 `np.mean(axis=...)`，
一律先逐帧/逐块小轴归约或用 `dtype=np.float64` 显式累加。
