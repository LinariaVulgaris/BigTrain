# Poppy Reproduction

本仓库保存了 Poppy 复现过程中的源代码、实验日志、模型输出及可视化结果。以下是主要目录说明和推荐阅读顺序。

## 📂 项目文件结构与实验结果说明
> 这一部分是gpt写的, 与实验结果有出入的地方以实验结果为准

### 1. 文件结构

```text
BigTrain/
│
├── poppy/                         # Poppy 核心代码及实验脚本
│   ├── README.md                  # 方法说明、运行方式及输出格式
│   ├── run.py                     # 实验入口
│   ├── make_figs.py               # 生成实验对比可视化图
│   ├── full.log                   # 完整 Poppy 实验日志
│   ├── imgonly.log                # Image Only 实验日志
│   ├── steps.log                  # 不同步数实验日志
│   ├── test_base.log              # 初始 Base 测试日志
│   └── test_poppy.log             # 初始 Poppy 测试日志
│
├── out_full_base/                 # MoGe-2 原始基线实验
├── out_full_imgonly/              # 仅使用 Image Shift 的实验
├── out_full_poppy/                # 完整 Poppy 实验
│   ├── log.csv                    # 实验指标汇总
│   ├── vis/                       # 法线可视化结果
│   └── outputs/                   # npz 数据（normals、Ls、Ld）
│
├── out_test_base/                 # 初始 Base 测试
├── out_test_poppy/                # 初始 Poppy 测试
│
├── out_steps_ballblack_10/        # Ballblack：10 步优化
├── out_steps_ballblack_25/        # Ballblack：25 步优化
├── out_steps_ballblack_50/        # Ballblack：50 步优化
│
├── out_steps_kettle_10/           # Kettle：10 步优化
├── out_steps_kettle_25/           # Kettle：25 步优化
├── out_steps_kettle_50/           # Kettle：50 步优化
│
├── figs/                          # 实验可视化图
└── figs_full/                     # 完整实验可视化图
```

### 2. 核心实验结果（建议优先阅读）

| 实验目录 | 实验内容 | 推荐程度 |
|---|---|---|
| `out_full_base/` | MoGe-2 原始基线 | ⭐⭐⭐⭐⭐ |
| `out_full_imgonly/` | 仅使用 Image Shift | ⭐⭐⭐⭐⭐ |
| `out_full_poppy/` | 完整 Poppy 方法 | ⭐⭐⭐⭐⭐ |
| `out_steps_ballblack_10/` | Ballblack 的 10 步优化(无输出法线优化) | ⭐⭐⭐⭐ |
| `out_steps_ballblack_25/` | Ballblack 的 25 步优化(无输出法线优化) | ⭐⭐⭐⭐ |
| `out_steps_ballblack_50/` | Ballblack 的 50 步优化(无输出法线优化) | ⭐⭐⭐⭐ |
| `out_steps_kettle_10/` | Kettle 的 10 步优化(无输出法线优化) | ⭐⭐⭐⭐ |
| `out_steps_kettle_25/` | Kettle 的 25 步优化(无输出法线优化) | ⭐⭐⭐⭐ |
| `out_steps_kettle_50/` | Kettle 的 50 步优化(无输出法线优化) | ⭐⭐⭐⭐ |

### 3. 值得关注的实验发现

**（1）不同方法的效果对比**

在目前测试的 6 个物体上，三种方法的平均角度误差如下（越低越好）：

| 方法 | 平均角度误差 |
|---|---:|
| Base | 10.889° |
| Image Only | **8.300°** |
| Full Poppy | 8.722° |

初步结果表明：
- Image Only 在这 6 个物体上的平均表现最好。
- Full Poppy 相比 Base 有明显改善，但暂时没有超过 Image Only。
- 完整 Poppy 的额外优化机制是否能够带来稳定收益，值得进一步研究。

**（2）优化步数实验**

- Ballblack：25 步的结果优于 10 步和 50 步。
- Kettle：10 步和 25 步表现接近，50 步整体表现较差。
- 优化步数并非越多越好，不同物体可能具有不同的最优步数。

**（3）值得进一步分析的物体**

- `bird`：Image Only 和 Full Poppy 均明显优于 Base。
- `owlblue`：Poppy 相比 Base 有明显提升，适合观察法线改善情况。
- `kettle`：不同方法和优化步数之间存在明显差异。
- `ballblack`：适合分析优化步数对结果的影响。
- `pineapple`：适合研究复杂表面上的法线估计效果。

> 注意：以上结论基于目前保存的 6 个物体实验结果，属于阶段性发现，尚不能代表整个数据集上的泛化表现。

### 4. 推荐查看顺序

1. **查看 `log.csv`**：了解各实验的平均角度误差、RMSE 和准确率等指标。
2. **查看 `vis/`**：直观比较输入图像、预测法线及误差图。
3. **查看 `outputs/`**：获取原始 `.npz` 数据，可用于后续定量分析和重新绘图。
4. **查看 `figs_full/`**：浏览整理好的完整实验可视化结果。
5. **查看 `out_steps_*/`**：分析优化步数对实验效果的影响。

### 5. 重要文件说明

- `log.csv`：记录每个物体的实验指标，包括平均角度误差、Median、RMSE、Acc@11.25°、Acc@22.5° 和 Acc@30°。
- `vis/`：保存法线预测的可视化图片。
- `outputs/*.npz`：保存预测法线 `normals`、恢复的镜面分量 `Ls` 和漫反射分量 `Ld`。
- `poppy/make_figs.py`：支持从 `.npz` 数据重新生成 Base 与 Poppy 的对比图。

**特别说明：** 如果后续需要深入分析实验结果，建议优先保留 `out_full_base/`、`out_full_imgonly/`、`out_full_poppy/` 中的 `log.csv`、`vis/` 和 `outputs/`，以及全部优化步数实验目录。这些数据对后续方法改进、误差分析和论文实验撰写具有较高价值。


## Environment

- Python 3.12
- PyTorch 2.7.0a0+7c8ec84dab.nv25.03
- NumPy 1.26.4
- Transformers 5.3.0
- Diffusers 0.37.1
- OpenCV 4.10.0

## Notes

Dataset files, model weights, Hugging Face caches, and other large external resources are not included in this repository.
