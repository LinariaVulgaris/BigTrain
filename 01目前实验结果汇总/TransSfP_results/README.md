# TransSfP 复现结果（ICCV 2023）

本目录汇总 TransSfP（*Transparent Shape from a Single View Polarization Image*, ICCV 2023）
在官方数据划分上的从零训练复现结果，包括实验报告、论文风格可视化、逐帧指标、训练日志与诊断脚本。

- 论文：<https://arxiv.org/pdf/2204.06331.pdf>
- 官方代码：<https://github.com/shaomq2187/TransSfP>，commit `e9823ad95a736bafd3a0b1628fff8467517169a8`
- 数据：官方 TransSfP 数据集（10 个真实物体 / 486 帧，13 个合成物体 / 936 帧；测试集为 7 个留出真实物体 / 299 帧）
- 训练：单卡 NVIDIA A10 23 GB，从零训练，未使用官方预训练权重（作者表示权重已无法找回）

## 主结果

物体级平均角误差（Mean angular error，越低越好；All 为 7 个测试物体的算术平均）：

| 配置 | Hemi-Big | Hemi-Small | Bird | Cat-Back | Cat-Front | Tiny-Cup | Tiny-Cup-Edges | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 论文 Table 1（Ours） | 8.61° | 11.64° | 23.61° | 18.81° | 19.51° | 15.02° | 16.83° | **16.29°** |
| 本次复现 epoch 20 | 12.08° | 13.81° | 24.35° | 18.42° | 19.86° | 14.93° | 17.07° | **17.22°** |
| 本次复现 epoch 49（过拟合对照） | 22.45° | 21.14° | 25.00° | 18.70° | 20.07° | 14.19° | 16.07° | **19.66°** |

要点：

- **主结果取 epoch 20（17.22°）**：这是本次已保存 checkpoint 中验证均值最低的，不需要在测试物体上选 epoch。
- 官方代码默认 `MAX_EPOCH=20` 实际训练 epoch 0–19，终点是 **epoch 19**，训练日志中的验证均值为
  **17.02°**（与论文差 0.73°）。该 checkpoint 因本次每 5 epoch 保存而缺失，因此没有对应的图和逐帧指标。
- 与论文的差距集中在两个半球（+3.47° / +2.17°），7 个物体中 5 个差距 ≤1°，Cat-Back 与 Tiny-Cup 甚至优于论文。
- 继续训练到 epoch 49 会过拟合：All 回升到 19.66°；两个半球 **0% 的测试帧**变好，杯子类 78–90% 的帧变好。
- 17.22° 仍优于论文表中所有 baseline 的 All（最好为 SPW 19.00°），7 个物体中 6 个第一；唯一例外是
  hemi-sphere-small（13.81° vs DeepSfP 12.87°）。

完整分析、论文与官方代码的配置差异、局限说明见 [`TransSfP_reproduction.md`](TransSfP_reproduction.md)。

## 目录内容

```
TransSfP_results/
├── README.md                       # 本文件
├── TransSfP_reproduction.md        # 完整实验报告（指标 / 过拟合 / 逐像素诊断 / 局限）
├── figs/
│   ├── fig7_ours.png               # 论文 Fig. 7 风格：7 个物体的 DoLP / AoLP / GT / Ours
│   ├── fig8_full.png               # 论文 Fig. 8 风格：GT / Ours / Confidence / Error map
│   ├── fig_inputs.png              # 输入与中间量：DoLP / AoLP / 4 个 prior / confidence / GT / Ours
│   └── val_curve.png               # 训练/验证收敛曲线
├── metrics/
│   ├── metrics_summary.csv/.md     # epoch 20 的物体级指标（Mean/Median/Acc@11.25/22.5/30）
│   ├── metrics_per_frame.csv       # epoch 20 的 299 帧逐帧指标
│   ├── val_by_epoch.csv            # 各 epoch 的验证均值
│   └── eval.log                    # epoch 20 评估日志
├── control_epoch49/
│   ├── figs/fig7_ours.png          # epoch 49 的 Fig. 7 风格图
│   ├── figs/fig_inputs.png         # epoch 49 的输入/中间量图
│   ├── metrics_summary.csv/.md     # epoch 49 的物体级指标
│   ├── metrics_per_frame.csv       # epoch 49 的逐帧指标
│   └── eval.log                    # epoch 49 评估日志
├── tools/
│   └── analyze_transsfp_errors.py  # 逐像素诊断脚本（重尾 / 边缘 / 置信度 / DoLP / 配对比较）
└── logs/
    └── transsfp_train.log          # 50 epoch 完整训练日志（含 epoch 19 的 17.02°）
```

## 报告中的路径与仓库文件对应

报告 `TransSfP_reproduction.md` 写于工作区 `/mnt/workspace/`，其中路径与仓库内文件的对应关系：

| 报告中的路径 | 本目录中的文件 |
|---|---|
| `results/transsfp_eval/figs/fig7_ours.png` | `figs/fig7_ours.png` |
| `results/transsfp_eval/figs/fig8_full.png` | `figs/fig8_full.png` |
| `results/transsfp_eval/figs/fig_inputs.png` | `figs/fig_inputs.png` |
| `results/transsfp_train/val_curve.png` | `figs/val_curve.png` |
| `results/transsfp_eval/metrics_summary.{csv,md}` | `metrics/metrics_summary.{csv,md}` |
| `results/transsfp_eval/metrics_per_frame.csv` | `metrics/metrics_per_frame.csv` |
| `results/transsfp_train/val_by_epoch.csv` | `metrics/val_by_epoch.csv` |
| `results/transsfp_eval_epoch49/*` | `control_epoch49/*` |
| `baselines/tools/analyze_transsfp_errors.py` | `tools/analyze_transsfp_errors.py` |
| `logs/transsfp_train.log` | `logs/transsfp_train.log` |

未提交的内容：

- **逐帧可视化与逐像素误差**：epoch 20 的 `results/transsfp_eval/frames/`（约 286 MB）与 epoch 49 的
  `results/transsfp_eval_epoch49/frames/`（约 137 MB）未提交，避免仓库过大。需要时可按报告 §5 重跑
  `eval_transsfp.py` 生成。
- **模型 checkpoint**：体积较大，未提交；复现命令与超参数见报告 §3 / §5。
- **数据集与官方代码**：数据集需从官方渠道下载；代码使用官方仓库 commit `e9823ad` 加少量 Python 3.12 /
  PyTorch 2.7 兼容性改动（报告 §3 有说明）。

## 复现口径说明

- 指标在 512×512 resize 上计算（官方验证循环行为；原图 1232×1028，存在各向异性拉伸）。
- 报告 Table 1 的 All 是**物体级算术平均**；全部测试像素混池的结果是 19.49°，两种口径不可混用。
- 论文 §3.6 与官方代码在优化器（Adam vs SGD）、LR 衰减间隔（9 vs 7）、512×512 处理（crop vs resize）
  上不一致；本次跟随官方代码，因此属于“代码口径复现”，不是逐位复现论文配置。
- 单次训练、单随机种子（`setup_seed(20)`），未做多次重复的均值/方差。

## 引用

官方仓库给出的引用信息：

```bibtex
@article{mingqi2022transparent,
  title={Transparent Shape from Single Polarization Images},
  author={Mingqi, Shao and Chongkun, Xia and Zhendong, Yang and Junnan, Huang and Xueqian, Wang},
  journal={arXiv preprint arXiv:2204.06331},
  year={2022}
}
```
