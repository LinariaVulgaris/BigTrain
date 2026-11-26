# TransSfP 复现报告

论文：*Transparent Shape from a Single View Polarization Image* (ICCV 2023)
代码：https://github.com/shaomq2187/TransSfP，commit `e9823ad95a736bafd3a0b1628fff8467517169a8`
复现日期：2026-11-25 ~ 2026-11-26（最后更新：2026-11-26）
硬件：单卡 NVIDIA A10 23 GB

## 1. 结论

- 用官方代码、官方数据划分从零训练（无官方预训练权重）。主结果取 **epoch 20**（本次已保存 checkpoint
  中验证均值最低的）：All mean **17.22°**，论文 **16.29°**。
- 需要区分“官方代码默认终点”：`MAX_EPOCH=20` 实际训练 epoch 0–19，终点是 **epoch 19**，其日志验证均值为
  **17.02°**。该 checkpoint 因本次每 5 epoch 保存而缺失，无法出图/做逐帧指标，所以主表/主图仍用 epoch 20；
  若按官方默认终点算，与论文的差距是 **0.73°** 而不是 0.93°。
- 差距不是均匀的：7 个物体中 **5 个差距 ≤1°**（Bird 0.74、Cat-Back 0.39、Cat-Front 0.35、
  Tiny-Cup 0.09、Tiny-Cup-Edges 0.24），Cat-Back 与 Tiny-Cup 甚至优于论文；只有两个半球差
  3.47/2.17，合计占 7 物体净差距（6.49°）的 **87%**（按 epoch 19 算也占约 81%，结论不变）。
- 横向看，17.22° 仍优于论文表中所有 baseline 的 All（最好 SPW 19.00°），7 个物体中 6 个第一；
  唯一例外是 hemi-sphere-small（13.81° vs DeepSfP 12.87°）。因此论文“所有物体最优”的结论在本次
  复现中不完全成立。
- 训练到 epoch 49 会过拟合：All mean 回升到 **19.66°**，而且方向性明显——两个半球 **0% 的测试帧**
  变好，杯子类 78–90% 的帧变好。主结果使用 epoch 20，epoch 49 作为对照。
- 结论：该复现可作为 TransSfP 的 baseline 使用，但不是逐位复现；论文 §3.6 与官方代码在优化器、
  LR 调度、512×512 处理方式上存在出入（见 §3）。

## 2. 数据

下载并解压到 `/mnt/workspace/datasets/TransSfP/`：

| split | 物体数 | 帧数 | 说明 |
|---|---|---|---|
| real-world | 10 | 486 | 与数据集 README 一致 |
| synthetic | 13 | 936 | 13 × 72 |

校验（`baselines/tools/verify_transsfp_dataset.py`）结果：0 问题。
- 每个物体各模态（DoLP / AoLP / masks / normals-png / synthesis-normal-0..3）帧数一致、帧号对齐
- mask 取值 **0/255**，符合 loss 中 `mask < 255` 判无效像素的逻辑
- `real-world/bird-front` 不存在；官方代码构造了该数据集但**从未使用**（未加入训练集、未返回），已在 `get_dataloader.py` 中改为“目录存在才构造”

## 3. 训练配置

网络、损失、优化器与 dataloader 沿用官方代码；但 epochs / checkpoint 间隔是本次自选，且官方代码与
论文 §3.6 的文字描述并不一致（见下表）。**本次复现跟随的是代码，不是论文文字。**

| 项 | 值 |
|---|---|
| 训练集 | 13 synthetic + middle-round-cup / middle-square-cup / middle-white-cup（936 + 187 = **1123 帧 / 16 个物体**） |
| 测试集 | tiny-cup, tiny-cup-edges, bird-back, cat-back, cat-front, hemi-sphere-big, hemi-sphere-small（299 帧） |
| 输入 | DoLP + AoLP + 4 个 physics-based prior normals（共 14 通道；mask 只用于把无效像素置零，不单独作为输入通道） |
| 网络 | ResNet50 + EPSABlock 三编码器（raw / prior / confidence）+ ASPP + U-Net 解码器 |
| 分辨率 | 512×512 resize（官方 dataloader 行为；原始图像 1232×1028） |
| batch size | 5 |
| 优化器 | SGD, lr 1e-6, momentum 0.9, weight decay 5e-4（官方代码值） |
| 调度 | StepLR, step_size 7, gamma 0.1 |
| loss | cosine + 0.05 × AoLP（confidence 加权） |
| epochs | 50；主结果取 epoch 20（已保存中最优；官方默认 `MAX_EPOCH=20` 的终点是 epoch 19，验证 17.02°，未保存） |
| checkpoint | 每 5 epoch + 最后一个 epoch（作者默认每 epoch 保存） |
| 训练耗时 | 2026-11-25 21:47 → 2026-11-26 04:40，约 6.9 小时 |

论文 vs 官方代码：

| 项 | 论文 §3.6 | 官方代码（commit e9823ad） | 本次复现 |
|---|---|---|---|
| 优化器 | Adam | SGD（momentum 0.9） | SGD |
| LR 衰减 | 每 **9** epoch ×0.1 | 每 **7** epoch ×0.1 | 每 7 |
| 512×512 处理 | **crop** patch | **resize**（1232×1028 → 512×512，各向异性） | resize |

- 三项都会影响最终数值，因此与论文的差距（0.73–0.93°，取决于用 epoch 19 还是 epoch 20）中可能有
  非随机成分；只改训练设置无法判断是哪一项。
- resize 下 GT 与预测在同一坐标系比较，指标口径自洽；但 physics-based prior 来自原始几何，与拉伸后
  的半球（约 1.2× 各向异性）不再严格对应，可能是半球误差偏大的一个来源（未验证）。

代码改动（不改变网络/损失/优化器）：
- 之前的会话已加 `code/compat_env.py` 适配 Python 3.12 / PyTorch 2.7（替代 imgaug/termcolor/tensorboardX），
  并加了 `-epochs` 等参数
- 本次新增：
  - `get_dataloader.py`：`import os`；`bird-front` 目录不存在时跳过（该数据集本来就未使用）
  - `train.py`：新增 `-epochs/-save_interval/-num_workers/-gpu/-detect_anomaly`；
    本次用 50 epoch / 每 5 epoch 保存（作者默认 20 / 每 1）；最后一个 epoch 也保存，
    否则按 5 的间隔会丢掉 epoch 49

## 4. 结果

### 4.1 Table 1（Ours 行）复现

Mean angular error ↓（每个数字是该物体所有测试帧的平均；All 为 7 个物体平均）

| Method | Hemi-Sph-Big | Hemi-Sph-Small | Bird | Cat-Back | Cat-Front | Tiny-Cup | Tiny-Cup-Edges | All |
|---|---|---|---|---|---|---|---|---|
| 论文 Ours | 8.61° | 11.64° | 23.61° | 18.81° | 19.51° | 15.02° | 16.83° | 16.29° |
| **本次复现（epoch 20）** | **12.08°** | **13.81°** | **24.35°** | **18.42°** | **19.86°** | **14.93°** | **17.07°** | **17.22°** |
| **复现（epoch 20） − 论文** | +3.47 | +2.17 | +0.74 | −0.39 | +0.35 | −0.09 | +0.24 | **+0.93** |
| 本次复现（epoch 49，过拟合对照） | 22.45° | 21.14° | 25.00° | 18.70° | 20.07° | 14.19° | 16.07° | 19.66° |

完整指标（epoch 20，299 帧；Median 为各帧中位数的平均，All 行的 Acc 为 7 个物体 Acc 的平均）：

| Object | Frames | Mean ↓ | Median ↓ | Acc@11.25° ↑ | Acc@22.5° ↑ | Acc@30° ↑ |
|---|---|---|---|---|---|---|
| hemi-sphere-big | 14 | 12.08 | 11.98 | 45.24 | 96.85 | 99.55 |
| hemi-sphere-small | 16 | 13.81 | 13.51 | 36.18 | 91.63 | 98.79 |
| bird-back | 53 | 24.35 | 19.83 | 24.29 | 56.44 | 70.39 |
| cat-back | 45 | 18.42 | 15.07 | 33.55 | 72.38 | 84.26 |
| cat-front | 46 | 19.86 | 16.12 | 35.10 | 65.68 | 79.90 |
| tiny-cup | 61 | 14.93 | 10.08 | 54.90 | 82.75 | 88.85 |
| tiny-cup-edges | 64 | 17.07 | 11.70 | 48.38 | 76.74 | 84.71 |
| **All** | **299** | **17.22** | **14.04** | **39.66** | **77.50** | **86.64** |

说明：

- 论文 Table 1 的 All 列恰好等于其 7 个物体均值的算术平均（114.03 / 7 = 16.29°），本报告采用同一口径。
- **不能改用像素加权口径**：把全部测试像素混池算是 **19.49°**，因为 bird-back 贡献了 39% 的有效像素
  且误差最高。论文 Table 1 是物体级平均，两种口径不可混用，否则会得到错误的横向结论。
- 横向定位：复现的 17.22° 仍低于论文所有 baseline 的 All（SPW 19.00°、DeepSfP 19.83°、Kondo 21.39°），
  7 个物体中 6 个第一；唯一被反超的是 hemi-sphere-small（13.81° vs DeepSfP 12.87°）。
- 官方代码默认 20-epoch 训练的终点是 epoch 19（epoch 0–19），日志给出的 7 物体验证均值为 **17.02°**
  （Hemi-Big/Small 11.14/13.23、Bird 24.13、Cat-Back 18.49、Cat-Front 19.98、Tiny-Cup 14.96、
  Tiny-Cup-Edges 17.18；与论文差 0.73°）。该 checkpoint 未保存，无法生成主图；本表保留可复现的
  epoch 20 作为主结果。

### 4.2 训练收敛

| epoch | 0 | 5 | 10 | 15 | 18 | 20 | 25 | 30 | 40 | 49 |
|---|---|---|---|---|---|---|---|---|---|---|
| val mean (°) | 23.54 | 18.34 | 17.43 | 17.36 | **16.89** | 17.22 | 17.86 | 17.66 | 19.52 | 19.66 |

- 按 7 个测试物体的验证均值，最低点是 epoch 18（16.89°），epoch 24 为 16.93°；但要拿到这两个点必须在
  **测试物体上挑 epoch**，属于 test-set selection，因此不作为主结果，只作为“选点敏感性”的诊断。
- 主结果 epoch 20 是本次已保存 checkpoint 中验证均值最低的，不需要测试集选点；但它不是官方默认终点。
  官方默认 `MAX_EPOCH=20` 训练 epoch 0–19，终点 epoch 19 的验证均值为 17.02°，比 epoch 20 更好。
  前 20 个 epoch 与单独跑 20 epoch 完全一致（同 seed、同调度，50 的设置只影响循环上界和保存间隔），
  因此 epoch 19 的日志值可视为作者默认 20-epoch 配置的终点结果。
- 本次 `save_interval=5` 是自选配置（作者默认每 epoch 保存），所以 epoch 18 / 19 都没有 checkpoint；
  这是配置选择造成的，不是作者代码的限制。若要拿到官方默认终点的完整结果，需以 `SAVE_INTERVAL=1`
  重跑，或至少补存该 epoch。
- epoch ~24 之后开始过拟合，两个半球退化最明显（见 §4.4）。
- 曲线：`results/transsfp_train/val_curve.png`，数据：`results/transsfp_train/val_by_epoch.csv`

### 4.3 可视化

主图（对应论文 Fig. 7，仅 "Ours"，baseline 未复现）：
`results/transsfp_eval/figs/fig7_ours.png`

- 行：7 个测试物体；列：DoLP / AoLP / Ground-Truth / Ours
- Ours 格右下角标注该物体平均角误差（与 Table 1 一致）
- 每个物体取帧序中间的帧作为代表，规避“按误差挑图”

补充图：
- `results/transsfp_eval/figs/fig_inputs.png`：DoLP、AoLP、4 个 physics-based prior、confidence map、GT、Ours
- `results/transsfp_eval/figs/fig8_full.png`：论文 Fig. 8 风格（仅 Full 模型）：GT / Ours / Confidence / Error map + 0–60° colorbar

逐帧输出：`results/transsfp_eval/frames/<object>/<idx>-<frame>-{dolp,aolp,gt,pred,conf,err,prior0..3}.png`，
其中 `err.npy` 为逐像素角误差。注意：`results/transsfp_eval_epoch49/frames/` 只保存了 png，没有
`err.npy`（2990 个 png / 0 个 npy），因此 §4.5 的逐像素分析只针对 epoch 20；如需对 epoch 49 复算，
重跑一次 `eval_transsfp.py` 即可。

### 4.4 过拟合与泛化

| 指标 | epoch 20 | epoch 49 | 变化 |
|---|---|---|---|
| 训练运行 mean (°) | 14.91 | 11.25 | −3.66 |
| 验证 mean (°) | 17.22 | 19.66 | +2.44 |

逐物体对比（epoch 49 相对 epoch 20）：

| Object | ep20 | ep49 | Δ | ep49 逐帧变好的比例 |
|---|---|---|---|---|
| hemi-sphere-big | 12.08 | 22.45 | **+10.37** | **0%**（0/14） |
| hemi-sphere-small | 13.81 | 21.14 | **+7.33** | **0%**（0/16） |
| bird-back | 24.35 | 25.00 | +0.66 | 25%（13/53） |
| cat-back | 18.42 | 18.70 | +0.28 | 33%（15/45） |
| cat-front | 19.86 | 20.07 | +0.22 | 46%（21/46） |
| tiny-cup | 14.93 | 14.19 | **−0.74** | **90%**（55/61） |
| tiny-cup-edges | 17.07 | 16.07 | **−1.00** | **78%**（50/64） |

- 训练误差持续下降、验证误差上升，是典型过拟合；更关键的是方向性：继续训练让“见过的杯子类”几乎逐帧
  变好，却让训练集中完全不存在的半球类**逐帧全部变差**。
- 训练集只有 16 个物体 / 1123 帧，合成集 13 个物体中没有球形物体；官方代码的“数据增强”实际只有
  resize（无随机裁剪/翻转）。因此过拟合是结构性的，不是随机波动。
- 实际含义：TransSfP 在论文中优势最大的半球物体，同时也是泛化最不稳定、对训练时长最敏感的部分。
  引用其结果时应固定 checkpoint（如本报告的 epoch 20），不要按测试集挑 epoch。

### 4.5 逐像素诊断（补充分析，不在论文表格口径内）

在 epoch 20 的逐帧 `err.npy` 上，配合数据集的 0/255 mask 重算（脚本：
`baselines/tools/analyze_transsfp_errors.py`）：

- **误差是重尾的，不是灾难性翻转**：逐帧中位数 14.0°，p90 37.0°，p99 68.1°；>60° 的像素占 2.75%，
  >90° 仅 0.28%，>120° 仅 0.10%。没有出现整体前后向/方位角符号错误，误差主要集中在局部。
- **轮廓边缘效应明显**：距 mask 边界 ≤12 px 的像素占 36.9%，平均 21.9°；内部为 18.1°。
  杯子类最明显：tiny-cup 边缘 19.7° vs 内部 10.3°，tiny-cup-edges 21.9° vs 11.6°（约 2 倍）。
  bird-back 内部本身就有 23.7°，说明它的困难是全局形状，而不只是薄边。
- **物理置信图有信息量，但校准较粗**：按置信度十分位（全测试集混池分箱），误差从低置信的约 20–24°
  降到最高十分位的 11.4°；最低一档（20.7°）反而略好于第二档（24.0°），说明置信图不是精确的
  不确定性估计。逐帧 corr(confidence, error) 中位数约 −0.25，属于弱到中等相关。
- **DoLP 与误差的关系符合物理直觉**：最低 DoLP 十分位 22.1°，最高十分位 15.4°；跨帧
  corr(平均误差, 平均 DoLP) = +0.42，corr(平均误差, 平均置信) = −0.48。即偏振质量差的物体/视角更难，
  支持论文“用置信度调节 prior 贡献”的动机。
- 这一节与 §4.1 的物体级平均不冲突：像素混池会把 bird-back 的大面积高误差放大，得到 19.49°；
  两种口径回答的是不同问题，报告中不要混用。

## 5. 复现方法

```bash
# 数据校验
python -I baselines/tools/verify_transsfp_dataset.py /mnt/workspace/datasets/TransSfP

# 训练（本次 50 epoch / 每 5 epoch 保存；作者默认 20 / 每 1）
EPOCHS=50 SAVE_INTERVAL=5 bash baselines/run_transsfp_train.sh \
  /mnt/workspace/datasets/TransSfP /mnt/workspace/results/transsfp_train

# 评估主结果 checkpoint（会自动生成图与表）
python -I baselines/tools/eval_transsfp.py \
  --checkpoint /mnt/workspace/results/transsfp_train/CheckPoints/check-point-epoch-0020.pth \
  --out /mnt/workspace/results/transsfp_eval

# 训练收敛曲线
python -I baselines/tools/parse_transsfp_log.py /mnt/workspace/logs/transsfp_train.log \
  --out /mnt/workspace/results/transsfp_train

# Fig. 8 风格面板
python -I baselines/tools/make_fig8.py --eval_dir /mnt/workspace/results/transsfp_eval \
  --objects hemi-sphere-big,bird-back,tiny-cup

# 逐像素诊断（§4.5）
python -I baselines/tools/analyze_transsfp_errors.py \
  --eval_dir /mnt/workspace/results/transsfp_eval \
  --dataset_dir /mnt/workspace/datasets/TransSfP \
  --compare_dir /mnt/workspace/results/transsfp_eval_epoch49
```

一键后处理（训练结束后自动选最佳 checkpoint、出图、写完成标记）：
`baselines/tools/post_train_transsfp.sh`

## 6. 局限与注意事项

1. **没有官方预训练权重**：作者在 issue #6 明确表示 checkpoint 已无法找回，本复现是从零训练。
2. **epoch / checkpoint 口径**：主结果 epoch 20 是已保存 checkpoint 中最优的，不需要在测试物体上选点；
   但官方代码默认 `MAX_EPOCH=20` 的终点是 epoch 19（验证 17.02°），本次因 `save_interval=5` 未保存。
   验证均值最低的 epoch 18（16.89°）与 epoch 24（16.93°）只能通过在 7 个测试物体上挑 epoch 获得，
   属于 test-set selection，不建议作为主结果。若只看最终 epoch 49，结果是 19.66°。
3. **论文与代码不一致**：论文写 Adam / 每 9 epoch 衰减 / crop 512×512，官方代码是 SGD / 每 7 epoch /
   resize。本次跟随代码，因此不能称为“完全按论文配置复现”；0.73–0.93° 的差距中可能包含非随机成分。
   如需追平，最值得做的是按论文配置（Adam + step 9）重跑 20 epoch 作为对照（A10 约 2.7 小时）。
4. **baseline 未复现**：Fig. 7 / Table 1 中的 SPW、DeepSfP、Kondo、ClearGrasp、Miyazaki、Mahmoud
   需要各自代码/权重与额外训练，本次只复现 "Ours"。
5. **消融未复现**：Fig. 8 / Table 3 的 W/o AoLP Loss、W/o Confidence、单分支等需要额外训练，
   本次只给出 Full 模型的 confidence/error map。
6. **分辨率与聚合口径**：指标在 512×512 resize 上计算（官方验证循环行为；原始 1232×1028，各向异性），
   论文表格未说明测试分辨率。Table 1 的 All 是物体级平均，不能与像素级混池结果（19.49°）混用。
7. **单次训练、单随机种子**（`setup_seed(20)`），未做多次重复的均值/方差；亚度级差异是否显著未知。
8. **epoch 49 的逐帧误差未落盘**：`results/transsfp_eval_epoch49/frames/` 只有 png，没有 `err.npy`；
   逐像素复算需要重跑该 checkpoint 的评估（不影响 §4.1/§4.2 的表格）。
