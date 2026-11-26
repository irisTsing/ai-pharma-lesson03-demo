# demo 2：ECFP4 + 机器学习打分器（随机森林 / 多层感知机）

## 1. 对应方法与文献

- 分子表示：ECFP4（Morgan r=2，2048 bit）。
- 模型一（CPU）：scikit-learn 随机森林，300 棵树，`class_weight=balanced_subsample`。
- 模型二（GPU）：torch 多层感知机 2048 → 512 → 128 → 1，ReLU + Dropout，Adam，BCE 损失，正样本加权。
- 划分与泄漏：主口径用 Murcko scaffold 划分（同一骨架不跨训练/测试），另跑随机划分作对照。
- 评价：与 demo 1 同一测试集、同一份参考分子，比较 EF1%/EF5%/ROC-AUC/top-k 命中。
- 数据偏倚的文献背景：Chen et al., PLoS ONE 2019, 14, e0220113（DUD-E 诱饵偏倚导致机器学习指标虚高）。

## 2. 运行

```bash
cd demos/demo_vs
bash demo_ml_screen/run_demo.sh                    # 自动选设备
bash demo_ml_screen/run_demo.sh --device cuda      # gpu1：MLP 在 GPU 上训练
bash demo_ml_screen/run_demo.sh --device cpu       # 纯 CPU 回退（同一模型）
```

输入：`data/drd3_library_10k.csv`、`data/drd3_split.json`。
输出：`demo_ml_screen/results/` 下的 `comparison_table.csv`、`metrics.json`、`enrichment_ml.png`、`library_scores.csv`。

## 3. 实测结果（2026-11-25，开发机，CPU，4 线程，30 epochs）

对照表（scaffold 测试集 2,000 个分子，其中 96 个活性，基准率 4.8%）：

| 方法 | ROC-AUC | EF1% | EF5% | EF10% | top-20 命中 | top-50 | top-100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 随机排序 | 0.500 | 1.00 | 1.00 | 1.00 | 0.96 | 2.4 | 4.8 |
| Tanimoto 相似性（demo 1） | 0.960 | 20.83 | 13.96 | 8.12 | 20 | 50 | 67 |
| 随机森林（scaffold 划分） | 0.9997 | 20.83 | 19.58 | 10.00 | 20 | 50 | 94 |
| MLP（CPU，scaffold 划分） | 0.9953 | 20.83 | 19.79 | 9.90 | 20 | 50 | 95 |
| 随机森林（随机划分） | 0.9986 | 20.83 | 19.38 | 10.00 | 20 | 50 | 93 |
| MLP（CPU，随机划分） | 0.9978 | 20.83 | 19.58 | 9.90 | 20 | 50 | 94 |

耗时（4 线程）：

| 步骤 | 实测 |
| --- | --- |
| ECFP4 指纹（10,000 分子） | 2.8 s |
| 随机森林训练（8,000 × 2048，300 棵树） | 3.5 s |
| 随机森林对全库推理 | 0.38 s |
| MLP 训练（30 epochs，CPU） | 5.7 s（约 0.19 s/epoch） |
| MLP 对全库推理（CPU） | 0.13 s |
| 全流程（两个划分 × 两个模型） | 24.3 s |

GPU 路径：本机无 NVIDIA 驱动，未实测，待 gpu1 首次验证。模型只有约 1.1 M 参数，
batch 512 × 2048 的输入约 4 MB，按设计峰值显存应小于 0.3 GB；脚本会打印真实单轮耗时与
`torch.cuda.max_memory_allocated()`，并与 CPU 数字并列写进 `metrics.json`。

## 4. 结果怎么读（课堂上要讲的三点）

1. **AUC 接近 1 不代表能筛新药。** 随机森林和 MLP 的 ROC-AUC 都在 0.995 以上，
   top-100 能捞到 94～95 个活性；但这是 DUD-E 诱饵太“容易”造成的：诱饵来自 ZINC，
   与活性分子的 2D 拓扑差异明显，模型学到的是“像不像这类化合物”，不是“结合不结合”。
2. **EF1% 饱和。** 测试集基准率 4.8%，EF1% 上限 20.8，前三行都顶到上限。
   区分方法要看 EF5%、top-50、top-100。MLP 的 EF5% 最高（19.79），Tanimoto 最低（13.96）。
3. **scaffold 划分与随机划分在这份数据上几乎没差别**（0.9997 对 0.9986）。
   原因是测试侧只有 96 个活性、诱饵又极易区分，泄漏的影响被淹没了。
   不要据此说“随机划分没问题”；讲清楚这一点，比报一个漂亮数字重要。

## 5. 课堂上可以改的参数

- `--epochs 100`：MLP 在 CPU 上约 19 s，在 GPU 上 1～2 s；看训练是否更快过拟合（loss 会到 1e-4 量级）。
- `--n-trees 50` 与 `300` 对比：训练时间与 top-100 命中。
- `--hidden 128`：MLP 变小，看 EF5% 掉多少。
- `--device cuda` / `--device cpu`：同模型两种设备，记录单轮耗时、峰值显存、加速比。
- `--n-jobs 1`：模拟单核，观察随机森林训练时间如何变化。

## 6. 常见坑

- `--n-jobs -1` 在 50 人共用的节点上会开 64 个线程，把整台机器拖死。脚本默认 4，课堂上不要改大。
- MLP 的 loss 很快降到 1e-4，不是 bug，是数据太容易分；这也是上面第 4 节第 1 点的证据。
- `library_scores.csv` 里训练集分子（8,000 个）的打分是模型见过的，偏乐观，只能做教学演示，不能当筛选结果汇报。
- scaffold 划分必须做标签分层：DRD3 的 480 个活性分子恰好对应 480 个不同骨架，
  而诱饵里有 146 个分子共享同一骨架。简单按骨架组大小切分，会把活性分子几乎全部挤到训练侧（本 demo 已处理该问题）。
- 每次重跑前 `prepare_data.py` 已固定随机种子，不要手动改 JSON 里的下标，否则两个 demo 的对照表不再可比。
