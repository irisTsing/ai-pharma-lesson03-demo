# demo 1：基于 2D 指纹的相似性检索与虚拟筛选评价

## 1. 对应方法与文献

- 分子表示：ECFP4（Extended Connectivity Fingerprint，半径 2，2048 bit），RDKit Morgan 实现。
  - Rogers & Hahn, *Extended-Connectivity Fingerprints*, J. Chem. Inf. Model. 2010, 50, 742–754。DOI：10.1021/ci100050t。
- 相似度：Tanimoto 系数。
- 评价框架：DUD-E（Database of Useful Decoys-Enhanced）的 actives/decoys 与富集因子 EF、ROC-AUC。
  - Mysinger et al., *Directory of Useful Decoys, Enhanced (DUD-E)*, J. Med. Chem. 2012, 55, 6582–6594。DOI：10.1021/jm300687e。
- 需要注意的既有结论：DUD-E 的活性分子大量来自同一化学系列，诱饵与活性分子的 2D 拓扑差异明显，
  所以 2D 相似性与机器学习方法在 DUD-E 上的富集普遍偏乐观。参见
  - Chen et al., *Hidden bias in the DUD-E dataset leads to misleading performance of deep learning in structure-based virtual screening*, PLoS ONE 2019, 14, e0220113。DOI：10.1371/journal.pone.0220113。

## 2. 方法与评价口径

- 靶点：DUD-E DRD3（多巴胺 D3 受体，晶体结构 3PBL）。
- 筛选库：480 个 actives + 9,520 个 decoys = 10,000 个分子（decoys 从 34,050 个里按 `seed=42` 随机抽样，固定可复现）。
- 参考分子：10 个已知活性分子，只从训练侧抽取（`seed=11`）。库中每个分子的打分 = 与这 10 个参考分子的最大 Tanimoto 相似度。
- 评价集：Murcko scaffold 划分的测试集 2,000 个分子（96 个活性，基准率 4.8%），
  与 demo 2 用同一批分子、同一份参考分子，两条方法可以直接对照。
- 指标：EF1%、EF5%、EF10%、ROC-AUC、top-k 命中数。EF 的定义是“前 x% 的活性命中率 ÷ 全库活性比例”。

## 3. 运行

CPU（开发机实测，无需 GPU）：

```bash
cd demos/demo_vs
bash demo_similarity/run_demo.sh                    # 自动选设备，无 GPU 时回退 CPU
bash demo_similarity/run_demo.sh --device cuda      # gpu1 上用 GPU 算 Tanimoto
```

直接调用：

```bash
../venv_chemprop/bin/python demo_similarity/demo_similarity.py --device auto --topk 20
```

输入：`data/drd3_library_10k.csv`、`data/drd3_split.json`（由 `common/prepare_data.py` 生成）。
输出：`demo_similarity/results/` 下的 `metrics.json`、`top_hits.csv`、`scores.csv`、`enrichment_curve.png`。

## 4. 实测结果（2026-11-25，开发机，CPU，4 线程）

| 项目 | 实测值 |
| --- | --- |
| ECFP4 指纹生成（10,000 分子 × 2048 bit） | 2.9～3.3 s |
| Tanimoto 打分（10,000 × 10） | 0.105 s |
| 全流程（含数据加载、画图、写文件） | 6.3 s |
| 测试集 ROC-AUC | 0.960 |
| 测试集 EF1% / EF5% / EF10% | 20.83 / 13.96 / 8.12 |
| 测试集 top-20 / top-50 命中 | 20 / 50（随机期望 0.96 / 2.4） |
| 全库 ROC-AUC | 0.933 |
| 全库 EF1% / EF5% | 20.83 / 13.58 |

EF1% 在上表里已经饱和：测试集活性比例 4.8%，EF1% 的理论上限是 1/0.048 ≈ 20.8，
前 1%（20 个）全部命中活性分子，所以 EF1% 区分不出方法优劣，要比 EF5% 和 top-100 命中数。

GPU 路径（`--device cuda`）：本机没有 NVIDIA 驱动，未实测，待 gpu1 首次验证。
按设计，指纹矩阵 10,000 × 2048 以 float32 常驻显存约 82 MB，分块后峰值显存估计 0.25～0.3 GB，
脚本会打印 `torch.cuda.max_memory_allocated()` 的真实峰值与 CPU/GPU 加速比，写进 `metrics.json`。

## 5. 课堂上可以改的参数

- `--topk 50`：改变 top-hits 表长度，讨论“检索结果要多少条才够用”。
- `--radius 3`（ECFP6）与 `--radius 2`（ECFP4）对比：半径更大更看重局部环境，回收率通常下降。
- `--n-bits 1024`：指纹变短，速度更快，碰撞增多。
- 参考分子数：改 `common/prepare_data.py --n-refs 30` 后加 `--force` 重跑，看“已知活性分子越多，检索越稳”。
- `--device cuda` 与 `--device cpu` 各跑一次，对比 `metrics.json` 里的 `timing_s`。

## 6. 常见坑

- **EF1% 饱和**：小库 + 强富集时前 1% 全中，EF1% 顶到理论上限，不要用它下结论。
- **参考分子自身排在第一位**：10 个参考分子就在库里，Tanimoto = 1.0，必然占据 top-10。这是检索任务的正常设定，统计 EF 时没有剔除。
- **DUD-E 的 analog bias**：DRD3 的 480 个活性分子覆盖的化学系列集中，相似性方法因此显得很强。把这个数字当成“真实筛选能力”会高估。
- **GPU 结果与 CPU 必须逐位核对**：Ampere 显卡默认可能开 TF32，2048 个 1 相加会丢精度。脚本已显式关闭 TF32，并输出两边最大误差（应为 1e-6 量级）。
- 首次运行若报找不到数据，先跑 `bash data/download_data.sh` 与 `common/prepare_data.py`。
