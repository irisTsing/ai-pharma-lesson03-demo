# demo 4：融合流水线——2D 特征粗筛 + 对接精排（教师预计算，学生分析）

## 1. 对应方法与文献

- 粗筛：ECFP4（Morgan r=2，2048 bit）+ Tanimoto 相似度，参考 10 个已知活性分子（与 demo 1 同一套）。
- 精排：AutoDock Vina 1.2.3（CPU）或 Uni-Dock（GPU）分子对接打分。
- 思路对应"层级虚拟筛选"（hierarchical / cascade virtual screening）：先用便宜的 2D 特征把库缩小一到两个数量级，
  再对剩下的分子做昂贵的结构对接。文献综述见
  - Lavecchia & Di Giovanni, *Virtual Screening Strategies in Drug Discovery: A Critical Review*, Curr. Med. Chem. 2013, 20, 2839–2860。PMID：23746218。
- 对接引擎与参数：同 demo 3（vina 1.2.3 / Uni-Dock v1.2.0，vina 打分函数，网格 0.375 Å，20³ 盒子）。

## 2. 两段式设计（谁跑什么）

| 阶段 | 谁跑 | 耗时 | 产物 |
| --- | --- | --- | --- |
| 教师预计算 | 教师课前 | 本机 CPU 约 2 小时；gpu1 上 Uni-Dock 约 10 分钟 | `results/full_docking.csv` |
| 学生分析 | 课堂 8 分钟 | 10～30 秒 | `results/strategy_comparison.csv`、`pipeline_summary.json`、`pipeline_plot.png` |

学生只读预计算好的 CSV，不再跑对接。这样 50 人同时上课也不会把 8 张卡挤爆。

数据：`data/drd3_dock_library.csv` = 480 个活性 + 1,520 个诱饵 = 2,000 个分子（seed 7，库内抽样）。
其中 1,981 个配体的 pdbqt 准备成功（19 个因互变异构体不一致被排除，见 `data/dock_ligands/prep_summary.csv`）。

## 3. 教师预计算

### 3.1 推荐路线：gpu1 上 Uni-Dock GPU 批量（待 gpu1 验证）

```bash
# gpu1，已按 setup_gpu_env.sh 装好共享环境
export VS_PY=/public/AI药学实践/demo_vs/envs/vs/bin/python

$VS_PY demo_pipeline/precompute_docking.py \
    --method unidock --search-mode fast \
    --unidock-mem-mb 3000 --chunk 500 \
    --out demo_pipeline/results/full_docking.csv
```

- `--chunk 500` 表示每 500 个配体作为一批交给 Uni-Dock，跑完立刻落盘；
  批越大 GPU 利用越充分，显存 24 GB 下单批 500～1000 个配体是合理区间，按实测调整。
- 预计耗时：fast 模式下 2,000 个配体约 10 分钟量级（官方论文报告的加速是相对 CPU vina 一到两个数量级；
  实际值以 gpu1 首跑为准，脚本会打印每批的 `wall_per_lig_s`）。
- 产物列与 CPU 路径完全一致：`mol_id, label, affinity, valid, status, method, wall_s, ...`。
- **待 gpu1 验证**：Uni-Dock 批量输出文件名（`<id>_out.pdbqt` / `<id>_out.sdf`）与
  `> <Uni-Dock RESULT>` 属性的解析已在本机用 v1.2.0 核实，但真实 GPU 运行未验证；
  若首跑解析不到分数，先看 `demo_pipeline/results/unidock_out/unidock.log`。

### 3.2 退路：本机 vina CPU 逐配体（已验证可跑）

```bash
cd demos/demo_vs
../venv_chemprop/bin/python demo_pipeline/precompute_docking.py \
    --method vina --workers 20 --cpu 3 --exhaustiveness 4 --chunk 100 \
    --out demo_pipeline/results/full_docking.csv
```

实测耗时（本机，64 核，但同一台机器还跑着其他用户的任务，负载 70～120）：

| 场景 | 吞吐 | 1,981 个预计 |
| --- | --- | --- |
| 机器空载（12 配体并行 6 进程，cpu 4） | 约 36 个/分钟 | 约 55 分钟 |
| 本机 20 进程 × 3 线程，负载 70～120 | 10～27 个/分钟 | 1.2～3.3 小时（实测约 2 小时量级） |

结论：**教师课前不要用本机 CPU 跑全量**，用 3.1 的 GPU 路径；
只有在没有 GPU 可用时，才用 CPU 路径并在课前至少留 2 小时。

### 3.3 增量落盘与断点续跑

- 每跑完 `--chunk` 个配体就把结果合并写一次 `full_docking.csv`，中途停止不丢结果；
- 重新执行同一命令时，脚本读已有 CSV 里的 `mol_id`，只跑没跑过的配体；
- 配体顺序在预计算开始时用固定种子打乱（`--seed 2024`），所以“只跑了一部分”时，
  已完成部分的活性/诱饵比例接近全库，学生可以先用部分结果走通流程，不必等全量。

### 3.4 从 vina 日志补汇总

如果预计算中途被打断、或想从日志重新汇总：

```bash
../venv_chemprop/bin/python demo_pipeline/collect_dock_logs.py
```

它扫描 `results/dock_work/logs/*.log`，把 `   1   -9.398` 这种最优模式行解析出来，
生成与预计算同列名的 `full_docking.csv`，并打印覆盖率与有效数。

## 4. 学生分析

```bash
cd demos/demo_vs
bash demo_pipeline/run_demo.sh                       # 默认 top-frac 0.10
bash demo_pipeline/run_demo.sh --top-frac 0.05       # 粗筛只留 5%
```

或直接调用：

```bash
../venv_chemprop/bin/python demo_pipeline/student_pipeline.py --top-frac 0.10 --topk 20
```

学生这一步做三件事：

1. 用 ECFP4 + Tanimoto（10 个参考活性分子）给 2,000 个分子打分，取 top 10%（约 198 个）；
2. 把粗筛结果与预计算的对接打分对齐，统计：
   - 粗筛 top 10% 的活性召回率（相对全库 480 个活性）；
   - 只对接这 10% 与对接全库，top-20 命中数差多少、重合多少；
   - 省下的对接次数与机时；
3. 画富集曲线（对接 vs 2D 相似性 vs 随机）与“2D 分数—对接分数”散点图。

输出（`demo_pipeline/results/`）：

| 文件 | 内容 |
| --- | --- |
| `strategy_comparison.csv` | 四种策略的对接次数、占全库成本比例、活性召回、top-20 命中 |
| `pipeline_summary.json` | 全部指标与成本估算，供讲稿引用 |
| `pipeline_plot.png` | 富集曲线 + 2D/对接散点 |
| `full_docking.csv` | 教师预计算的对接打分（学生只读） |

成本估算公式：`单次对接秒数（CSV 里 wall_s 的中位数）× 分子数`。全库 2,000 个 vs 粗筛后 200 个，
直接给出机时比例；GPU 场景把单次对接秒数换成显卡吞吐即可。

## 5. 当前状态（诚实说明）

- 交付时预计算仍在后台继续，`full_docking.csv` 是**部分结果**（交付时约 220 个分子，
  其中活性约 140、诱饵约 80）。首轮按库顺序执行、活性在前，只跑出活性的问题已在重启后修正
  （现在按固定种子打乱顺序，重启后的批次已含诱饵）。
- 部分结果的基准率（约 63%）远高于全库（24%），**富集数字只用于验证流程，不能当结论**。
  脚本已对单类样本做保护：ROC-AUC 与 EF 返回 `nan` 并打印提示，不会崩；
  当前部分数据下 `student_pipeline.py` 能正常跑完（1.5 s），输出的 `strategy_comparison.csv`、
  `pipeline_summary.json`、`pipeline_plot.png` 可用于课堂讲流程。
- 预计算全部完成后，重新跑一次 `student_pipeline.py`，三份输出会自动写入真实富集与成本数字，
  本 README 第 4 节的表格直接引用这两份文件，不需要手工填表。
- 部分结果暴露出的现象值得课堂讨论：在这 220 个分子上，对接打分的 ROC-AUC 只有 0.30（低于随机），
  而 2D 相似性有 0.94。也就是说 vina 在这个受体/盒子上对这一组分子没有富集能力，
  粗筛反倒更有效。等全量跑完看这个结论是否稳定；无论稳定与否，这都是“结构筛选不是万灵药”的真实例子。

## 6. 课堂上可以改的参数

- `--top-frac`：0.05 / 0.10 / 0.20，看“粗筛更严”对召回与成本的双向影响。
- `--topk`：20 / 50 / 100，top-k 命中数随 k 变化。
- 把粗筛打分从 Tanimoto 换成 `demo_ml_screen/results/library_scores.csv` 里的 `rf_score`/`mlp_score`，
  比较“2D 相似性粗筛”和“机器学习粗筛”谁更适合当前置过滤器（注意训练集泄漏，见 demo 2 README）。
- 讨论：如果单次对接不是 15 秒而是 0.1 秒（GPU 批量），粗筛还值得做吗？成本模型会给出答案。

## 7. 常见坑

- **不要用部分预计算数据下富集结论**：只有一类样本时 EF/ROC 无定义，脚本会返回 `nan` 并提示。
- CPU 预计算很吃核：本机 `--workers 20 --cpu 3` 已经把 64 核占满 60 线程，同一台机器上再跑别的重活会互相拖慢。
- 用 Uni-Dock 预计算时不要忘了 `--unidock-mem-mb`（默认 3000），否则一次占满 24 GB，同卡的人会 OOM。
- 个别配体的 vina 打分是正数（本机 12 个里有 1 个，源于 obabel 生成的扭转树对稠环不稳健），
  CSV 的 `invalid` 列会标 False，分析时过滤掉，不要当作强结合。
- `student_pipeline.py` 读的是 `drd3_dock_library.csv` 与 `full_docking.csv` 的交集；
  如果换靶点，两个文件都要换，否则交集为空、报“对接库 0 个分子”。
- 对接库 2,000 个是抽样，活性比例约 24%（真实筛选库远低于此），富集数字不能直接外推到真实库。
