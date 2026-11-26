# 第三讲 demo：虚拟筛选四层流水线（特征方法 → 机器学习 → 结构对接 → 融合流水线）
学生第一次上课请看 [学生操作手册](学生操作手册.md)，里面按环节给出可以直接复制给 Agent 的话。


本目录是"AI 药学实践"第三讲的课堂 demo 套件，主题是"基于特征的快速药物筛选方法"与"基于结构的药物筛选方法"。
所有指标来自真实运行；开发机配有 4 张 NVIDIA A800，GPU 路径（demo 1/2/3）已在本机实测跑通，3090 上的表现需要课前在 gpu1 再确认一次。

## 0. 完整运行步骤

仓库里已经带好数据和预计算好的对接分数，clone 下来就能跑。下面是从环境到五个环节全部跑完的顺序。

### 0.1 装环境

```bash
conda create -y -n vs python=3.10
conda activate vs
pip install numpy pandas scikit-learn matplotlib rdkit torch

# 外部程序
sudo apt install autodock-vina openbabel      # vina 与 obabel
conda install -c conda-forge unidock          # demo 3 的 GPU 路径
```

没有 GPU 的机器也能跑，把下面命令里的 `--device cuda` 换成 `--device cpu` 即可，速度慢一些。

### 0.2 依次跑五个环节

```bash
# 环节 0：确认能看到 GPU、torch 能用
bash demo_gpu_env/check_gpu.sh

# 环节 1：相似性检索，输出 EF1% 与富集曲线（约 4 秒）
bash demo_similarity/run_demo.sh --device cuda

# 环节 2：机器学习打分，输出随机森林与 MLP 的对照表（约 20 秒）
bash demo_ml_screen/run_demo.sh --device cuda

# 环节 3：GPU 对接，Uni-Dock 10 个配体 + vina CPU 5 个对照（约 1 分钟）
bash demo_docking/run_demo.sh --device cuda

# 环节 4：逐级筛选，读仓库里算好的 1,981 个分子分数表（约 2 秒）
bash demo_pipeline/run_demo.sh
```

### 0.3 每个环节的产物

| 环节 | 输出去向 | 关键文件 |
| --- | --- | --- |
| 0 | 屏幕 | 显卡列表、torch 自检结果 |
| 1 | `demo_similarity/results/` | `metrics.json`、`top_hits.csv`、`enrichment_curve.png` |
| 2 | `demo_ml_screen/results/` | `comparison_table.csv`、`enrichment_ml.png` |
| 3 | `demo_docking/results/` | `docking_scores.csv`、`compare_cpu_gpu.csv`、`timing.json`、`poses/` |
| 4 | `demo_pipeline/results/` | `strategy_comparison.csv`、`pipeline_summary.json`、`pipeline_plot.png` |

### 0.4 需要重新算数据时

仓库里的 `data/` 已含 DUD-E 原始文件与配体构象，正常不用重跑。要换靶点或重建时按顺序：

```bash
bash data/download_data.sh                    # 下载并解压 DUD-E（约 66 MB）
python common/prepare_data.py                 # 建库、划分、对接盒子
python common/ligand_prep.py                  # 配体 3D 构象与 pdbqt
bash common/prepare_receptor.sh               # 受体 pdbqt

# 重跑 demo 4 的全库对接分数（GPU，约 2 分钟）
python demo_pipeline/precompute_docking.py --method unidock --search-mode fast --chunk 500 --unidock-mem-mb 3000
```

## 1. demo 清单与一行命令

| # | 目录 | 方法 | 一行命令 | 课堂限时 |
| --- | --- | --- | --- | --- |
| 0 | `demo_gpu_env/` | 环境自检（学生第一件事） | `bash demo_gpu_env/run_demo.sh 2023012345` | 5 分钟 |
| 1 | `demo_similarity/` | ECFP4 + Tanimoto 相似性检索，EF/ROC/富集曲线（GPU 批量 Tanimoto） | `bash demo_similarity/run_demo.sh --device cuda` | 8 分钟 |
| 2 | `demo_ml_screen/` | ECFP4 + 随机森林(CPU) / MLP(GPU)，与 demo 1 同表对照 | `bash demo_ml_screen/run_demo.sh --device cuda` | 8 分钟 |
| 3 | `demo_docking/` | Uni-Dock GPU 批量对接 10 个配体 + vina CPU 对照 5 个 | `bash demo_docking/run_demo.sh --device cuda` | 10 分钟 |
| 4 | `demo_pipeline/` | 2D 粗筛 top 10% → 对接精排，与全库对接比富集与成本 | `bash demo_pipeline/run_demo.sh` | 8 分钟 |

gpu1 上先设共享环境的 python：

```bash
export VS_PY=/public/AI药学实践/demo_vs/envs/vs/bin/python   # setup_gpu_env.sh 打印的实际路径
eval "$(bash demos/demo_vs/demo_gpu_env/gpu_assign.sh 2023012345 --export)"
bash demos/demo_vs/demo_gpu_env/check_gpu.sh
```

全部 demo 支持 `--device auto|cuda|cpu`：`auto` 在无 GPU 时自动回退 CPU 并打印原因，
`--require-gpu` 会在没有 GPU 时直接报错（课堂上防止学生跑错设备）。

## 2. 数据（DUD-E DRD3，已缓存）

- 来源：`https://dude.docking.org/targets/drd3/drd3.tar.gz`（66 MB，缓存于 `data/cache/drd3.tar.gz`，不重复下载）。
- 活性分子 480 个，诱饵 34,050 个（去重后 34,037）。
- **已知坑**：官方 tar 包里 `receptor.pdb` 与 `crystal_ligand.mol2` 是指向 `docking/` 的坏符号链接，
  必须从 dude.docking.org 单独下载。`data/download_data.sh` 已处理，下载后自检。
- 筛库：`data/drd3_library_10k.csv` = 480 actives + 9,520 decoys（seed 42），demo 1/2 用。
- 对接库：`data/drd3_dock_library.csv` = 480 actives + 1,520 decoys（从上面再抽，seed 7），demo 3/4 用。
- 划分：`data/drd3_split.json`，scaffold 划分 8,000/2,000（测试侧 96 个活性，基准率 4.8%），另有随机划分对照。
- 盒子：`data/drd3_box.json`，晶体配体质心 (8.478, 21.580, 23.906)，20×20×20 Å。
- 配体 3D 构象与 pdbqt：`data/dock_ligands/`（1,981/2,000 个准备成功，其余 19 个是互变异构体不一致，已排除并记录）。

## 3. 预计算清单（需要重跑时用）

| 步骤 | 命令 | 实测耗时（本机） | 产物 |
| --- | --- | --- | --- |
| 下载数据 | `bash data/download_data.sh` | 12 s（缓存后 0 s） | `data/drd3/` |
| 建库与划分 | `../venv_chemprop/bin/python common/prepare_data.py` | 17 s | `data/drd3_library_10k.csv` 等 |
| 配体 3D + pdbqt | `../venv_chemprop/bin/python common/ligand_prep.py` | 28 s（2,000 个，24 线程） | `data/dock_ligands/` |
| 受体 pdbqt | `bash common/prepare_receptor.sh` | 0.5 s | `data/drd3/receptor.pdbqt` |
| 全库对接预计算（CPU 退路） | `../venv_chemprop/bin/python demo_pipeline/precompute_docking.py --method vina --workers 20 --cpu 3 --chunk 100` | 本机实测约 2 小时（负载高时 1.2～3.3 小时） | `demo_pipeline/results/full_docking.csv` |
| 全库对接预计算（推荐，GPU） | `$VS_PY demo_pipeline/precompute_docking.py --method unidock --search-mode fast --chunk 500` | 本机 A800 实测 Uni-Dock 约 0.45 秒/配体，2,000 个约 15 分钟 | 同上 |
| gpu1 共享环境 | `bash setup_gpu_env.sh`（在 gpu1 上执行一次） | 10～20 分钟 | `/public/.../envs/vs` |

预计算脚本支持断点续跑与增量落盘：每跑完 100 个配体写一次 CSV，学生不必等全部跑完；
配体顺序已打乱，已完成部分里活性与诱饵比例接近全库，可以直接做 demo 4。

## 4. 课堂安排（50 分钟）

| 时间 | 内容 | 谁做 |
| --- | --- | --- |
| 0–5 min | 开场：8 卡 50 人怎么分（`gpu_assign.sh --table` 投屏）；学生分卡并跑 `check_gpu.sh` | 学生 |
| 5–13 min | demo 1：ECFP4 + Tanimoto。先 CPU 跑一遍，再 `--device cuda` 跑，对比 `timing_s` 与加速比 | 学生 |
| 13–21 min | demo 2：随机森林 vs MLP，看 `comparison_table.csv`；讲 DUD-E 诱饵偏倚 | 学生 |
| 21–31 min | demo 3：Uni-Dock fast 模式跑 10 个配体，再看 vina CPU 对照 5 个，比单配体耗时 | 学生 |
| 31–39 min | demo 4：粗筛 top 10% 与全库对接的富集、成本对比 | 学生 |
| 39–44 min | 讲评：EF1% 饱和、DUD-E analog bias、对接打分重叠、GPU 使用规范 | 教师 |
| 44–50 min | 自由尝试：改参数（`--topk`、`--epochs`、`--unidock-mode`、盒子大小） | 学生 |

分组错峰建议：同一张卡上的 6～7 人按学号尾号错开 1 分钟启动，避免 8 张卡同时满负荷。
每组指定 1 名组长，负责在组内跑一次并把 `results/` 里的数字投屏对照。

## 5. gpu1 上 50 人分卡方案（8 × RTX 3090 24 GB）

### 5.1 分卡

```bash
# 学号 % 8 得到 GPU_ID，组号 = GPU_ID + 1
eval "$(bash demo_gpu_env/gpu_assign.sh 2023012345 --export)"
bash demo_gpu_env/check_gpu.sh
```

- 50 人：卡 0、1 各 7 人，卡 2～7 各 6 人。
- `CUDA_VISIBLE_DEVICES` 只影响之后启动的进程；设完再启动程序。
- 不要写进 `~/.bashrc`，换机器会指到不存在的卡。

### 5.2 每 demo 的显存与耗时预算（按 6～7 人/卡估算）

| demo | 单进程显存 | 同卡 7 人合计 | 单次耗时（3090，预计） | 说明 |
| --- | --- | --- | --- | --- |
| 0 环境自检 | <0.2 GB | <1.5 GB | 2 s | 矩阵乘自检很小 |
| 1 Tanimoto | <0.3 GB | <2.1 GB | 指纹 CPU 3 s + GPU Tanimoto <1 s | 指纹矩阵 82 MB，分块后峰值约 0.25 GB |
| 2 MLP 训练 | <0.3 GB | <2.1 GB | 30 epochs 约 1～3 s | 模型约 1.1 M 参数 |
| 3 Uni-Dock（fast，10 配体） | **限 2.5 GB/人** | 17.5 GB | 预计数秒到数十秒/配体 | 必须加 `--max_gpu_memory 2500`，否则一人占满 24 GB |
| 4 融合流水线 | 0（纯 CPU） | — | <30 s | 只读预计算 CSV |

结论：24 GB 卡上 7 人 × 2.5 GB = 17.5 GB，留 6.5 GB 余量，可行。
**Uni-Dock 默认 `--max_gpu_memory 0`（占满显存），不加限制一定 OOM，这是本 demo 最需要注意的一点。**

### 5.3 错峰与排队

- 按组错峰：第 1 组 0 秒启动，第 2 组 1 分钟后，依次错开。同一张卡上 7 人不要同时按下回车。
- demo 3 最重：同一张卡分两批跑，每批 3～4 人，另一批先做 demo 4（纯 CPU，不占显存），再交换。
- 每 demo 严格限时（见第 4 节表），教师用投屏计时；跑不完的组课后补，不占用下一段。
- 若某张卡排队过长：允许临时换到空闲卡（先 `nvidia-smi` 看），但要 `export CUDA_VISIBLE_DEVICES=<新卡>` 重跑，不要两人用同一张卡硬挤。
- 不建议把 50 人分到 8 张卡的 CPU 上跑 vina：每人 2～4 线程 × 50 人会严重超卖，CPU 版只用于 demo 1/2 的对照和学生本机回退。

### 5.4 看谁在跑、OOM 了怎么办

```bash
nvidia-smi                                        # 8 张卡总览：显存、利用率、进程 PID
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # 谁占了多少显存
nvidia-smi -i 3                                   # 只看 3 号卡
```

- OOM（`torch.OutOfMemoryError` 或 Uni-Dock 的 CUDA error）：
  1. 先 `nvidia-smi` 看这张卡还剩多少，确认是不是同卡同学占满；
  2. 减小 `--max_gpu_memory`（如 1500）、减小 batch size；
  3. 仍不行就等 1～2 分钟或换卡，不要连续重试把卡打满。
- 进程卡死：`nvidia-smi` 找到 PID，`kill <PID>`；不要 `nvidia-smi --gpu-reset`（需要 root，且会影响同卡所有人）。

## 6. 实测数字汇总（开发机，CPU 与 GPU 均已实测）

| 指标 | 实测值 |
| --- | --- |
| DUD-E DRD3 actives / decoys | 480 / 34,050（去重 34,037） |
| 筛选库规模 | 10,000（480 + 9,520） |
| ECFP4 指纹（10,000 分子） | 2.9～3.3 s |
| Tanimoto CPU 打分（10,000 × 10） | 0.105 s |
| demo 1 全流程（CPU） | 6.3 s |
| demo 1 测试集 ROC-AUC | 0.960 |
| demo 1 测试集 EF1% / EF5% | 20.83 / 13.96 |
| demo 2 对照表最优（RF，scaffold 划分） | ROC-AUC 0.9997，EF5% 19.58，top-100 命中 94/96 |
| demo 2 全流程（CPU，两个划分 × 两个模型） | 24.3 s |
| 随机森林训练（8,000 × 2048，300 棵树，4 线程） | 3.5 s |
| MLP 训练（30 epochs，CPU，4 线程） | 5.7 s |
| vina 单配体（cpu 4，exhaustiveness 4，20³ 盒） | 12.8 s |
| vina 单配体（cpu 2，exhaustiveness 4） | 约 15～20 s |
| 受体 pdbqt 准备 | 0.52 s |
| obabel 配体转 pdbqt | 约 46 ms/个 |
| 红接验证（晶体配体重对接） | 最优 pose -7.36 kcal/mol，重原子平均最近距离 0.80 Å |
| demo 4 预计算（1,981 个，20 进程 × 3 线程） | 本机约 2 小时（10～27 个/分钟）；gpu1 Uni-Dock 约 10 分钟 |

## 7. 目录结构

```
demos/demo_vs/
├── README.md                 本文件
├── setup_gpu_env.sh          gpu1 共享环境一次性安装 + 自检
├── common/
│   ├── vs_common.py          数据、指纹、Tanimoto（CPU/GPU）、EF/ROC、划分、设备选择
│   ├── prepare_data.py       建库、划分、对接盒子
│   ├── ligand_prep.py        DUD-E 3D mol2 -> pdbqt（含连通性核对）
│   ├── prepare_receptor.sh   受体 pdb -> pdbqt
│   └── dock_run.py           vina 多进程 / Uni-Dock 批量对接与输出解析
├── data/
│   ├── download_data.sh      下载（缓存）与解压、坏符号链接修复
│   ├── cache/                原始 tar 包缓存
│   ├── drd3/                 原始文件 + 受体 pdbqt
│   ├── drd3_library_10k.csv  筛选库（demo 1/2）
│   ├── drd3_dock_library.csv 对接库（demo 3/4）
│   ├── drd3_split.json       划分与参考活性分子
│   ├── drd3_box.json         对接盒子
│   ├── demo3_ligands.csv     demo 3 的 10 个配体
│   └── dock_ligands/         3D mol2 与 pdbqt
├── demo_gpu_env/             环境自检、分卡（check_gpu.sh / gpu_assign.sh / run_demo.sh）
├── demo_similarity/          demo 1
├── demo_ml_screen/           demo 2
├── demo_docking/             demo 3
└── demo_pipeline/            demo 4（含教师预计算脚本与学生分析脚本）
```

## 8. 首次在 gpu1 上运行时的检查清单（本机 A800 已跑通）

1. `nvidia-smi` 正常、`torch.cuda.is_available()` 为 True（`demo_gpu_env/check_gpu.sh`）。
2. `setup_gpu_env.sh` 装出的环境里 `torch.version.cuda` 与驱动匹配；`unidock --version` 可执行。
3. demo 1 GPU Tanimoto 峰值显存 <0.5 GB，与 CPU 结果最大误差 <1e-5（脚本自动打印）。
4. demo 2 MLP 单轮耗时与峰值显存（脚本写入 `metrics.json`）。
5. demo 3 `--gpu_batch` 一次性传入 10 个配体，输出目录出现 10 个 `*_out.pdbqt`，`docking_scores.csv` 能解析到分数。
6. `CUDA_VISIBLE_DEVICES` 分卡生效；`--max_gpu_memory 2500` 下 3 人同卡并发不 OOM。
7. vina 与 Uni-Dock 同一配体的打分差（`compare_cpu_gpu.csv`）：fast 模式搜索更充分，分数通常更负，属正常。

## 9. 说明

- `demos/venv_chemprop` 是开发机现成的 CPU 环境（rdkit 2025.03.3、torch 2.3.1+cu121、chemprop 1.7.1）。
  gpu1 上不要用它：按 `setup_gpu_env.sh` 装到 `/public` 共享路径，学生统一 source。
- 开发机上的 `.tools/`（约 3.5 GB）只是本机验证 Uni-Dock 命令行与输出格式用的临时 conda 环境，
  **不是交付物，可以整个删掉**，不影响 `/public` 共享环境方案与任何脚本。
- 不要改 `demos/demo_esol`、`demos/demo_admet`、`ppt/`、`docs/` 下的内容。
