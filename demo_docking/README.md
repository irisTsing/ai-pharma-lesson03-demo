# demo 3：基于结构的药物筛选——GPU 批量对接与 CPU 对照

## 1. 对应方法与文献

- 对接引擎（课堂主用）：**Uni-Dock**，GPU 批量对接，Apache 2.0。
  - Yu et al., *Uni-Dock: GPU-Accelerated Docking Enables Ultralarge Virtual Screening*, J. Chem. Theory Comput. 2023, 19, 3336–3345。DOI：10.1021/acs.jctc.2c01145。
  - 官方仓库：<https://github.com/dptech-corp/Uni-Dock>
- 对照引擎（CPU 参考实现）：**AutoDock Vina 1.2.3**。
  - Eberhardt et al., *AutoDock Vina 1.2.0*, J. Chem. Inf. Model. 2021, 61, 3891–3898。DOI：10.1021/acs.jcim.1c00203。
- 数据：DUD-E DRD3，受体 `receptor.pdb`（去水），晶体配体 `crystal_ligand.mol2`（3PBL）。
- 对接盒子：晶体配体质心 (8.478, 21.580, 23.906)，20 × 20 × 20 Å，网格间距 0.375 Å（vina 默认）。

## 2. 流程

1. 受体准备：`obabel receptor.pdb -O receptor.pdbqt -xr -p 7.4`
   —— `-xr` 只保留刚性受体原子，`-p 7.4` 按生理 pH 加氢。注意 obabel 2.4 的选项必须写在输入文件之后。
2. 配体准备：DUD-E 附带的 3D mol2 → `obabel lig.mol2 -O lig.pdbqt -p 7.4`（10 个配体）。
3. GPU 对接：一次调用把 10 个配体交给 Uni-Dock：
   `unidock --receptor receptor.pdbqt --gpu_batch lig1.pdbqt ... --search_mode fast --scoring vina --center_x ... --size_* 20 --num_modes 1 --max_gpu_memory 2500 --dir out`
4. CPU 对照：同一批里取前 5 个配体，用 `vina --cpu 2 --exhaustiveness 4` 跑同一个盒子。
5. 输出：打分表、CPU/GPU 对照表、每个配体的最优 pose（pdbqt + pdb）、单配体耗时与加速比。

## 3. 运行

```bash
cd demos/demo_vs

# GPU 版（gpu1）
bash demo_docking/run_demo.sh --device cuda

# CPU 回退（开发机 / GPU 出问题时）
bash demo_docking/run_demo.sh --device cpu
```

直接调用：

```bash
../venv_chemprop/bin/python demo_docking/run_docking.py --device auto \
    --n-ligands 10 --vina-n 5 --vina-cpu 2 --vina-exhaustiveness 4 \
    --unidock-mode fast --unidock-mem-mb 2500
```

输入：`data/demo3_ligands.csv`（5 个活性 + 5 个诱饵，交替排列，前 5 个用于 CPU 对照）、
`data/drd3/receptor.pdb`、`data/dock_ligands/mol2/*.mol2`、`data/drd3_box.json`。
输出：`demo_docking/results/` 下的 `docking_scores.csv`、`compare_cpu_gpu.csv`、`timing.json`、`poses/`。

## 4. 实测结果

### 4.1 受体与配体准备（开发机实测）

| 步骤 | 实测 |
| --- | --- |
| 受体 pdb → pdbqt（obabel -xr -p 7.4） | 0.52 s，2,549 个原子（其中极性氢 431），无 HETATM 残留 |
| 10 个配体 mol2 → pdbqt（obabel -p 7.4） | 0.46 s（约 46 ms/个） |
| 盒子口袋检查 | 盒心到受体最近原子 3.29 Å，10 Å 内 177 个受体原子 |

### 4.2 红接（redocking）验证：受体与盒子是否可用

在讲对接结果之前必须先验证协议：把 DUD-E 晶体配体从 `crystal_ligand.mol2` 重新对接到同一个盒子里。

- 命令：`vina --receptor receptor.pdbqt --ligand crystal_ligand.pdbqt --cpu 4 --exhaustiveness 8 --seed 42 --out xtal_out.pdbqt`（盒子同 demo）。
- 结果：最优 pose 打分 -7.36 kcal/mol；与晶体构象的重原子平均最近距离 **0.80 Å**，
  结合模式基本复现（按原子顺序算的 RMSD 是 4.70 Å，差异来自苯环对称原子的对应关系，不是结合模式错）。
- 结论：受体 pdbqt 与 20³ 盒子可用，对接流程可信。注意该分数（-7.36）比 demo 3 的多数配体更“弱”，
  说明 vina 打分跨配体不可直接比较，只能用于同一次筛选内的排序。

### 4.3 vina CPU（AutoDock Vina 1.2.3，本机，20³ 盒）

| 设置 | 单配体耗时 | 备注 |
| --- | --- | --- |
| `--cpu 4 --exhaustiveness 4`，12 个配体并行 6 进程 | 12.8 s/配体 | 36 个配体/分钟吞吐（空载时） |
| `--cpu 2 --exhaustiveness 4`，10 个配体并行 4 进程 | 15～45 s/配体 | 课堂对照用；本机预计算并发时实测 44 s/配体，全流程 87.5 s |
| `--cpu 4 --exhaustiveness 8` | 21～38 s/配体 | 机器共享时波动大 |
| `--cpu 2 --exhaustiveness 8` | 约 67 s/配体 | 不建议课堂用 |

打分（`--cpu 4 --exhaustiveness 4`，10 个配体；另一轮 `--cpu 2` 复跑结果一致，差值 <0.05 kcal/mol）：

| 配体 | 类型 | affinity (kcal/mol) |
| --- | --- | --- |
| C39218205 | decoy | -9.76 |
| CHEMBL399164 | active | -9.37 |
| CHEMBL8514 | active | -9.29 |
| CHEMBL519240 | active | -8.86 |
| C66945038 | decoy | -8.67 |
| C13148479 | decoy | -8.55 |
| C25534482 | decoy | -8.35 |
| CHEMBL286252 | active | -8.10 |
| C07778731 | decoy | -7.46 |
| C63531067 | decoy | -7.08 |

5 个活性平均约 -8.9 kcal/mol，5 个诱饵平均约 -8.3 kcal/mol，排序上有区分但**重叠明显**。
课堂上要讲清楚：在这个小样本上，对接打分不能可靠区分活性与诱饵，
这正是 demo 4 用 1,000～2,000 个分子做富集统计的原因。

### 4.4 Uni-Dock GPU（命令与输出已在开发机核实，运行待 gpu1 验证）

开发机没有 NVIDIA 驱动，无法实际跑 GPU。已核实的内容：

- conda-forge 包为 **Uni-Dock v1.2.0**（`conda create -n unidock_env unidock -c conda-forge`）；
- `--gpu_batch` 接受多个配体文件，PDBQT 与 SDF 都支持（`--help` 原文：`gpu batch ligand (PDBQT or SDF)`）；
- 输出文件为 `<配体名>_out.pdbqt`（PDBQT 输入）或 `<配体名>_out.sdf`（SDF 输入）；
  SDF 里的打分写在 `> <Uni-Dock RESULT>` 属性下的 `ENERGY=` 行，PDBQT 里是 `REMARK VINA RESULT:`，解析脚本两种都支持；
- 默认占满可用显存，必须加 `--max_gpu_memory` 限制（MB）。**50 人共用 8 张卡，每人 2,500～3,000 MB**；
- 无 GPU 时报 `cudaErrorNoDevice`，脚本据此提示改用 `--device cpu`。
- 单配体耗时：官方（JCTC 2023）报告的加速是相对 CPU vina 一到两个数量级，具体数值随配体大小、
  批大小、搜索模式变化。本课程不写死预计值，`run_docking.py` 会打印真实
  `wall_per_lig_s`（fast 模式）与相对 vina 的加速比，跑完第一组后把数字填回本节。

### 4.5 待 gpu1 首次验证的清单

1. `unidock --version` 能正常执行，且 CUDA 运行时与驱动匹配（conda 包为 CUDA 12.x 构建）。
2. `--gpu_batch` 一次传 10 个 pdbqt：输出目录里是否生成 10 个 `*_out.pdbqt`，`parse_sdf_affinity`/`parse_pose_affinity` 能否读到分数。
3. `nvidia-smi` 里能看到自己的进程，显存占用在限制范围内。
4. `--max_gpu_memory 2500` 下 3 人同卡并发是否稳定、是否 OOM。
5. vina 与 Uni-Dock 对同一配体的打分差（`compare_cpu_gpu.csv` 的 `delta_kcal`）：fast 模式搜索更充分，
   分数通常会比 `--exhaustiveness 4` 更低（更负），属正常现象，比较时要说清协议不同。
6. 两个配体若出现正分（本机遇到 1 例，源于 obabel 生成的扭转树对个别稠环配体不稳健），
   `docking_scores.csv` 的 `invalid_positive_score` 会标出，课堂跳过该配体并解释原因。

## 5. 课堂上可以改的参数

- `--n-ligands 10`：配体数，默认 10 个。
- `--unidock-mode fast|balance|detail`：搜索强度与耗时同步上升，fast 是课堂默认。
- `--unidock-mem-mb 2500`：显存上限，同卡人多时减小（如 1500）。
- `--vina-n 5`、`--vina-cpu 2`、`--vina-exhaustiveness 4`：CPU 对照的规模与预算。
- 盒子：改 `data/drd3_box.json` 的 `size` 为 22 或 24，看打分与耗时如何变。
- 换靶点：`TARGET=akr1c3 bash data/download_data.sh` 后改 `data/drd3_box.json` 对应文件与配体清单。

## 6. 常见坑

- **DUD-E tar 包里的 receptor.pdb / crystal_ligand.mol2 是坏符号链接**，必须单独从网站下载，否则 obabel 报“找不到文件”。
- obabel 2.4 的 `-xr`、`-p` 必须写在输入文件之后，写前面会报 `No input file or format spec`。
- vina 1.2.3 没有 `--log` 参数，要 `> vina.log` 重定向；`--batch` 选项要求特殊格式的多模型 PDBQT，本 demo 不用，改多进程并行。
- 受体 pdbqt 里 `A` 是芳香碳、`NA` 是氮受体，不是钙和钠，看原子类型统计时别误判。
- 个别配体的 obabel pdbqt 扭转树不稳健，会出现正分（本机 1/12）。脚本标记 `invalid_positive_score`，不要把它当成强结合。
- GPU 上跑之前一定先 `bash demo_gpu_env/check_gpu.sh`，确认 `CUDA_VISIBLE_DEVICES` 与 `torch.cuda.is_available()`。
- 同一张卡 6～7 人时，Uni-Dock 不加 `--max_gpu_memory` 会占满 24 GB，导致其他人 OOM。
