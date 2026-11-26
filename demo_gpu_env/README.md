# demo 0：gpu1 环境与分卡（学生上机第一件事）

## 1. 这个 demo 解决什么问题

50 名学生共用 8 张 RTX 3090，第一步不是写代码，而是搞清楚三件事：

1. 自己在哪张卡上跑（`CUDA_VISIBLE_DEVICES`）；
2. torch 能不能看到 GPU（避免跑到一半才发现装的是 CPU 版）；
3. 这张卡现在还有多少显存（同卡 6～7 人，先看再跑）。

三件事对应三个脚本，一条命令即可完成。

## 2. 用法

```bash
cd demos/demo_vs

# 按学号分卡：GPU_ID = 学号 % 8，组号 = GPU_ID + 1
bash demo_gpu_env/gpu_assign.sh 2023012345
eval "$(bash demo_gpu_env/gpu_assign.sh 2023012345 --export)"

# 一条命令：分卡 + 自检（推荐发给学生）
bash demo_gpu_env/run_demo.sh 2023012345

# 只看分组表（可投屏）
bash demo_gpu_env/gpu_assign.sh --table
```

`gpu_assign.sh` 输出示例：

```
学号：2023012345
GPU_ID = 2023012345 % 8 = 1
组号：2
执行下面这一行：
  export CUDA_VISIBLE_DEVICES=1
```

`check_gpu.sh` 依次检查：`nvidia-smi` 驱动与 8 张卡 → 每张卡的显存占用与占用进程 →
torch 的 `cuda.is_available()`、设备名、显存、compute capability，并做一次 1024×1024 矩阵乘。
全部通过打印“GPU 环境正常”，否则给出具体排查项（不在 GPU 节点、驱动未加载、torch 是 CPU 版、
`CUDA_VISIBLE_DEVICES` 指到不存在的卡），并以非零码退出。

## 3. 环境变量

| 变量 | 作用 | 例子 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | 只让本进程看到指定的卡 | `export CUDA_VISIBLE_DEVICES=3` |
| `VS_PY` | 指定共享环境的 python | `export VS_PY=/public/AI药学实践/demo_vs/envs/vs/bin/python` |
| `N_GPU` | 卡数，默认 8 | `N_GPU=8` |

## 4. 实测（开发机，无 NVIDIA 驱动）

- `gpu_assign.sh --table`、`gpu_assign.sh 2023012345`、`--export` 均正常，分卡结果可复现。
- `check_gpu.sh` 在无驱动机器上按设计失败：nvidia-smi 报驱动错误，torch 报 `cuda.is_available()=False`，
  脚本给出排查清单并以退出码 1 结束，不会静默继续。
- 8 张卡、50 人的分卡结果为：卡 0、1 各 7 人，卡 2～7 各 6 人（脚本自动算出）。

## 5. 常见坑

- `CUDA_VISIBLE_DEVICES` 只影响**之后启动**的进程。设完变量再启动 python；已经跑起来的进程不会改。
- 设成 `CUDA_VISIBLE_DEVICES=1` 后，程序里看到的卡编号变成 `cuda:0`（重新编号），这是正常的。
- 分卡脚本按学号取模，同组同学一定在同一张卡上，组内要错峰启动（见顶层 README 的时间表）。
- 不要在 `~/.bashrc` 里写死某张卡：换机器或换账号后会指到不存在的卡，程序直接报 CUDA 错误。
- `nvidia-smi` 能看到其他人进程，但看不到他们的命令行；用 `--query-compute-apps` 只看显存占用。
- 显存不足（OOM）时先 `nvidia-smi` 看这张卡剩多少，再减小 batch size 或 `--max_gpu_memory`，不要反复重试。
