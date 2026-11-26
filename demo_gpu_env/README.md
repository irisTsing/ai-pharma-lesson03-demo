# demo 0：环境自检（学生上机第一件事）

## 1. 这个 demo 解决什么问题

确认当前机器能看到 GPU、PyTorch 能用 GPU。看不到卡时，后面的 demo 要么跑不动，要么会悄悄退回 CPU。

## 2. 运行

```bash
bash demo_gpu_env/run_demo.sh
```

等价于直接运行 `check_gpu.sh`。想看某一张卡时：

```bash
CUDA_VISIBLE_DEVICES=2 bash demo_gpu_env/check_gpu.sh
```

## 3. 输出与判断

- 显卡列表：型号与总显存，是 RTX 3090 时显示 24 GB 左右。
- 显存占用：看当前谁在跑、还剩多少。
- PyTorch 自检：torch 版本、`cuda.is_available()`、设备名、一次矩阵乘。

`cuda.is_available()` 为 True 才算通过。为 False 时依次检查：是否在 GPU 节点、`nvidia-smi` 是否正常、装的是不是 CPU 版 torch。

## 4. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 找不到 nvidia-smi | 不在 GPU 节点，或驱动没装 | 确认已连到 gpu1 |
| cuda.is_available 为 False | 装了 CPU 版 torch | 重装 CUDA 版 torch |
| 显存快满了 | 同卡别人在跑 | 看 `nvidia-smi`，等一会儿或换卡 |
