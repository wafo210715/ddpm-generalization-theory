# Progress 2026-05-23: WSL2 环境与双 3090 开发方案

> **当前分支: `wafo/5090-setup`**
> **目标硬件: NVIDIA RTX Pro 6000 Blackwell — 96GB 统一显存, ~20k CUDA cores**
> **替代硬件（当前）: Dual RTX 3090 — 2×24GB, NVLink, Ampere 架构**
>
> ⚠️ 所有环境配置、代码修改、性能估算均以 RTX Pro 6000 Blackwell 为最终目标。
>    Dual 3090 相关讨论仅作为过渡参考和性能对比基准。

## 一、WSL2 基础概念

### Windows 上的多个 Linux 发行版

- `wsl --list --verbose` 查看已安装的发行版
- 带 `*` 的是默认发行版（PowerShell 输入 `wsl` 时进入的）
- 每个发行版是独立的 Linux 实例，各有独立文件系统和包管理
- 例如 "Ubuntu"（默认，可能 22.04）和 "Ubuntu-24.04" 是两个独立环境

### VSCode WSL 集成

- VSCode 选 "Ubuntu (WSL)" = 自动进入指定发行版的 Linux shell
- 等同于 PowerShell 输入 `wsl`，但 VSCode 会自动安装 VSCode Server 到对应发行版
- 终端是货真价实的 Linux shell（bash/zsh），支持 apt install、git、python 等一切 Linux 程序
- 文件系统是 `/home/xxx/`，不是 `C:\Users\xxx\`

### WSL2 本质

- WSL2 是运行在 Windows 上的轻量级虚拟机
- 与 Windows 共享网络、文件系统（通过 `/mnt/c/` 访问 Windows 磁盘）
- 几乎零开销
- 可以完全当一台远程 Linux 服务器来使用

---

## 二、SSH 连接 WSL

### 方案一：直接 SSH 到 WSL（推荐）

```bash
# 1. 在 WSL 里装 SSH server
sudo apt update
sudo apt install openssh-server -y
sudo service ssh start
sudo service ssh status

# 2. 查看 WSL IP
hostname -I
# 例如: 172.x.x.x
```

注意：WSL2 每次重启 IP 会变。

### 固定端口转发（稳定方案）

```powershell
# PowerShell 管理员，把 WSL 的 22 端口转发到 Windows 的 2222
netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=22 connectaddress=$(wsl hostname -I)
```

然后从 Mac：`ssh your_username@windows_ip -p 2222`

### 方案二：SSH 到 Windows 本身（不推荐用于开发）

```powershell
# PowerShell 管理员
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```

**推荐：直接 SSH 到 WSL，不要 SSH 到 Windows 再进 WSL。**

---

## 三、WSL2 + Dual RTX 3090 CUDA 开发

### GPU 支持情况

- CUDA 完整支持（NVIDIA 官方支持，CUDA 11.4+）
- PyTorch / TensorFlow 官方支持 WSL2
- Dual 3090 正常可见，`nvidia-smi` 正常工作
- 性能接近原生 Windows，某些场景甚至更快

### WSL2 vs Native Windows 跑深度学习

| | WSL2 | Native Windows |
|---|---|---|
| CUDA 支持 | 完整 | 完整 |
| Docker | 原生 Docker | 需要 Docker Desktop / WSL backend |
| 包管理 | apt, pip 一切正常 | 混合环境，经常出兼容问题 |
| 文件系统 I/O | Linux fs 快，`/mnt/c/` 慢 | 原生 NTFS |
| conda/pip | Linux 生态，无坑 | 经常遇到编译问题 |
| 论文代码兼容性 | 和 Linux 服务器完全一致 | 和服务器有差异 |

### 快速搭建环境

```bash
sudo apt update
conda create -n ddpm python=3.10
conda activate ddpm
pip install torch --index-url https://download.pytorch.org/whl/cu121
nvidia-smi  # 确认两张 3090 可见
```

### 关键注意事项

**文件必须放在 Linux 文件系统里，不要放在 `/mnt/c/` 下。**

```
快: /home/you/projects/ddpm-generalization-theory/
慢: /mnt/c/sd/ddpm-generalization-theory/
```

跨文件系统 I/O 走 9P 协议，速度显著下降。

---

## 四、Windows 与 WSL 之间传输大文件（数据集）

### 方案一：符号链接（不占空间，读取稍慢）

```bash
ln -s /mnt/c/sd/datasets ~/datasets
```

- 不占用 Linux 磁盘空间
- 读取走 9P 协议，比纯 Linux fs 慢
- 但训练瓶颈在 GPU 计算，不在磁盘 I/O，实际差距不大
- 适合数据不常变、Linux 磁盘空间不够时

### 方案二：robocopy（一次性迁移，最快）

```powershell
# PowerShell 里，通过 \\wsl$\ 直接写入 WSL 文件系统
robocopy C:\sd\datasets \\wsl$\Ubuntu\home\you\datasets /E /MT:32
```

- 利用 Windows 的 SMB 共享访问 WSL 文件系统
- 多线程复制，速度最快
- 适合数据量大的一次性迁移

### 方案三：tar pipe（中等文件推荐）

```bash
# 在 WSL 里，从 Windows 磁盘打包到 Linux 磁盘
cd /mnt/c/sd/datasets
tar cf - . | (cd ~/datasets && tar xf -)
```

- 比 cp 略快（减少系统调用开销）
- 适合中等大小数据集

### 方案四：直接 cp

```bash
cp -r /mnt/c/sd/datasets/ ~/datasets/
```

- 简单但慢（走 9P 协议）
- 100GB 预计 20-30 分钟
- 适合小数据集（<10GB）

### 选择建议

| 方案 | 适合场景 | 速度 |
|---|---|---|
| symlink | 数据不常变，空间不够时 | 读取稍慢，训练瓶颈不在 I/O |
| robocopy `\\wsl$\` | 一次性迁移，数据量大 | 最快 |
| tar pipe | 中等数据集 | 中等 |
| cp | 小数据（<10GB） | 慢但不用想 |

如果 Linux 磁盘空间够，直接 robocopy 过去一份最快。

---

## 五、WSL2 磁盘空间管理

### 查看已用空间

```powershell
# 查看 vhdx 文件路径和状态
wsl --list --verbose

# 查看 vhdx 实际占用
(Get-Item "C:\Users\wafo2\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu*\LocalState\ext4.vhdx").Length / 1GB
```

```bash
# 在 WSL 里查看
df -h /
```

### 磁盘空间特点

- **默认无上限** — vhdx 自动增长，直到填满 Windows 磁盘
- **不会自动缩小** — 删文件后 vhdx 不自动回收空间
- 限制 = Windows 磁盘剩余空间

### 手动压缩 vhdx（回收空间）

```powershell
# 先关闭 WSL
wsl --shutdown

# 压缩
diskpart
# diskpart 里：
select vdisk file="C:\Users\wafo2\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu...\LocalState\ext4.vhdx"
compact vdisk
exit
```

---

## 六、Dual 3090 vs RTX Pro 6000 Blackwell 硬件选型讨论

### 讨论起源

问题：双 3090 通过 NVLink 跑 DDPM，CUDA cores 能相加吗？对比 RTX Pro 6000 Blackwell (~20k cores) 如何？

### 推理过程

1. **初始错误判断**：我最初说 NVLink 不合并算力，双卡通信开销大，实际效率只有 ~80%。**这个判断过于保守。**

2. **用户纠正**：单卡 10k cores，双卡不就是 2x speed 吗？——这个直觉是对的。对于 DataParallel 训练（DDPM 就是这种），NVLink 带宽（100 GB/s）足够让梯度同步开销很小，**双 3090 实际吞吐量确实接近 2x 单卡**。

3. **重新对比**：
   - 双 3090：2 × 10,496 Ampere cores ≈ 20,992 cores（接近 2x 吞吐）
   - RTX 6000 Blackwell：~20k cores（架构代际更新，单核性能更强）
   - CUDA cores 总量接近，但 Blackwell 赢在代际差

4. **最终结论——真正拉开差距的其实是显存**：

   | | 双 3090 | RTX Pro 6000 Blackwell |
   |---|---|---|
   | CUDA cores | ~21k（两卡） | ~20k |
   | 吞吐量 | 接近 2x 单卡 | 单卡强 |
   | 显存 | **2×24GB = 48GB（分裂的）** | **96GB（统一的）** |

   **48GB 分裂 vs 96GB 统一是最大差距。**

### 为什么显存分裂是核心瓶颈

不是总容量的问题，是**单模型能装多大的问题**：

```
双 3090:
  模型 + 数据必须均分到两张卡
  → U-Net 必须能塞进单卡 24GB
  → batch size 受单卡显存限制
  → 大模型/大分辨率直接 OOM

RTX Pro 6000 Blackwell:
  单卡 96GB 统一显存
  → 整个模型 + 大 batch 都在一块卡上
  → 没有"必须塞进单卡"的限制
  → 可以跑更大模型、更大分辨率、更大 batch
```

### 对我们项目的意义

当前 HCMC 数据是 128×128×3，U-Net 9.88M 参数，单卡 24GB 就够装。所以双 3090 和 6000 Blackwell 在当前实验上差距不大。

但如果后续要：
- 升级到更高分辨率（512×512、1024×1024）
- 训练更大的模型（更多 channels、更深的 U-Net）
- 用更大的 batch size 提高训练稳定性

**96GB 统一显存会成为不可替代的优势**，而双 3090 的 24GB 单卡上限会成为硬瓶颈。

---

## 七、结论

WSL2 = 一台 Linux 虚拟机，可以直接在上面跑 CUDA / PyTorch / 双 3090 训练。不需要单独的 Linux 机器。从 Windows 迁移数据集用 robocopy `\\wsl$\` 最快。代码和数据放 Linux 文件系统内，避免 `/mnt/c/` 性能损失。最终硬件目标为 RTX Pro 6000 Blackwell 96GB，其 96GB 统一显存是相比双 3090 的核心优势。
