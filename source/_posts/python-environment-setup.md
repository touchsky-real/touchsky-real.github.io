---
title: Ubuntu中Python环境的配置
date: 2026-03-01 00:53:38
tags:
---

# Ubuntu中Python环境的配置

## 准备工作 & 核心原则

**环境**

* 系统：Ubuntu
* shell：zsh

**⚠️ 核心注意事项：绝对隔离原则**
操作系统中很多系统工具是依赖于系统的自带 Python（System Python）的，对它只看不摸。**千万不要用系统 Python 去跑业务代码，也不要用 `sudo pip install` 往里面装包。** 无论是使用现代的 `uv` 还是传统的 `pyenv`，核心目的都是为了实现环境的完全隔离。

---

## 方案一：现代极速方案（基于 uv，强烈推荐）

`uv` 是一个由 Rust 编写的极速 Python 包和项目管理器。它可以**一站式替代**下文传统方案中的 `pyenv`、`pipx` 和 `Poetry`，且无需繁琐的系统级编译依赖。

### 1. 安装 uv (一体化工具)

`uv` 会直接拉取预编译好的 Python 二进制文件，跳过漫长的源码编译过程。

```bash
# 下载并运行 uv 官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh

# 使环境变量立刻生效 (uv 会自动将路径写入你的 .zshrc)
source ~/.zshrc

```

### 2. 管理 Python 版本 (替代 Pyenv)

```bash
# 查看可用的 Python 版本
uv python list

# 安装指定的 Python 版本 (瞬间完成)
uv python install 3.13.12

# 将某个版本设为全局默认环境 (可选)
uv python pin 3.13.12

```

### 3. 管理全局命令行工具 (替代 Pipx)

用于全局安装独立的 Python 命令行软件（如 `ruff`, `httpie`），自动隔离，不污染系统。

```bash
# 全局安装 Python 代码格式化工具 ruff
uv tool install ruff

```

### 4. 项目与依赖管理 (替代 Poetry)

`uv` 原生支持 `pyproject.toml` 标准，默认在项目下创建 `.venv`。

```bash
# 初始化新项目
uv init my_project
cd my_project

# 添加依赖并自动创建/更新虚拟环境
uv add fastapi

# 运行代码 (自动使用项目内的虚拟环境)
uv run main.py

```

---

## 方案二：传统经典方案（Pipx + Pyenv + Poetry）

如果你需要维护老旧的系统，或者偏好每个工具各司其职的 Unix 哲学，可以使用这套久经考验的经典组合。

### 1. 安装系统依赖 + Pipx

准备构建 Python 缺少的底层库，并安装全局工具管理器 Pipx。

```bash
sudo apt update && sudo apt upgrade -y

# 安装构建 Python 所需的依赖库 (防止编译时报错)
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev git

# 安装 pipx 并配置环境变量
sudo apt install -y pipx
pipx ensurepath

```

### 2. 安装 Pyenv 并配置 Zsh

用于管理多个 Python 解释器版本。

```bash
# 下载并安装 pyenv
curl -fsSL https://pyenv.run | bash

# 将配置写入 .zshrc
echo '' >> ~/.zshrc
echo '# Pyenv Configuration' >> ~/.zshrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc

# 让配置生效
source ~/.zshrc

```

### 3. 编译安装 Python 3.13.12

```bash
# 更新版本列表并从源码编译安装 (需耐心等待数分钟)
pyenv update
pyenv install 3.13.12

# 设置为全局默认版本并验证
pyenv global 3.13.12
python --version

```

### 4. 安装 Poetry (通过 Pipx)

用于具体项目的依赖管理。

```bash
# 安装 Poetry
pipx install poetry

# 强制 Poetry 在项目目录下创建 .venv (强烈推荐)
poetry config virtualenvs.in-project true

```

---

## 总结与架构对比

### 方案对比表

| 特性 | 方案一：现代方案 (`uv`) | 方案二：传统方案 |
| --- | --- | --- |
| **核心工具链** | `uv` (All-in-one) | `apt`, `pipx`, `pyenv`, `Poetry` |
| **安装速度** | ⚡ **极快** (直接下载预编译二进制包) | 🐢 **较慢** (需从源码编译 Python 解释器) |
| **系统依赖** | 几乎为零 | 需要安装大量 `build-essential` 等 C 库 |
| **上手难度** | 平缓 (只需掌握少量 `uv` 命令) | 较高 (需理解三个不同工具的边界与配合) |

### 架构演进图

**传统方案架构 (多工具协作)：**

```text
+-------------------------------------------------------+
|   System Python (系统底层，给操作系统用的，别碰！)      |
+-------------------------------------------------------+
          ^
          | (完全隔离)
          v
+-------------------------------------------------------+
|   Pyenv (提供各种纯净的 Python 版本供上层使用)          |
+-------------------------------------------------------+
          |
          +------------------------+
          v                        v
+----------------------+   +----------------------------+
| Pipx (装工具)         |   | Poetry (装库)              |
| 管理全局命令行软件     |   | 管理具体项目的依赖           |
+----------------------+   +----------------------------+

```

**现代方案架构 (uv 一体化)：**

```text
+-------------------------------------------------------+
|   System Python (系统底层，给操作系统用的，别碰！)      |
+-------------------------------------------------------+
          ^
          | (完全隔离)
          v
+-------------------------------------------------------+
|   uv (核心管家：版本管理 + 虚拟环境 + 依赖解析 + 工具安装) |
+-------------------------------------------------------+
          |
          +------------------------+
          v                        v
+----------------------+   +----------------------------+
| uv tool (装工具)      |   | uv project (装项目库)       |
| 替代 Pipx             |   | 替代 Poetry                 |
+----------------------+   +----------------------------+

```
