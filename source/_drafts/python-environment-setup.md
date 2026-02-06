---
title: Ubuntu中Python环境的配置
tags:
---

## 准备

### 环境

- 系统：ubuntu

- shell： zsh

### 注意事项

操作系统中很多系统工具是依赖于系统的自带 Python（System Python）的，对它**只看不摸**。不要用系统 Python 去跑代码，不要往里面装包。

## 安装

---

### 第一阶段：安装系统依赖 + Pipx

这里把构建 Python 缺少的库，以及 **Pipx** 一次性装好。

Bash

```
# 1. 更新软件源
sudo apt update && sudo apt upgrade -y

# 2. 安装构建 Python 所需的依赖库 (防止安装 Python 时报错)
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev git

# 3. 安装 pipx (直接用系统 apt 装最稳)
sudo apt install -y pipx

# 4. 确保 pipx 的路径加入到环境变量
pipx ensurepath
```

---

### 第二阶段：安装 Pyenv 并配置 Zsh

这步会安装 Pyenv 并且把必要的环境变量写入你的 `.zshrc`。(我用的是 **zsh** )

Bash

```
# 1. 下载并安装 pyenv
curl -fsSL https://pyenv.run | bash

# 2. 将 Pyenv 配置写入 .zshrc
echo '' >> ~/.zshrc
echo '# Pyenv Configuration' >> ~/.zshrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc

# 3. 让所有配置立即生效 (包括刚才 pipx 的和 pyenv 的)
source ~/.zshrc
```

---

### 第三阶段：安装 Python 3.13.12

Pyenv 准备好后，我们来编译安装指定的 Python 版本。

Bash

```
# 1. 更新 pyenv 的版本列表 (确保能找到最新的 3.13)
pyenv update

# 2. 安装 Python 3.13.12
# 注意：这步会从源码编译，需要几分钟，请耐心等待直到结束
pyenv install 3.13.12

# 3. 设置为全局默认版本
pyenv global 3.13.12

# 验证
python --version
# 应该输出 Python 3.13.12
```

> 可以运行 `pyenv install --list | grep 3.13` 查看当前可用的最新版（比如 3.13.1）。

---

### 第四阶段：安装 Poetry (通过 Pipx)

最后，用 Pipx 安装 Poetry 并开启“项目内生成虚拟环境”功能。

Bash

```
# 1. 安装 Poetry
pipx install poetry

# 2. 配置 Poetry 在项目目录下创建 .venv (强烈推荐)
poetry config virtualenvs.in-project true
```

---

### 检查

依次运行下面三个命令，确认全绿通过：

Bash

```
pipx --version
pyenv --version
poetry --version
```

## 结构图

```
+-------------------------------------------------------+
|   System Python (系统底层，给操作系统用的，别碰！)      |
+-------------------------------------------------------+
          ^
          | (完全隔离)
          v
+-------------------------------------------------------+
|   Pyenv (提供 Python 3.10, 3.11, 3.12 给你的项目用)    |
+-------------------------------------------------------+
          |
          +------------------------+
          |                        |
          v                        v
+----------------------+   +----------------------------+
| Pipx (装工具)         |   | Poetry (装库)               |
| 管理全局命令行软件     |   | 管理具体项目的依赖           |
| (如: install poetry) |   | (如: add fastapi)          |
+----------------------+   +----------------------------+
                                   |
                                   v
                           +------------------+
                           |  你的项目代码     |
                           |  (main.py)       |
                           +------------------+
```
