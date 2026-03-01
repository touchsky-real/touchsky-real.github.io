---
title: Hexo 自动化部署：GitHub Actions 终极指南
tags:
    - Hexo
    - 博客
    - Github action
date: 2026-03-01 00:53:20
---

### 1. 为什么选择 GitHub Actions？

- **解放双手：** 告别繁琐的 `hexo clean && hexo g && hexo d`。
- **多端同步：** 只要有浏览器，在公司电脑、iPad 或手机上直接修改 GitHub 里的 Markdown 文件，博客就会自动更新。
- **环境隔离：** 不用担心本地 Node.js 版本升级导致博客跑不起来，GitHub 会每次提供一个干净的标准环境。

---

### 2. 准备工作

假设你的仓库分支结构如下（**请务必确认**）：

- **`source` 分支**：存放博客源码（Markdown, `_config.yml` 等）。
- **`main` 分支**：用于展示生成的网页（HTML）。

#### 第一步：配置权限（一次性）

1. 进入 GitHub 仓库，点击 **Settings** -> **Actions** -> **General**。
2. 找到 **Workflow permissions**，勾选 **Read and write permissions**。
3. 点击 **Save**。

#### 第二步：创建自动化脚本

在你的本地博客根目录下，创建文件 `.github/workflows/deploy.yml`。

#### 第三步：填入配置（Pro 版）

复制以下代码，它包含了**防坑配置**：

```yaml
name: Deploy Hexo Blog

# 触发条件：当 source 分支有变动时
on:
    push:
        branches:
            - source

concurrency:
    group: ${{ github.workflow }}-${{ github.ref }}
    cancel-in-progress: true

jobs:
    build-and-deploy:
        runs-on: ubuntu-latest
        permissions:
            contents: write # 授予发布权限

        steps:
            # 1. 拉取代码（核心修改：fetch-depth: 0 和 submodules: recursive）
            # 这样可以防止某些依赖 Git 信息的插件报错，并确保主题文件夹被正确下载
            - name: Checkout
              uses: actions/checkout@v3
              with:
                  fetch-depth: 0
                  submodules: recursive

            # 2. 安装 Node.js 并开启缓存（核心修改：cache: 'npm'）
            # 这会利用缓存，让后续的部署速度提升 50%
            - name: Setup Node.js
              uses: actions/setup-node@v3
              with:
                  node-version: "20" # 推荐使用目前的 LTS 版本
                  cache: "npm"

            # 3. 安装依赖
            - name: Install Dependencies
              run: npm install

            # 4. 生成静态文件
            - name: Build
              run: npx hexo generate

            # 5. 部署到 main 分支
            - name: Deploy
              uses: peaceiris/actions-gh-pages@v3
              with:
                  github_token: ${{ secrets.GITHUB_TOKEN }}
                  publish_dir: ./public
                  publish_branch: main # 部署的目标分支

                  # 如果你绑定了自定义域名（如 example.com），请取消下面这行的注释
                  # cname: example.com

                  # 记录是谁提交的（可选，让 Commit 记录更好看）
                  user_name: "github-actions[bot]"
                  user_email: "github-actions[bot]@users.noreply.github.com"
```

---

### 3. 注意事项（必读）

1. **不要手动运行 `hexo d`：** 以后所有的部署都由 GitHub 接管。你只需要关注 `git push` 源码即可。
2. **`.gitignore` 很重要：** 确保你的源码分支中**不包含** `public/` 文件夹。因为 `public` 是由于 Action 现场生成的，如果源码里也有，可能会导致冲突。
3. **分支说明：**

- **Source (源码):** 这里是 `source` 分支。是你真正干活的地方。
- **Pages (网页):** 这里是 `main` 分支。是给访客看的地方。
- _请去仓库 Settings -> Pages 中，确认 Build and deployment 的 Branch 选的是 `main`。_

---

### 改动解析（为什么这么改？）

1. **`submodules: recursive`**: 很多好看的主题（Next, Butterfly）通常建议作为 submodule 安装。如果没有这一行，Action 下载代码时主题文件夹是空的，会导致生成失败。加上这行是“保命”操作。
2. **`cache: 'npm'`**: GitHub 每天有免费额度限制。开启缓存后，如果你没有新增插件，`npm install` 几乎瞬间完成，既省时间又省资源。
3. **Node 版本升至 20**: 紧跟时代，Node 18 即将进入维护期末尾，新版 Hexo 对 Node 20 支持更好。
