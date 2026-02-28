---
title: how-to-update-blog
tags: [Hexo, 博客]
---

### 🔧 一键更新 Hexo 的完整步骤

1. **更新全局 Hexo CLI（命令行工具）**

    ```bash
    npm install -g hexo-cli@latest
    ```

    > 更新全局命令工具 (可以全局使用 hexo 命令) ，保证 `hexo` 命令兼容最新核心。

2. **进入你的 Hexo 博客项目目录**

    ```bash
    cd your-hexo-blog
    ```

    > 在这个目录下应能看到 `package.json`。

3. **更新 Hexo 主体（核心包）**

    ```bash
    npm install hexo@latest
    ```

    > 更新 Hexo 的核心逻辑部分（生成、部署等）。

4. **更新所有插件依赖**

    ```bash
    npm update
    ```

    > 更新渲染器、部署器等第三方 Hexo 插件。

5. **可选：修复依赖安全问题**

    ```bash
    npm audit fix
    ```

    > 自动修复安全漏洞（非必需，但推荐）。

6. **清理并重新生成网站**

    ```bash
    hexo clean
    hexo g
    ```

    > 清理旧缓存，确认更新后生成正常。

7. **查看更新结果**

    ```bash
    hexo version
    ```

    > 确认 `hexo` 和 `hexo-cli` 都是最新版本。

---

💡 总结一句：

> **CLI 全局更新 → 项目内更新 Hexo → 更新依赖 → 清理验证。**

### ⚠️ 更新建议与踩坑警告（必读）

对于 Hexo 这样的静态网页生成工具，社区的普遍共识是：**“只要没出 BUG 打包正常，就不要轻易更新”**。平时**强烈不推荐**为了追求最新版本而频繁执行更新操作，原因如下：

1. **大概率产生兼容性报错**：Hexo 大版本更新经常伴随着底层 API 更新或废弃。例如，较新版本的 Hexo 移除了对 Swig 渲染器的默认支持。如果你还在使用旧版 NexT 主题（如 `layout` 文件夹中均是 `.swig` 文件），升级 Hexo 后直接会导致 `hexo g` 生成失败。
2. **环境依赖牵一发而动全身**：一次简单的 `npm install hexo@latest`，往往会导致 Node.js 版本限制、第三方插件（部署器、生成器、字数统计等）出现连锁不兼容，修复这些报错本身可能比写博客还要耗时。

**什么时候才真的需要更新？**

- 换了新电脑，默认安装了最新版的 Node.js，导致旧版本 Hexo 疯狂报错无法运行。
- 想要折腾和使用了新主题/新插件，而它们明确要求 Hexo 核心必须在某个版本以上。
- 遇到了旧版 Hexo 中阻碍博客生成、部署的致命 Bug。

👉 **防坑终极指南**：在输入任何 update 命令之前，**务必先备份项目根目录下的 `package.json` 和 `package-lock.json` 文件**（如果是 Git 管理的项目，请确保在干净的分支状态下先 commit 提交一次）。万一更新后全线崩溃无法还原，直接利用 Git 回滚这两个文件，然后删掉 `node_modules` 重新 `npm install` 就能恢复原状。
