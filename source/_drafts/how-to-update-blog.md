---
title: how-to-update-blog
tags:
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
