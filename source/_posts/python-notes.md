---
title: python笔记
date: 2025-10-21 00:27:39
tags: python
---

## 模块

模块（module）是组织 Python 代码的单位。
模块拥有一个命名空间，其中可以包含任意的 Python 对象。例如：变量、函数、类、常量、甚至其他模块。
模块通过“导入（import）”被加载到 Python 程序中。

### 最基本的情况：模块 = 单个 `.py` 文件

例如你有一个文件：

```
mymodule.py
```

内容是：

```python
def greet():
    print("Hello")
```

你就可以在别的地方导入它：

```python
import mymodule
mymodule.greet()
```

在这个意义上，**模块就是单个 Python 文件**。

---

### 进阶：包（package）是一种特殊的模块

当你有很多 `.py` 文件想组织在一起时，可以创建一个**包**。
包其实也是一种“模块”，只是它是一个**文件夹**而不是单个文件。

例如目录结构：

```
mypackage/
    __init__.py
    utils.py
    models.py
```

你可以这样导入：

```python
import mypackage.utils
from mypackage import models
```

这里：

-   `mypackage` 是一个“包模块”；
-   `utils` 和 `models` 是包中的子模块；
-   包里必须有一个 `__init__.py` 文件（哪怕是空的）才能被识别为包（Python 3.3 以后也支持“命名空间包”，可以省略）。

---

### 3. 模块的实际形式

| 模块类型   | 例子                 | 说明                                          |
| ---------- | -------------------- | --------------------------------------------- |
| 单文件模块 | `math.py`            | 一个普通的 `.py` 文件                         |
| 包模块     | `mypackage/`         | 一个包含 `__init__.py` 的文件夹               |
| 内建模块   | `sys`, `os`          | 编译进 Python 解释器的模块（没有 `.py` 文件） |
| 扩展模块   | `_socket.pyd`, `.so` | 用 C/C++ 编写的模块                           |

---

## 导入

当导入单个 Python 文件（模块）或整个包时，Python 的 `import` 有几种常见语法形式 👇

---

### 🧩 一、导入单个 Python 文件（模块）

假设你有一个文件：

```
mymodule.py
```

内容：

```python
def greet():
    print("Hello")
```

#### ✅ 基本导入方式

```python
import mymodule
mymodule.greet()   # 使用“模块名.函数名”调用
```

这里：

-   `import mymodule`：告诉 Python 去加载 `mymodule.py`；
-   Python 会创建一个命名空间 `mymodule`；
-   你通过 `mymodule.greet()` 访问它的内容。

---

#### ✅ 给模块起别名

```python
import mymodule as mm
mm.greet()
```

`as` 只是起个简短名字，常用于像 `import numpy as np` 这样的情况。

---

#### ✅ 从模块中导入部分对象

```python
from mymodule import greet
greet()
```

这表示只把 `mymodule` 里的 `greet` 函数导入到当前命名空间。
调用时就不用加前缀了。

---

#### ✅ 同时导入多个对象

```python
from mymodule import greet, say_hello, version
```

---

#### ✅ 导入模块中所有公开对象（不推荐）

```python
from mymodule import *
```

这样会导入模块中所有没有以下划线开头的名字。
⚠️ 一般不推荐，因为容易让命名空间变乱。

---

### 📦 二、导入包（Package）

假设结构如下：

```
mypackage/
    __init__.py
    utils.py
    models.py
```

#### ✅ 导入整个包

```python
import mypackage
```

如果在 `__init__.py` 里写了内容，这样导入时会执行 `__init__.py`。
例如：

```python
# mypackage/__init__.py
print("Package imported")
```

---

#### ✅ 导入包中的子模块

```python
import mypackage.utils
mypackage.utils.some_function()
```

---

#### ✅ 从包中导入子模块或对象

```python
from mypackage import utils
utils.some_function()
```

或者更深一层：

```python
from mypackage.utils import some_function
some_function()
```

---

#### ✅ 起别名

```python
from mypackage import utils as u
u.some_function()
```

### ✅ 总结表

| 语法形式                           | 说明                     | 调用方式                 |
| ---------------------------------- | ------------------------ | ------------------------ |
| `import mymodule`                  | 导入模块                 | `mymodule.func()`        |
| `import mymodule as mm`            | 导入模块并起别名         | `mm.func()`              |
| `from mymodule import func`        | 只导入某个对象           | `func()`                 |
| `from mymodule import *`           | 导入所有对象（不推荐）   | 直接使用名字             |
| `import mypackage.utils`           | 导入包的子模块           | `mypackage.utils.func()` |
| `from mypackage import utils`      | 导入子模块到当前命名空间 | `utils.func()`           |
| `from mypackage.utils import func` | 直接导入函数             | `func()`                 |

---
