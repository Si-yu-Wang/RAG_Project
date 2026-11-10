# 图推理结果交互展示

本目录提供一个英文静态页面，展示已有 ReaRev 预测结果中的问题、局部关系图和候选节点分数。数据已打包在页面中，切换问题时直接读取对应记录。

## 查看页面

直接用浏览器打开 [rearev_replay.html](rearev_replay.html)，无需启动服务器。分享时只需复制这个 HTML 文件。

开发时也可以在项目根目录运行：

```powershell
python demo/server.py
```

打开 http://127.0.0.1:8780。服务器提供 `static/` 下的页面和资源；修改这些源码后刷新即可查看。默认端口为 8780，可通过 `--port 8781` 更改，按 `Ctrl+C` 停止。

## 页面内容与操作

页面由 Query 选择框和一张流程画布组成：

```text
Input Graph → Graph Neural Network → Final Node Scores
```

- **Input Graph**：展示实体节点和保存的关系连接。点击节点查看路径及已有分数，点击边查看关系名称。
- **Graph Neural Network**：包含 Pass messages、Update nodes、Score nodes 三个流程块，点击查看各步骤的作用。
- **Final Node Scores**：展示当前问题排名前五的候选节点及分数，编号与左侧图对应。点击结果查看节点详情。

下拉框提供五个问题：

| 问题 ID | Query |
| --- | --- |
| WebQTest-380 | where is mount st helens volcano |
| WebQTest-84 | what type of music did john lennon sing |
| WebQTest-124 | what language does egyptian people speak |
| WebQTest-302 | what are the official colors of the dallas cowboys |
| WebQTest-237 | what did fred durst do |

切换问题会更新图与分数，并关闭当前详情。拖动画布可平移，滚轮或加减按钮可缩放，Fit 恢复适配视图。窄屏下可用 Graph / GNN / Scores 按钮聚焦相应区域。About 显示数据说明。

## 数据来源与处理

`build_static_replay.py` 从原始交付包 `webqsp_complete_no_index_20261109.tar.gz` 中读取两份文件：

| 包内文件 | 使用内容 |
| --- | --- |
| `artifacts/webqsp_test_member2/predictions.jsonl` | 问题文本、候选节点的关系路径文本 |
| `rankings/webqsp_rearev_s1_test.jsonl` | ReaRev 候选排名与分数 |

脚本按问题 ID 关联记录，取 ReaRev 前五名，将保存的路径文本拆成图节点和关系边。默认火山问题额外包含 Volcano、Crater Glacier、Volcano Adventure 三个候选节点，展示图共 9 个节点、8 条边。

展示图是从保存路径恢复的局部连接，不是模型当时使用的完整子图。所有候选分数均取自原始 ReaRev 排名，未对可见节点重新归一化；条形长度按当前最高分缩放。数据中的 `provenance` 保存原始问题 ID、来源文件、排名记录摘要和完整候选数量。

中间的 GNN 块是处理框架示意。页面没有逐层张量或消息权重，不现场运行模型；Query 选择只切换已保存结果。

## 修改与重新打包

以下命令均在项目根目录执行，构建脚本只使用 Python 标准库。

修改 `static/replay.html`、`static/replay.css`、`static/replay.js` 或已打包数据后，重新生成独立页面：

```powershell
python -B demo/build_static_replay.py --reuse-data
```

从原始交付包重新导出数据并生成页面：

```powershell
python -B demo/build_static_replay.py --archive 'D:\path\webqsp_complete_no_index_20261109.tar.gz'
```

将示例路径替换为实际文件位置。导出问题由脚本中的 `QUESTION_IDS` 指定。构建会写入 `static/replay-data.js` 和 `rearev_replay.html`，不加载模型权重。独立页面是生成文件，页面修改应在 `static/` 源码中进行后再打包。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `rearev_replay.html` | 内嵌样式、脚本和数据的独立页面 |
| `static/replay.html` | 页面结构与资源引用 |
| `static/replay.css` | 页面与画布样式 |
| `static/replay.js` | SVG 绘图、问题切换、详情与视图操作 |
| `static/replay-data.js` | 五个问题的图结构、结果和来源信息 |
| `build_static_replay.py` | 数据导出与独立页面打包 |
| `server.py` | 可选的本地静态资源服务器 |
| `test_replay_frontend.cjs` | 前端交互与打包检查 |
| `test_static_replay.py` | 图结构与原始排名数据检查 |
| `test_demo.py` | 静态服务器检查 |

## 检查命令

前端检查需要 Node.js，Python 检查使用标准库：

```powershell
node --test demo/test_replay_frontend.cjs
python -B -m unittest discover -s demo -p 'test_*.py' -v
```

原始分数比对检查使用构建脚本中 `DEFAULT_ARCHIVE` 指定的压缩包；该文件不存在时跳过这项比对，其余检查仍可执行。
