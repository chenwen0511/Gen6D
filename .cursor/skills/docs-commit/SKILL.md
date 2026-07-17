---
name: docs-commit
description: >-
  Sync code changes into README/doc, then create a git commit from the user's
  staged (git add) changes. Use when the user says docs-commit, 同步文档并提交,
  根据 git add 提交, 同步修改到文档并 commit, or asks to update docs then commit
  staged changes.
---

# docs-commit

用户已 `git add` 后，完成两件事：文档同步 + 按暂存区提交。

## 流程

### 1. 查看暂存区（并行）

```bash
git status
git diff --cached --stat
git diff --cached
git log -8 --oneline
```

若暂存区为空：提示用户先 `git add`，不要空提交。

### 2. 同步文档

对照 **staged + unstaged** 的代码改动，更新用户可见文档（只改与本次变更相关的内容）：

| 变更类型 | 更新位置 |
|----------|----------|
| UI 页签名 / 按钮文案 | `README.md` 页签表、`doc/sam3_seg_tab.md`、`doc/接口文档.md`、`doc/grasp_api.md`、`doc/pem.md` |
| REST 路径 / 请求响应字段 | `doc/grasp_api.md`、`doc/接口文档.md`、`README.md` 文档索引 |
| 抓取点算法（p_i / q_i / P1） | `doc/sam3_seg_tab.md` |
| SAM-6D / PEM | `doc/pem.md`、`doc/sam6d_rest_api.md` |
| 启动方式 / 端口 | `README.md` 快速开始、`doc/接口文档.md` |

规则：

- 优先改用户可见名称与行为说明；算法实现细节文件名（如 `sam3_tab.py`）可保留。
- 不要写无关长文；与 diff 不一致的旧称呼一并改掉。
- 文档改完后 `git add` 相关 doc / README。

### 3. 提交（遵循仓库 commit 约定）

1. 再跑一次 `git status` / `git diff --cached` / `git log` 确认最终暂存内容。
2. 用 HEREDOC 写 1–2 句中文或与仓库风格一致的 commit message（侧重 why）。
3. 提交后 `git status` 验证。
4. **不要** `push`，除非用户明确要求。
5. **不要**改 git config；不要 `--amend`（除非用户明确要求且满足 amend 安全条件）。

示例：

```bash
git commit -m "$(cat <<'EOF'
docs: 将 UI 页签「SAM3 分割」更名为「抓取 位姿估计」

同步 README / doc 与代码中的页签称呼，避免文档与界面不一致。
EOF
)"
```

### 4. 回复用户

简短说明：文档改了哪些、commit hash、一句话摘要。提醒下次可说 **docs-commit**。
