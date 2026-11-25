---
name: medtrajectory
description: 从 MedTrajectory 官方 GitHub 固定版本下载最小推理包和已训练 A2 权重，对本地 JSON 格式的历史临床事件生成带时间的未来轨迹。用于 MedTrajectory 真实推理、合成示例演示和输入输出校验；不提供校准后的疾病或死亡风险评分。
metadata:
  version: "0.2.0"
  upstream_commit: "3e0acac344bb1eefe3babedd2706adac1a11c89c"
---

# MedTrajectory

调用官方仓库的最小 CPU 推理入口，使用 UK Biobank A2 / Ours(Relative)
seed42 的正式训练权重。输入和输出均留在本地。模型、配置、词表和源码由
[references/upstream-lock.json](references/upstream-lock.json) 固定到同一个 commit 并校验 SHA-256。

## 工作流程

1. 明确用户需要合成示例还是对自己的历史数据推理。准备用户输入时先读
   [references/input-output.md](references/input-output.md)。缺少事实信息时使用明确的
   `missing` 类别或询问，不编造患者历史、代码映射或测量年龄。
2. 使用独立的空工作目录获取包。把 `SKILL_DIR`、`WORK_DIR`、`PYTHON` 替换为实际路径：

```text
PYTHON "SKILL_DIR/scripts/medtrajectory.py" fetch --workspace "WORK_DIR"
```

3. 下载返回 `runtime_directory`、`requirements` 和 `example_input`。如缺依赖，在独立目录建立
   虚拟环境，安装返回的 `requirements`；不要改用户其他环境：

```text
python -m venv "WORK_DIR/.venv"
VENV_PYTHON -m pip install -r "RETURNED_REQUIREMENTS_PATH"
```

   Windows 的 `VENV_PYTHON` 为 `WORK_DIR/.venv/Scripts/python.exe`；Linux 为
   `WORK_DIR/.venv/bin/python`。已具备 PyTorch/NumPy 的独立环境可以复用。

4. 使用同一个 Python 环境运行：

```text
VENV_PYTHON "SKILL_DIR/scripts/medtrajectory.py" infer --workspace "WORK_DIR" --input "INPUT_JSON" --output predictions.json
```

   仅演示时，运行真实模型处理随包提供的**合成**历史：

```text
VENV_PYTHON "SKILL_DIR/scripts/medtrajectory.py" demo --workspace "WORK_DIR" --output synthetic-demo.json
```

5. 返回 `WORK_DIR/outputs/` 中的新 JSON。说明模型版本、历史截断情况，以及各条轨迹的停止原因。
   演示结果必须注明“合成输入、已训练权重实时推理”。不要把它说成原论文指标的重新验证。

## 实际能力与边界

- 输入：临床事件代码与发生年龄、四项静态类别及测量年龄、采样数量/种子/时间窗。
- 输出：诊断、OPCS-4 手术/操作、癌症登记及死亡事件的随机未来轨迹，含年龄、间隔和停止原因。
- 仅使用模型词表中的代码；拒绝未知代码，不直接支持自由文本或未经映射的 MIMIC-IV 输入。
- 预测起点固定为最后一条历史事件；超过 128 条历史时保留最近 128 条并报告截断。
- 使用训练模型和原生采样协议，不使用随机初始化或历史结果回放。缺失或损坏权重时停止并说明错误。
- 这是一个 UKB 生成模型的单 seed 发布包。生成死亡事件表示该条轨迹终止，不是个体死亡风险评估；
  不把多次生成频率解释为已校准概率，不输出治疗建议。
- `fetch` 只从固定的官方 GitHub 地址下载锁定文件；`infer`/`demo` 不联网，不上传用户数据，
  不启动训练或调用 embedding API。拒绝覆盖已有文件和复用无本工具标记的非空目录。
- 保留上游 LICENSE/NOTICE；代码与本次权重按上游 Apache-2.0 许可发布，原始数据访问权限不随之转移。
- 此包为通用 `SKILL.md` 结构；GenPilot 导入与上架由平台管理方处理，不声称已获平台审核。
