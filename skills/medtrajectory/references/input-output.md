# 输入输出约定

完整字段表和 JSON Schema 随固定版本下载，位于返回的
`runtime_directory/inference/README.md`、`input.schema.json` 和 `output.schema.json`。
词表为 `inference/model/vocabulary.json`，示例为 `inference/examples/synthetic_history.json`。
下载器只获取运行推理需要的 19 个文件；若需训练文档中引用的完整脚本和配置，请使用完整 GitHub 仓库。

输入是一名个体的一个 JSON，不需要姓名、证件号或患者 ID：

- `schema_version: 1`。
- `static`：`anchor_age_days`、`sex`、`bmi`、`smoking`、`alcohol`，均必填；除年龄外均为指定类别字符串。
  BMI 使用分箱类别，不能直接传数字。未知值使用 `missing`。
- `history`：按年龄非递减排列的 `{ "code": "diag:I10", "age_days": 18262.5 }` 列表。
  同日事件保持输入顺序；年龄单位为出生后天数，年转天使用 365.25。
  代码必须精确匹配词表。已经有死亡事件的历史不能继续生成。
- `generation` 可选：`seed`（默认 20260819）、`num_trajectories`（1–20，默认 1）、
  `max_new_events`（1–256，默认 30）、`horizon_days`（大于 0、最多 3652.5，默认 3652.5）。

静态特征年龄应对应基线测量时间，不能晚于最后历史事件。若该时间确实未知，原数据处理约定是
使用 `max(1, 首事件年龄)`，同时把 BMI/吸烟/饮酒设为 `missing`；向用户说明这个处理。
不要把当前年龄随意当作测量年龄。预测从最后历史事件开始；本版本不支持最后事件之后的已知无事件期条件化。

输出中的 `trajectories` 是随机生成序列。每条含 `event_count`、`events` 和 `stop_reason`：

- `events`：代码、token ID、事件类型、发生年龄（天/年）、距预测起点天数、采样等待天数、是否应用一天最小间隔。
- `death`：生成了死亡事件，随后停止。
- `horizon`：下一事件超出时间窗；可以返回空序列。
- `event_limit`：达到事件数上限，不能声称其后没有事件。

顶层同时返回模型和权重哈希、实际采样参数、运行环境、预测区间，以及历史接收/保留/截断数量。
少于八条历史事件会附上偏离论文生成队列入选条件的提示。年龄沿用原评估器的 float32 精度。

该接口不输出疾病 AUROC、个体疾病风险或死亡风险。它使用预训练生成检查点，不能替代另外训练和校准的风险头。
