# 多 Probe 信息对比实验

此入口独立于 `main.py`。原 `physics.py`、`belief.py`、候选标签生成、任务决策、地图、旧 HTML 报告均未修改。

本实验只比较信息效果：从同一个原始接近状态出发，改变 `Config.probe_force_max`，调用原 `World.probe()`。整个时间轴固定使用原候选库 `J_door < J_bypass` 标签，不执行或改变原任务在 Probe 后的 continuation 重算。报告中的 Uncertain 表示原 D 门控尚未明确；原任务此时的单次 Probe 后保守执行动作仍为 Avoid。

## 集中配置

`probe_comparison_config.json` 默认包括：

| Probe ID | 指令力上限 | 模拟时间 |
| --- | ---: | ---: |
| probe_1p0N | 1.00 N | 1.20 s |
| probe_1p4N | 1.40 N | 1.20 s |
| probe_1p8N | 1.80 N | 1.20 s |
| probe_standard_2p2N | 2.20 N | 1.20 s |

采样间隔 0.05 s（含 t=0 共 25 帧），D_clear=0.36，persistence_frames=4，随机种子 0。Sobol 是原项目确定性、未扰乱、保留首点的 1024 点，种子不改变点集；未增加观测噪声。MuJoCo 步长沿用 0.002 s。

原一致性尺度保持 0.30 N / 0.035 rad，alpha=2。每帧只比较同 Probe、同一时刻的 `[F_peak_sofar, delta_theta]`，独立调用 `consistency_update(original_prior, prediction(t), observation(t))`。不把相关的累计前缀逐帧相乘，不向更新函数提供真实门参数。时间取真实采样帧，禁止插值。

`t_clear` 是第一段连续 4 帧低于阈值、且支持同一方向的区间起点；`t_confirmed` 是第四帧时间。离线分析可以回溯起点；实际在线停止必须等确认帧。末尾不足 4 帧不记达标。后续阈值反复或决策反向单独保存。Not reached 对应 JSON null / CSV 空值。

几何统计包含物理坐标及参数域归一化坐标的加权协方差、PCA 主次轴、特征值、方向角、斜率、长短轴标准差、各向异性、ESS、90% 高权重候选集合范围。物理坐标角度依赖两维单位，因此报告额外显示归一化方向与宽度。整体 RMS 或任一参数边际标准差保留 prior 至少 50% 时标为“参数较宽”，两个阈值均可配置；几何指标不参与 Probe 选择或原策略。

## 命令

在本项目目录运行 PowerShell：

```powershell
$py = 'G:/anaconda3/envs/mujoco/python.exe'

# 只构建/恢复 4 × 1024 候选响应；指定全新目录
& $py probe_comparison.py --library-only --output results/probe_comparison_my_run

# 在相同目录运行 A/B/C/E，复用已完成响应；也可省略上一步直接执行本命令
& $py probe_comparison.py --output results/probe_comparison_my_run

# 从保存的数据验证并重新生成自包含 HTML，不运行 MuJoCo
& $py probe_comparison.py --report-only --output results/probe_comparison_my_run

# 不传 --output 则自动创建带时间戳的独立目录
& $py probe_comparison.py

# 最小完整 Case 实验（仍构建全部候选响应），使用另一个新目录
& $py probe_comparison.py --case B --output results/probe_comparison_caseB

# 定向单元测试，以及原报告脚本的 DOM 回放交互测试
& $py -m unittest test_probe_comparison -v
node verify_probe_report.js results/probe_comparison_my_run
# 可选：真实无头 Chrome/Edge；需要环境支持浏览器调试接口
node verify_probe_report_browser.js results/probe_comparison_my_run
# 第二个参数可指定已安装浏览器，例如：
# node verify_probe_report_browser.js results/probe_comparison_my_run 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
```

相对输出路径按项目根目录解析。`--config` 可指定另一份集中配置。新输出目录只能为空目录；不会写入旧任务结果。续跑要求配置和源码快照一致；修改参数或源码后应使用新目录。旧 baseline 候选库存在时只读加载，不重写历史文件；若对应版本库不存在，会调用原构建流程生成新签名目录。

## 输出及复现

```text
results/probe_comparison_<timestamp>/
  report.html                         自包含、离线可交互；四个 Case 各自播放/切 Probe/拖时间
  probe_summary.csv/json              t_clear、确认时间、最终支持度/分歧/判断
  research_answers.md/json            Q1–Q7 的数据驱动总结
  config_snapshot.json                完整展开的原配置、每档配置、Case、版本及源码 SHA256
  candidates.npz/csv                  完整原 Sobol 点集、标签及路线成本
  approach_state.json                 所有 Probe 共享的完整初始机械状态
  source_snapshot/                    实际运行时的代码、配置及原模型文件
  baseline_regression.json            新 Standard 对原终点和原 prefix 的全库误差
  validation.json                     全帧重新计算及 JSON/NPZ/CSV 一致性
  html_data_validation.json           HTML 内嵌数据与保存 JSON 完全一致
  replay_validation.json              全帧 DOM 交互测试后生成
  browser_validation.json             可选真实浏览器测试成功后生成
  report_browser_preview.png          可选真实浏览器截图
  candidate_responses/<Probe ID>/
    manifest.json                     Probe、force、contact、方向、控制配置、步长、版本/初始状态等
    responses.npz                     N × T × 2；逐 world key、done、status、error、wall_peak
  case_<A/B/C/E>/<Probe ID>/
    actual_response.npz               实际 MuJoCo stream，带 Probe ID/force/times/签名
    diagnostics.csv                   门角速度、瞬时接触力、倾角、角速度、接触状态
    belief_history.json/npz           所有时刻完整 1024 维 weights、p_D/D 及 PCA/ESS
    weights.csv                       T × 1024 完整权重
    belief_metrics.csv                所有几何/决策统计与实际观测
    decision_curves.csv               D(t)、p_D(t) 数据
  anomalies.jsonl                     仅出现异常时生成，绝不把异常 rollout 作为正常响应
```

缓存每完成 32 个候选原子落盘。不同 Probe 独立缓存，逐 world key 包括该世界 ID 和参数。数值异常或飞行包络失败保留诊断、保持未完成并阻止 posterior 计算；重跑只处理未完成世界。附加诊断不加入观测维数。

报告保留所有 1024 点、权重大小和透明度、Push/Avoid 颜色与真值十字。Standard 与当前 Candidate 共用相同坐标、同一时间滑块和同一权重显示尺度。D/p_D 图显示全部 Probe，并标注 t_clear；另附 ESS、归一化 PCA 方向及宽度曲线。低 D 代表帧按钮便于观察参数分散但标签已经一致的情形。

Standard 回归验证比较全部 1024 个终点响应，并在旧 prefix 存在时比较全部匹配前缀。`validate_saved` 从保存的真实/候选响应重新计算每个 posterior、PCA/ESS、t_clear 并比对 JSON/NPZ/CSV。回放测试沿用旧项目 `verify_report.js` 的小 DOM 方案，实际执行报告自己的脚本，逐 Case × Probe × time 检查全部粒子、指标、明确标记、同步滑块及播放，不依赖 npm 包。可选浏览器测试额外需要本机调试接口可用；接口超时不等于报告数据或回放脚本验证失败。

结论只适用于这些确定性 Case、固定标签与当前采样网格；没有执行有噪声的统计重复试验。较小指令力上限不意味着较小峰值接触力或较小冲量，应查看实际响应记录。
