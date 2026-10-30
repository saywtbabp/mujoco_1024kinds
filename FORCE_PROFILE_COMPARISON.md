# Force temporal profile 对比

本轮接续 `PROBE_COMPARISON.md` 的信息对比实验。保留 1024 个原 Sobol 候选、原路线标签、统一初始机械状态、地图和接近方式。`physics.py`、`belief.py`、`config.py`、`main.py`、原候选库生成逻辑与原任务的 continuation 均未修改。

## Profile Library

所有 profile 都运行一次 1.20 s，采样间隔 0.05 s。定义集中在 `force_profile_comparison_config.json`。

| ID | 分段水平指令力上限 | B_F / N·s | 平均上限 / N |
| --- | --- | ---: | ---: |
| standard_2p2N | 0–1.20 s: 2.20 N | 2.64 | 2.20 |
| constant_1p6N | 0–1.20 s: 1.60 N | 1.92 | 1.60 |
| low_to_high_1p0_2p2 | 0–0.60 s: 1.00 N；0.60–1.20 s: 2.20 N | 1.92 | 1.60 |
| high_to_low_2p2_1p0 | 0–0.60 s: 2.20 N；0.60–1.20 s: 1.00 N | 1.92 | 1.60 |
| pulse_hold_2p2_1p48 | 0–0.20 s: 2.20 N；0.20–1.20 s: 1.48 N | 1.92 | 1.60 |

各段在切换时刻右连续。`B_F=∫F_cmd,max(t)dt` 是指令力上限的积分，既不是接触冲量，也不是经过速度反馈、横向保持及限幅后的实际水平指令积分。四个 matched profile 只有完整 1.20 s 的预算相同；中途累计预算可以不同。报告的 slider 同时显示已用预算，终点 L1 对比与全程最大 L1 分开保存。

`ForceProfile` / `ForceSegment` 提供 `force_limit_at(t)` 与 `budget_until(t)`，没有按 profile ID 分散编写控制分支。`ProfileControlConfig` 仅把现有控制器读取的 `probe_force_max` 替换为 elapsed integration steps × timestep 的函数，所有其他配置转发原对象。此时钟等价于 Probe elapsed time，避免浮点时间令 0.20/0.60 s 切换晚一个物理步。profile 不读取门状态、真实参数或标签。Standard 保留原 Config 对象及原常数访问路径。

原 `World.probe()` 仍然执行原横向保持、推进速度阻尼、世界 +Y 方向、水平限幅和四旋翼反馈；没有额外门 torque，没有执行中覆盖 qpos/qvel。所有 profile 的段边界要求与原每 5 个物理步的电机控制节拍对齐。

## Belief 与报告

仍用原 `consistency_update`、0.30 N / 0.035 rad、alpha=2。每帧只比较同 profile、同一时刻的累计 `[F_peak_sofar, delta_theta]`，从 uniform prior 独立重算。完整 profile signature 包括所有 segment，峰值相同也不能混用预测。门角速度、瞬时接触力不加入 likelihood。

保持 D_clear=0.36、连续 4 帧且支持同一方向。t_clear 为稳定区间起点，t_confirmed 为第四帧；真实在线停止要等确认，报告后续反复。整个信息实验固定原 `J_door < J_bypass` 标签，不进行 Probe 后重新标注。

报告延续全部粒子的颜色、大小、透明度、真值十字、PCA 轴和播放。Reference 默认 Standard，也可切换 Constant 1.6 N 或其他 profile；Candidate 可选五种 profile。两图使用同一 slider、坐标和权重显示尺度。输入 profile、D/p_D、ESS、归一化 PCA 角/长短轴、RMS 宽度、门角/角速度、实际接触力曲线同步显示时间指针，并标记 0.20/0.60 s 切换点。

Comparison A 与 Standard 实际 early-stop 比较；Comparison B 专门比较 1.92 N·s 组。所有 fastest 标记只表示实验中的首次明确时间，不构成 Probe 选择策略。

PCA 主轴按无向轴（模 180°）比较，避免 ±90° 边界伪跳变。近各向同性 prior 没有可解释的唯一方向。报告可靠方向比较阈值为 ESS≥5、归一化各向异性≥2，方向变化观察阈值 10°、接近 Standard 的时间容差 0.10 s，均在集中配置中，只用于解释；所有原始角度仍保存在 JSON/CSV。

切换分析保存 s−0.05、s、s+0.05 的 D、差分斜率、ESS、PCA 夹角及物理响应。s 时刻观测反映 [0,s) 输入，切换后第一帧是 s+0.05。不能把切换附近的角度变化单独解释成已独立辨识惯量或摩擦。

## 命令

在项目根目录运行：

```powershell
$py = 'G:/anaconda3/envs/mujoco/python.exe'

# 扩展后的原入口：构建或恢复五种 profile × 1024 candidate streams
& $py build_probe_stream.py --profiles --output results/force_profile_comparison_my_run

# 完整 A/B/C/E，复用上述缓存（也可直接省略上一步）
& $py force_profile_comparison.py --output results/force_profile_comparison_my_run

# 显式指定 Cases，默认也是 A/B/C/E
& $py force_profile_comparison.py --case A B C E --output results/force_profile_comparison_my_run

# 保存数据验证与报告重建，不运行 MuJoCo
& $py force_profile_comparison.py --report-only --output results/force_profile_comparison_my_run

# 自动创建新的时间戳目录
& $py force_profile_comparison.py

# 回归与交互测试
& $py -m unittest test_force_profiles test_probe_comparison -v
node verify_profile_report.js results/force_profile_comparison_my_run
```

`--config` 可指定另一份 profile 配置。修改配置/源码后使用新的目录；同目录续跑严格核对完整快照。未带 `--profiles` 的 `build_probe_stream.py` 保留原 Standard 功能。

## 输出与缓存

新目录为 `results/force_profile_comparison_<timestamp>/`，不写入或覆盖 magnitude 结果。

- `report.html`：自包含交互式报告，可离线打开。
- `profile_summary.csv/json`：预算、t_clear/confirmation、最终 p_D/D、动作、ESS、PCA、宽度及最快标记。
- `config_snapshot.json`、`source_snapshot/`、`candidates.npz/csv`、`approach_state.json`：完整配置、代码/模型版本、原候选及标签和共享初始状态。
- `budget_validation.json`：解析积分与每 0.002 s 离散积分检查。
- `baseline_regression.json`、`validation.json`、`html_data_validation.json`、`replay_validation.json`：全库 Standard 回归、逐帧重算/物理记录/历史文件校验、内嵌数据与回放测试。
- `profile_pair_differences.csv/json`、`profile_pair_summary.csv/json`：每个 Case 所有 profile pair 的逐帧/终点 L1、D 差、PCA 夹角和可靠性。
- `transition_analysis.csv/json`：切换前后响应和 belief 指标。
- `research_answers.md/json`：由保存数据生成的 Q1–Q8 回答，与控制台一致。
- `candidate_responses/<profile_id>/manifest.json`、`responses.npz`：逐 candidate 的 profile/segment/config/contact/timestep/source/world key，预测及额外物理诊断。
- `case_<Case>/<profile_id>/actual_response.npz`、`belief_history.json/npz`、`weights.csv`、`belief_metrics.csv`、`decision_curves.csv`、`diagnostics.csv`、`physical_response.csv`、`command_trace.csv`：完整 response、belief、几何指标、20 Hz 物理诊断，以及每物理步输入上限与实际水平指令。

候选缓存每 32 个世界原子保存，续跑只补未完成世界。异常保存到 `anomalies.jsonl` 并阻止后验计算，不移除异常候选来制造收缩。所有旧 magnitude 文件与原基线库文件在运行前后用 SHA256 校验，证明历史输入未被修改。

Standard 全部 1024 个终点和前缀对原库回归，真实 A/B/C/E 的 Standard 权重、响应与 clear time 也逐一对上一轮结果检查。全部 posterior、PCA/ESS、确认时间从保存响应重算；command trace 每个物理步检查切换及 cap。DOM 测试执行报告自己的 JS，覆盖 Standard/Constant Reference、所有 Candidate、25 帧、全部 1024 粒子及播放。

当前为确定性 MuJoCo、理想观测和固定模型试验，不作噪声条件下的统计显著性声明。物理响应图中的瞬时接触力是 20 Hz 诊断采样；累计峰值仍在所有物理步上计算。
