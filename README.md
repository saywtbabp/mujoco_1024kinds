# 1024 个物理候选世界的主动交互导航验证

## 在 PyCharm 中运行

1. 打开整个 `mujoco_1024kinds` 文件夹。
2. 使用现有解释器 **`G:\anaconda3\envs\mujoco\python.exe`**。工程内已经提供对应的 PyCharm 模块和运行配置；如果 PyCharm 没有识别已注册的 SDK，在 Python Interpreter 中选择这个现有解释器即可。
3. 右键 **`main.py` → Run**。无需参数、训练数据或重新安装依赖。

`main.py` 默认生成/复用完整候选库，依次运行 Case A–E。**每个场景都显示 MuJoCo 原生窗口和实时面板**，形式沿用 `newmujoco` 的 Tkinter + Pillow 实时界面。面板左侧是真实 MuJoCo 渲染的无人机推门近景，右侧是随 Probe 观测更新的 1024 候选权重，下方显示接触力、门角、p_D/D 曲线及高权重候选。

2026-10-18 提前制动版本为 `results/libraries/large_shortcut_0d7390e3ebf7`：499/1024 完成 Door，482/1024 的 Door 更快；失败分布 {"push_timeout": 525}。ID=20/312/432 在 2.2 N 下均解除接触并到达 M，墙峰值为 0。完整 A–E 和报告在 `results/20261018_232925`，详细对照及仍存在的续程失败见 [VALIDATION.md](VALIDATION.md)。历史结果全部保留。

每个场景完成后，窗口保持结果，点击 **“下一场景”** 继续；最后点击 **“完成 / 关闭”** 结束。空格或暂停按钮可暂停/继续，速度框可选 0.25–4 倍，保存按钮可保存当前面板。关闭面板会关闭显示并继续后台实验。最终 HTML 报告中的权重图可以播放、拖动时间滑块，比较初始等权、Probe 过程、Probe 结束与最终决策。

其他入口：

| 文件 | 作用 |
| --- | --- |
| `build_library.py` | 只生成/恢复 1024 个候选世界库 |
| `build_probe_stream.py` | 生成/恢复 1024 个候选在 Probe 各时刻的累计响应预测 |
| `verify.py` | 运行数值、几何、控制和物理映射验证 |
| `verify_agreement.py` | 两个独立可行域内各 1024 点，实际验证无 Probe 的 Push/Avoid 分支 |
| `check_results.py` | 检查最近一次完整 A–E 实验的真实输出与研究不变量 |
| `sensitivity.py` | 实际绕行距离扫描和一致性尺度扫描 |
| `render_preview.py` | 渲染原无人机与门阶段场景预览，需正常 OpenGL |
| `test_live_belief.py` | 验证时刻匹配、实时更新和重复帧不重复计证据 |
| `test_recovery.py` | 验证电机可实现性、恢复门控、超时、数值异常及 ID=20/312/432 提前制动 |
| `verify_push.py` | 记录 63 条基线及四个 Probe 后状态的提前制动/补推物理对照 |
| `verify_recovery.py` | 保存 12 个不同惯量/阻力样本的实际恢复轨迹；`--baseline` 重放保留的修改前控制源码 |
| `audit_recovery.py` | 比较保留基线与最新完整 1024 世界结果，生成失败分布和 A–E 对照 |
| `verify_live.py` | 实际运行 Tk/MuJoCo 面板验证，自动切场景和关闭 |
| `experiment_config.json` | 修改物理可行域、无人机来源、进程数；未列出的参数使用 `navlab/config.py` 默认值 |

可选命令行（工作目录可任意，所有输出路径按入口文件定位）：

```powershell
& 'G:/anaconda3/envs/mujoco/python.exe' 'E:/Desktop/projectgroup/newfolder/mujoco_1024kinds/main.py' --headless
& 'G:/anaconda3/envs/mujoco/python.exe' 'E:/Desktop/projectgroup/newfolder/mujoco_1024kinds/main.py' --case E
& 'G:/anaconda3/envs/mujoco/python.exe' 'E:/Desktop/projectgroup/newfolder/mujoco_1024kinds/main.py' --auto-advance --close-on-done --speed 2
& 'G:/anaconda3/envs/mujoco/python.exe' 'E:/Desktop/projectgroup/newfolder/mujoco_1024kinds/verify.py'
```

Windows 并行仿真采用带 `__main__` 保护的进程池。默认 4 个进程；可在配置中设 `workers=1`。1024 点库首次生成需要几分钟；Probe 后每个新观测状态还要评估 1024 组剩余路线，也需要等待。控制台持续显示进度，离线库及 Probe 后 continuation 都每 32 个世界保存一次，重跑只计算未完成世界。带未完成标志或数值异常的缓存不能用于决策。完整运行完成后，后续默认运行复用候选库和 Probe 后路线缓存。

实时面板使用现有环境中的 Tk 8.6 和 Pillow 11.3.0，已实测可用；`requirements-live.txt` 单独列出 Pillow。`--headless` 完全不创建 Tk 或 OpenGL 窗口，但仍生成权重变化记录和可回放报告。初次补充 Probe 时序库约需几十秒，之后直接复用。

## 实时权重更新的含义

Probe 时每 0.05 s 计算一次权重，图像约每 0.10 s 刷新。当前观测为 **从 Probe 开始到当前时刻的累计峰值接触力和门角变化**：

```
z*(t) = [F_peak_so_far*(t), θ*(t)-θ*(0)]
z_hat_i(t) = MuJoCo 中同一标准 Probe、同一持续时刻的候选预测
w_i(t) ∝ w_i(0) exp(-α e_i(t))
```

当前观测与同一时刻的预测比较，避免把 0.2 s 的观测与 1.2 s 的最终预测混用。每次从原始先验重新计算，避免把高度相关的累计观测帧反复相乘。Probe 完成时，权重严格回到原方案使用最终 `[F_peak,Δθ]` 的更新结果；G、D 门控规则与标准 Probe 控制保持不变。本次调整的是正式推门中的提前制动、补推及后续恢复衔接，路线时间和标签需要重算。

**当前实时权重更新覆盖标准 Probe。** 后续正式 Push/Bypass 使用该 Probe 后验；真实执行画面、接触力和门角继续刷新，权重保留。正式推门使用不同的输入控制，继续用标准 Probe 预测更新会混用模型，因此界面明确标注“执行期间保留权重”。初始直接 Push / Avoid 或 Case D 没有 Probe 时，图保持等权，不人为制造权重收缩。

Probe 后重新计算路线标签时，面板保持响应并标注正在预测，实际物理状态保持在 Probe 终点。标签重算完成后，颜色与 p_D/D 更新，但不会再次乘入同一份 Probe 观测。原生 MuJoCo 窗口使用独立模型/数据映射显示真实状态，防止 `viewer.sync()` 的界面计算影响实际实验。

HTML 报告是**已记录过程的离线回放**，实时观看使用运行时面板。每个有 Probe 的场景记录 27 帧权重（初始、24 个前缀、Probe 终点和剩余路线标签），D 无 Probe 保留一帧；拖动或播放不会重新运行物理仿真。

## 实现的研究框架

维护的状态始终是 `{θ^(i), w_i}`，其中 `θ=[I,τ_f]`。没有 F/H/R、Easy/Hard 分类器，没有高斯分类模型，也没有用信息熵决定 Probe。

默认物理可行域：

```
I ∈ [0.05, 0.80] kg m²     （绕铰链的总转动惯量）
τ_f ∈ [0.01, 8.00] N m      （铰链 frictionloss）
N = 1024 = 2^10
```

这是用于验证的允许参数域，尚未通过真实门实验标定。它不是现实门的概率分布。二维 Sobol 序列对连续域进行数值覆盖，保留全部 1024 点，包括首点，不跳过、不抽稀。初始 `w_i=1/1024` 表示缺少额外证据，因此不人为偏向某个物理假设。视觉阶段假设各门外观、尺寸、铰链和初始关闭状态完全相同，`w_i^V=w_i`。视觉识别“这是门”被视为已成立的输入条件，首阶段没有实现检测网络。

离线流程：

```
Sobol θ^(i)
    → 写入 MuJoCo 门惯量与铰链阻力
    → 从相同关闭门 / 无人机接近状态执行统一 Probe
    → z_hat_i = [F_peak_hat_i, Δθ_hat_i]

每个候选世界
    → Door: S → 接近门 → 推门 → 脱离 → 穿越 → M
    → Bypass: S → 右侧绕行 → M
    → y_i = 1[J_door_i < J_bypass_i]
```

在线流程：

```
地图 A* → G
候选参数 + 权重 + 路线标签 → p_D → D

G < G_threshold                  → Avoid
G 高，D 低，p_D > 0.5            → Push
G 高，D 低，p_D ≤ 0.5            → Avoid
G 高，D 高                       → 一次 Probe
    → 真实观测 z*
    → prediction-observation consistency soft weighting
    → 在同一 Probe 后观测状态重算剩余路线标签
    → 重算 p_D、D → Push / Avoid
```

```
e_i = ((F_peak* - F_peak_hat_i) / 0.30 N)^2
    + ((Δθ* - Δθ_hat_i) / 0.035 rad)^2
s_i = exp(-2 e_i)
w'_i = w_i s_i / Σ_j w_j s_j

p_D = Σ_i w_i y_i
D = 4 p_D (1-p_D)
G = J_bypass^geo - J_door^geo
```

一致性容差是可配置的工程尺度，不被宣称为高斯标准差。代码在 log 空间归一化，避免所有 `exp(-αe)` 同时下溢。Probe 后支持度必须加权计算。

默认 `G_threshold=3 m`，`D_threshold=0.36`（低 D 相当于 p_D 小于约 0.1 或大于约 0.9）。这是可解释的门控阈值，不是已经证明最优的信息价值准则。

## MuJoCo 模型和控制

`models/Mujoco_0`、`models/Mujoco_1`、`models/Mujoco_2` 中的 XML 是对应原无人机 XML 的完整副本。保留自由杆件、拉索、机架、传感器、四旋翼执行器及其限制。默认使用 `Mujoco_0`，切换来源只需修改 `model_source`。本工程的 `control()`、`push_open()`、`disengage()` 和阶段流程由此前 agent 新增，未直接调用原工程的 `QuadSO3Controller` 或 `set_recovery_mode()`；原工程已有碰撞恢复能力，不能把本工程的控制缺陷归因于原飞控。导航固定高度 1.25 m，只给四个旋翼发送控制量，不用辅助姿态电机，不给门直接施加力矩，不在执行期间拖动/覆盖 qpos/qvel。

门宽 1.1825 m、厚 0.05 m、高 2 m，铰链位于 `(-0.61, 1.0625)`，转动范围 0–88°。固定铰链粘性阻尼 0.05 N m s/rad；隐藏变量只有 I、τ_f。

**惯量写入必须包含平行轴项。** MuJoCo 的 `body_inertia` 是质心处主惯量，不能直接把其 z 分量设成铰链惯量 I。本实现使用同尺寸均匀箱体：

```
I_hinge = m (width²/3 + thickness²/12)
m = I / (width²/3 + thickness²/12)
```

按质量同步缩放质心惯量，调用 `mj_setConst` 更新模型常数。测试直接检查 MuJoCo 质量矩阵的铰链元素等于输入 I。质量是 I 与固定几何决定的量，不是第三个隐藏参数；外观和碰撞几何始终一致。

标准 Probe 在共同接近状态 `(0.225, 0.66, 1.25)` 开始。名义接触横坐标 0.225 m，相对铰链力臂 0.835 m；方向固定为世界 +Y；持续 1.20 s；水平指令力范数上限 2.20 N；横向保持和推进速度阻尼采用统一反馈律。所有候选输入控制方案、初始状态、持续时间完全相同，反馈产生的具体电机信号允许因响应不同而变化。真实模型的名义接触会受到张拉杆件形变影响，首版没有强制约束接触点的接触夹具。

`F_peak` 是 Probe 时间窗内每个物理步上，**所有无人机几何与门接触的法向力之和的峰值**，由 `mj_contactForce` 读取。它不是电机指令，也不是单个接触点的力。由于碰撞冲量、接触约束与结构弹性，峰值接触力可以明显大于 2.20 N 的水平指令上限。`Δθ` 是相同时间窗终点与起点的门角差，不使用归一化分类标签替代原始量。

推门水平指令上限保持 **2.2 N**，总 Push 超时 10 s。从门角 <20° 的早期状态进入时启用提前制动反馈：门角 ≥20° 且观测转速 ≥0.75 rad/s 后转入 **PUSH_BRAKE**，冻结当前跟进位置和机头目标，用位置/速度反馈制动，让门依靠已有运动继续打开。若门转速降到 <0.15 rad/s，且无人机速率 <0.35 m/s、倾角 <15°，则恢复 PUSH 补推；补推再次加速后仍可制动，用转速滞回避免抖动。若执行开始时已在 ≥20° 的中途运动状态，则保持连续推门反馈直到 75°，避免中途突然切入冻结目标。所有世界采用同一观测状态规则，没有读取 I、τ_f。制动和等待不重置推门超时，没有统一降低推力上限；慢门保持原推动反馈。标准 Probe 完全不使用这些切换规则。

门角达到 **75°** 立即进入 BACKOFF。该阶段继续供给升力，按速度制动、沿观测门板法向退离，横向目标约束到现有通道航线 x=0.225 m，避免法向退离把目标放进右墙膨胀区；无需回到原起点。PUSH_BRAKE/BACKOFF 暂缓转机头，使用原先恢复阶段的可实现推力分配：可行的完整请求原样保留；超限时先确定 collective，再整体缩放 roll/pitch 差动，最后用剩余能力满足 yaw。标准 Probe 与正常 PUSH 的分配方式保持原样。

恢复阶段使用原高度反馈所需的垂直力除以 `max(R[2,2],0.35)` 计算 collective，防止大倾斜时原力投影丢失垂直支撑；随后仍受四电机实际范围限制。恢复结束即把机头目标交回路线方向，避免临时机头方向与后续到点条件互相冲突。

恢复不再固定等待 2.5 s。每个物理步检查门/墙接触力各 ≤0.10 N、倾角 ≤15°、角速度范数 ≤0.8 rad/s、速率 ≤0.35 m/s、高度偏差 ≤0.15 m；同时每 0.05 s 用原 **0.29 m** 净空检查当前位置和后续 A* 路径。条件持续满足至少 0.15 s 才转入 FLIGHT。统一恢复超时 10 s，记录未满足的条件。恢复允许带小速度进入穿越，不套用 M 的严格悬停标准。穿越路径显式经过门后安全航点 `(0.225,2.695,1.25)`，再到共同 M；所有执行时间计入 J_door。

标签表示同一控制规则在该候选世界下的实际路线优劣，仍由 `J_door < J_bypass` 生成。日志区分 `push_timeout`、`recovery_timeout`、`planning_clearance_failure`、`recovery_failure`、`crossing_failure` 和 `numerical_anomaly`；恢复超时细分未达标状态。失败目标时间为 +∞，另存真实耗时 `elapsed_time`。数值异常会保留诊断并阻止候选库或 continuation 用于决策，不解释成物理难推。

## 地图、几何价值与共同汇合状态

采用 `newmujoco` 已有地图的**门阶段建筑结构**：左侧封闭块、中间门通道、右侧封闭块及右侧绕行。补齐该阶段外围边界，避免绕出实验区域。当前研究只验证门阶段，未加入玻璃阶段。S、门位置和 M 沿用原门阶段坐标；增加一个明确的公共 M→G 后续段。`navlab/map.py` 的同一组墙同时生成 MuJoCo 场景与 A* 栅格。

两条路径都是从 `S=(-3.8,0.5125)` 到 `M=(0.225,2.8875)`。Door 几何假设开放门通道并关闭右侧备选通道，确保搜索得到真正经过门的路径；Bypass 几何假设门关闭。G 使用 8 邻域 A* 搜索的累计米制路径代价（含端点接入），不是启发函数 h(n)，也不是两个终点不同的路径之差。报告绘制的是经净空检查简化后的执行航点，G 保留原始 A* 搜索代价。

大收益地图右侧块末端 x=3.2625 m。小收益地图只把这一末端缩短到 x=1.05 m，保持门和附近通道不变。两个地图的 Door 控制空间相同；小收益地图重新执行 Bypass 获取时间。为了避免 1024 次重复的相同无接触模拟，离线 Bypass 在验证完全没有门/墙接触后共享其 rollout 时间；每个候选仍有明确 `J_bypass_i`。

M 的到达条件同时检查位置误差 <0.08 m、速率 <0.06 m/s、偏航误差 <0.10 rad、角速度 <0.15 rad/s，连续满足 0.30 s。两条路线从共同的初始稳定状态计时。公共初始化悬停不计入双方路线比较；接近门、推动、脱离、穿越和到 M 的稳定时间全部计入。M→G 另行实际执行并记录，不加入 J_door/J_bypass。

## 为保证实验有效性增加的边界处理

1. **失败不是很大的任意 reward。** 推门超时、飞行高度越界、脱离或穿越失败记 `J=+∞`，因为没有完成到 M 的任务。CSV 中保留 `inf`；JSON 用 `null` 并配合 `success=false` 和原因。MuJoCo 数值警告单独终止运行，不当成物理失败。
2. **Probe 后不能继续使用关闭门时的路线成本。** Probe 是物理交互，会改变门角/角速度、无人机状态。首阶段使用理想观测到的完整机械状态，对每个候选保留其 I、τ_f，从这个共同状态重算 Door/Bypass 剩余时间。真实参数不会传给权重更新或决策函数。报告同时保存“只更新权重、保持原标签”的 p_D/D，以区分信息效应和物理状态改变效应。
3. **一次 Probe 不保证解决分歧。** 单次 Probe 后若 D 仍高，记录 unresolved，并执行明确的保守 Avoid。不会重新执行无休止的 Probe，也不会把接近 0.5 的 p_D 宣称为明确 Push。两种剩余路线同时失败的候选数和权重也会记录。
4. **初始等权时真值不能改变初始动作。** 同一候选域和地图下，Case A/B/C/E 的初始判断相同，因为真值尚未被观测。它们只能在 Probe 后根据响应产生不同的最终决策。直接 Push / 直接 Avoid 分支另外由门控验证测试覆盖；不能为了展示而提前把真实门“易推概率”塞给策略。
5. **需要报告 Probe 的时间开销。** 实验同时给出实际总时间和真实环境中不做 Probe 的最佳路线时间，记录 regret。高 G、高 D 是当前设计的交互选择原则，并不在数学上保证每次 Probe 都减少总时间；后续应做阈值、容差及测量噪声敏感性实验。

这里的 no-Probe oracle 只是两种已定义固定控制方案中更快者。Probe 自己也会推动门，因此其总时间可能比两条无 Probe 路线更短，regret 可以为负；这不能单独证明节省时间全部来自信息获取。执行轨迹 CSV 每 0.05 s 采样，而 F_peak 在每个 0.002 s 物理步上计算，二者的峰值可能不同。

`verify_agreement.py` 使用两个明确分开的可行域进行补充验证：低阻力域 `I∈[0.05,0.35], τ_f∈[0.01,0.15]`，以及高阻力域 `I∈[0.05,0.80], τ_f∈[6,8]`，每组仍有等权 1024 个 Sobol 世界。默认 A–E 的参数域和等权条件不被这些补充实验改变。

## Case A–E 和输出

| Case | θ*=[I,τ_f] | 地图 | 验证目的 |
| --- | --- | --- | --- |
| A | [0.071,0.035] | 大 G | 低阻力下的交互及恢复控制 |
| B | [0.673,6.317] | 大 G | 高阻力下的交互与路线选择 |
| C | [0.317,2.735] | 大 G | 高 G、高初始 D，先 Probe，再决策 |
| D | [0.437,2.473] | 小 G | 相同隐藏动力学域，低收益直接 Avoid |
| E | [0.437,2.473] | 大 G | Θ 内部且不在 Sobol 点集的真值，检验软更新 |

全部 θ* 都运行显式域内、非候选点检查；其中 E 位于内部。参数集中度用归一化参数距离和半径 0.10 邻域总权重衡量，同时报告观测一致性误差。两维响应可能存在非唯一辨识，不要求更新必须唯一锁定 θ*，也不把候选世界叫作 1024 扇真实门的数据。

输出保存到 `results/`：

```
libraries/<地图名_签名>/
    hypotheses.npz        候选参数、Probe 预测、路线时间、成功状态、缓存检查点
    hypotheses.csv        可用 Excel/PyCharm 分析的完整 1024 行库和标签
    probe_prefix.npz      每个候选在 Probe 各时刻的累计峰值力与门角预测
    metadata.json         参数域、配置、G、模型与源代码版本签名
continuations/<状态签名>/
    continuation_costs.*  同一 Probe 后状态下各候选剩余时间与标签
<本地运行时间>/
    report.html           可离线打开的汇总与权重变化播放/时间滑块
    results.json          Case 结果
    case_A/...            summary.json, weights.csv, probe_trace.csv, mission_trace.csv
                          belief_history.npz, live_timeline.csv, live_views/*.png
    case_B/...            同上
    ...
LATEST.txt                最近完整运行目录的绝对路径
```

源 XML、物理仿真代码、采样、地图、库与实验流程源码、完整配置（进程数除外）及 MuJoCo/NumPy 版本进入签名。修改控制器后会建立新候选库；continuation 还对共同观测状态做散列，并检查完整库签名及完成标志，防止混用旧成本。每个世界只共享相同准备状态；初始化/候选克隆后，运动始终由 MuJoCo 与旋翼控制产生。

`render_preview.py` 导出的 `scene.xml` 仅用于查看场景拼装，不是运行入口或地图配置源；无人机的 S 初始化、门参数和状态克隆由 `World` 完成。修改地图请编辑 `navlab/map.py`，实验请运行 `main.py`。

## 代码对应关系

| 模块 | 内容 |
| --- | --- |
| `navlab/sampling.py` | 二维 Sobol、候选域、非候选真值检查 |
| `navlab/physics.py` | 原无人机、门物理参数写入、统一 Probe、旋翼控制、真实路线完成时间 |
| `navlab/library.py` | 1024 世界进程仿真、版本缓存、路线标签、Probe 后候选 rollout |
| `navlab/belief.py` | prediction-observation consistency weighting、p_D、D、Push/Avoid/Probe |
| `navlab/probe_stream.py` | 标准 Probe 的时刻匹配预测与检查点缓存 |
| `navlab/live_belief.py` | 原始先验上的实时累计观测更新、权重历史与实时记录 |
| `navlab/live_views.py` | 每个场景的实际 MuJoCo 画面、实时权重、暂停/速度/切换/保存 |
| `navlab/map.py` | 同源地图、膨胀占据栅格、A*、几何代价差 G |
| `navlab/experiments.py` | A–E 真实测试、评估隔离、单次 Probe、总时间与后悔 |
| `navlab/report.py` | 无额外依赖的 HTML/SVG 汇总 |

实现依据：[MuJoCo 3.3.2 的 mj_setConst / mj_contactForce API](https://mujoco.readthedocs.io/en/3.3.2/APIreference/APIfunctions.html)、[MuJoCo 惯量与 frictionloss 定义](https://mujoco.readthedocs.io/en/3.3.2/XMLreference.html)、[Sobol 2^m 采样与覆盖性质](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)。本工程的 Sobol 实现只需 NumPy，不需要安装 SciPy。

## 当前实验边界

这是同一 MuJoCo 前向模型与测试模型下的第一阶段验证：建筑地图已知、门被识别、接触与自身状态测量理想、固定高度、无观测噪声或模型失配。证明的是候选物理假设、响应预测、软更新与 G+D 门控流程可以连贯执行。它不等于已经验证真实未知环境中的完整感知系统，或对箱子、软障碍的泛化；这些可以在保留 G=导航价值、D=动作分歧的框架下逐步扩展。

## 本次完整输出

2026-10-18 的完整 A–E 报告：[report.html](results/20261018_214025/report.html)。修改前后失败分布及逐场景结果：[recovery_audit.md](results/20261018_214025/recovery_audit.md)。实现、实际验证范围和未解决的两例回归见 [VALIDATION.md](VALIDATION.md)。
