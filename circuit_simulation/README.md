# SOT 五端口记忆与方向判别电路仿真

本包保留三个主程序及其运行必需依赖，不含交互 notebook 或历史版本。用于生成 12345、321、345、12333 四组位置序列的完整电路波形，以及三个读取周期的控制时序细节。

## 如何运行

需要 Python 3.10+、ngspice，以及 requirements.txt 中的 Python 库。ngspice 是独立电路仿真软件，不通过该 requirements 文件安装；安装后须能在终端运行 `ngspice --version`。

在 circuit_simulation 目录运行：

```bash
python3 -m pip install -r requirements.txt
python3 run_all.py
```

`run_all.py` 按顺序执行三个阶段，并在任一步失败时停止。也可从 circuit_simulation 目录依次运行：

```bash
python3 scripts/generate_initialized_sequence_cases.py
python3 scripts/run_initialized_full_chain_combined_enable.py
python3 scripts/plot_three_rd_calok_combined_timing.py
```

请保留目录结构，不要单独挪动三个程序。结果写入 results/，重新运行会覆盖其中同名的生成结果，不会修改原研究目录。

## 文件说明

| 文件 / 目录 | 作用 |
|---|---|
| scripts/generate_initialized_sequence_cases.py | 生成初始化、WR/WR′ 和记忆波形激励 |
| scripts/run_initialized_full_chain_combined_enable.py | 构建网表、运行 ngspice、绘制完整波形 |
| scripts/plot_three_rd_calok_combined_timing.py | 运行三个 RD 周期的细节仿真并绘图 |
| memory_model/five_port_position_io.py | 位置到电流、记忆响应的接口，含同极性峰值保持 |
| memory_model/five_port_requested_memory.py | RequestedMemorySOT 状态规则及其函数依赖 |
| templates/full_chain_ad783_like_opamp_sha_rd_neutral_2inv_calok_behavioral_wr_logic.cir | 完整电路模板 |
| models/AD783_like_SHA_opamp.lib | 自建采样保持宏模型；不是 ADI 官方器件模型 |
| models/TLV3501.lib | 现有比较器模型文件 |
| models/generic_opamp.lib | 通用运放模型 |
| VALIDATION.md | 本次实际运行验证记录 |

## 怎么看结果

完整波形：`results/initialized_full_chain_<序列>_combined_dir_enable_waveforms.png`（另有 SVG）。

- 12345：校准后输出 Right。
- 321：校准后输出 Left。
- 345：校准后输出 Right。
- 12333：移动时输出 Right，重复位置时输出 Neutral。

局部时序：`results/combined_enable_three_rd_timing_large_text.png`（另有 SVG）。第一 RD 校准，下降沿置位 CAL_OK，后续 RD 通过延迟使能窗口输出方向。

数值文件虽然扩展名为 .csv，但 ngspice 的 `wrdata` 原始输出使用空白分隔；请用 `numpy.loadtxt` 或对应脚本的读取函数解析。激励生成器产生的小型 CSV 则为逗号分隔。

## 模型与范围

信号链：位置序列 → Python 五端口 SOT 记忆响应 → PWL 电压源 → 当前/前值采样保持 → 差分比较 → Right/Left/Neutral。

初始化 WR1+WR′5；输入电压为 34 V；红支平移 +0.95 V，四路记忆贡献求和后输入电路。差分共模为 2.50 V，比较死区为 ±20 mV。SOT 行为由 Python 预计算，后级由 ngspice 做连续瞬态仿真；它不是 SOT 微磁动力学仿真。

此包保留电路流程原有的五端口记忆接口，包含“较弱或相等的同极性输入保持状态”的包装逻辑，没有将其替换为 GitHub 简化核心包的单次反转接口。

本次整理只做两类可移植性处理：将绝对模型路径改成包内定位；将通用运放库放到生成网表能正确找到的相对目录。另增加一键运行入口和文档，没有调整原始电路参数或记忆算法。

模型库保留原文件内容与注释，未添加或推定新的授权许可证。

仓库直接提供程序、模型、生成网表和波形图；较大的原始瞬态 CSV 在运行时生成，不逐个上传。
