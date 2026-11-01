# 四个新录制组与第二架构：冻结规则复核

日期：2026-10-30。新组为Zurich07_a/08_a/09_a/10_a，在事件下载与推理前固定。它们此前未用于本项目推理或方法选择，但官方标签包已在本地；不是官方测试集、外部预注册或跨城市独立样本。所有完整有效查询和反方向结果均保留，本轮之后这些组也已暴露。

## 模型、输入与接入边界

原始／selected116适配SpikeFPN采用原180ms事件窗、9个20ms带符号分箱、256letterbox和三次内部步，神经元每查询重置；三步pre-smoothing特征均值用于因果排序。两个检查点属于同一架构，一个适配顺序种子。

RVT-Tiny使用原官方1Mpx检查点严格加载；原生640×480事件每极性10bin直方图、cutoff255、底部补零至512×640。首次50ms，此后相邻图像查询间的事件包约50ms；显式LSTM状态贯穿完整序列（含warm-up），仅序列开始重置。三个实际FPN输入检测头的矩形网格用于排序，三类先argmax再保留car/pedestrian。它是新的流式迁移条件，不是旧3×60ms重置参考或官方1Mpx基准复现。

两种架构的编码、预训练、网格、时序状态和描述子不同；比较在各模型内部成立，不是ANN对SNN的检测器优劣实验。原始SNN共1344位置/3anchor，RVT共6720位置/1anchor。相同位置比例不是相同计算预算；所有骨干、状态和检测头完整执行，掩码只影响NMS前候选准入。

6个预测器×3种残差公式+4参照共22排序；20%/35%/60%容量+dense共67条件。EMA alpha0.8、lag3和空间错位规则保持冻结；RVT的roll按各自矩形高度／宽度分别计算。前10个完整查询warm-up统一排除，事件、EMA与状态均只用过去及当前信息。

## Thun预检和冻结阈值

- SpikeFPN原始：225个Thun评价查询，dense阈值0.762166739，TP/GT 184/866，FP 225；全部67策略的最终阈值NMS/FP预算核对通过。原始Spike缓存精确复现查询数225；全开/状态对齐案例3。
- SpikeFPN适配：225个Thun评价查询，dense阈值0.794795275，TP/GT 186/866，FP 225；全部67策略的最终阈值NMS/FP预算核对通过。原始Spike缓存精确复现查询数225；全开/状态对齐案例3。
- RVT流式：225个Thun评价查询，dense阈值0.169258207，TP/GT 437/866，FP 225；全部67策略的最终阈值NMS/FP预算核对通过。原始Spike缓存精确复现查询数0；全开/状态对齐案例3。

dense及逐策略阈值都只在Thun校准到每查询最多1个FP，候选最低置信度0.01；实际最终NMS结果检查后，在新组推理之前冻结。Spike dense阈值必须复现旧值。校准集FP相同不意味着新组FP相同；逐策略阈值变化也不是纯门控作用。

## 新组dense检测健康度

高证据保留仅条件于本模型原本命中的身份。下列完整GT召回、误报、大小子集与轨迹覆盖是必要限制，不用高Q掩盖dense漏检。

- zurich_city_07_a/SpikeFPN原始：1449查询，TP/GT 488/5793，召回8.42%，precision 17.37%，FPPI 1.6025；小目标召回4.03%，dense可检/GT轨迹24/98。
- zurich_city_07_a/SpikeFPN适配：1449查询，TP/GT 580/5793，召回10.01%，precision 14.99%，FPPI 2.2698；小目标召回6.00%，dense可检/GT轨迹26/98。
- zurich_city_07_a/RVT流式：1449查询，TP/GT 3155/5793，召回54.46%，precision 66.44%，FPPI 1.1001；小目标召回52.15%，dense可检/GT轨迹52/98。
- zurich_city_08_a/SpikeFPN原始：773查询，TP/GT 976/4392，召回22.22%，precision 44.79%，FPPI 1.5563；小目标召回3.73%，dense可检/GT轨迹36/62。
- zurich_city_08_a/SpikeFPN适配：773查询，TP/GT 1153/4392，召回26.25%，precision 44.95%，FPPI 1.8266；小目标召回8.24%，dense可检/GT轨迹38/62。
- zurich_city_08_a/RVT流式：773查询，TP/GT 1997/4392，召回45.47%，precision 78.65%，FPPI 0.7012；小目标召回27.47%，dense可检/GT轨迹42/62。
- zurich_city_09_a/SpikeFPN原始：1799查询，TP/GT 1257/7190，召回17.48%，precision 24.96%，FPPI 2.1006；小目标召回1.05%，dense可检/GT轨迹61/110。
- zurich_city_09_a/SpikeFPN适配：1799查询，TP/GT 1608/7190，召回22.36%，precision 28.40%，FPPI 2.2535；小目标召回5.59%，dense可检/GT轨迹61/110。
- zurich_city_09_a/RVT流式：1799查询，TP/GT 2762/7190，召回38.41%，precision 69.15%，FPPI 0.6848；小目标召回11.89%，dense可检/GT轨迹69/110。
- zurich_city_10_a/SpikeFPN原始：2301查询，TP/GT 1398/5466，召回25.58%，precision 19.66%，FPPI 2.4824；小目标召回5.34%，dense可检/GT轨迹32/50。
- zurich_city_10_a/SpikeFPN适配：2301查询，TP/GT 1547/5466，召回28.30%，precision 25.53%，FPPI 1.9609；小目标召回9.98%，dense可检/GT轨迹35/50。
- zurich_city_10_a/RVT流式：2301查询，TP/GT 3131/5466，召回57.28%，precision 74.09%，FPPI 0.4759；小目标召回42.87%，dense可检/GT轨迹37/50。

## 主对照：20%容量，同一dense冻结阈值

- zurich_city_07_a/SpikeFPN原始：current Q 85.66%，absolute Q 98.98%，差+13.32pp；TP差+65/FP差+66，FPPI 1.4658→1.5114；最长丢失8→1查询，联合护栏通过=False。
- zurich_city_07_a/SpikeFPN适配：current Q 73.62%，absolute Q 95.34%，差+21.72pp；TP差+123/FP差+249，FPPI 1.9959→2.1677；最长丢失11→2查询，联合护栏通过=False。
- zurich_city_07_a/RVT流式：current Q 62.73%，absolute Q 92.58%，差+29.86pp；TP差+941/FP差+315，FPPI 0.8157→1.0331；最长丢失90→31查询，联合护栏通过=False。
- zurich_city_08_a/SpikeFPN原始：current Q 88.22%，absolute Q 98.57%，差+10.35pp；TP差+96/FP差+101，FPPI 1.3907→1.5213；最长丢失5→1查询，联合护栏通过=False。
- zurich_city_08_a/SpikeFPN适配：current Q 84.30%，absolute Q 96.53%，差+12.23pp；TP差+134/FP差+105，FPPI 1.6016→1.7374；最长丢失10→2查询，联合护栏通过=False。
- zurich_city_08_a/RVT流式：current Q 79.42%，absolute Q 94.94%，差+15.52pp；TP差+307/FP差+89，FPPI 0.4269→0.5420；最长丢失26→10查询，联合护栏通过=False。
- zurich_city_09_a/SpikeFPN原始：current Q 79.16%，absolute Q 98.17%，差+19.01pp；TP差+234/FP差+419，FPPI 1.6820→1.9150；最长丢失8→4查询，联合护栏通过=False。
- zurich_city_09_a/SpikeFPN适配：current Q 76.37%，absolute Q 94.65%，差+18.28pp；TP差+290/FP差+609，FPPI 1.6203→1.9589；最长丢失5→3查询，联合护栏通过=False。
- zurich_city_09_a/RVT流式：current Q 85.37%，absolute Q 95.26%，差+9.88pp；TP差+274/FP差+359，FPPI 0.2946→0.4942；最长丢失21→13查询，联合护栏通过=False。
- zurich_city_10_a/SpikeFPN原始：current Q 87.41%，absolute Q 98.93%，差+11.52pp；TP差+155/FP差+744，FPPI 1.9591→2.2825；最长丢失6→3查询，联合护栏通过=False。
- zurich_city_10_a/SpikeFPN适配：current Q 79.19%，absolute Q 95.09%，差+15.90pp；TP差+239/FP差+526，FPPI 1.5393→1.7679；最长丢失19→6查询，联合护栏通过=False。
- zurich_city_10_a/RVT流式：current Q 77.58%，absolute Q 92.88%，差+15.30pp；TP差+473/FP差+252，FPPI 0.2873→0.3968；最长丢失34→16查询，联合护栏通过=False。

联合护栏按配置固定：保留数、全GT TP不降，FP和最长采样丢失段不升，至少一项严格改善；dense分母为空时记undefined。它是描述性工作点检查，不是安全保证或实用显著性证明。Q增加而FP增加必须称权衡。

## 预测MAE与分配效用的具名反例是否重现

MAE仅在同一模型／特征空间内比较，不能跨SNN和ANN比较数值大小。下面固定current公式及dense阈值，对比EMA与额外lag3；所有其他预测器／归一化仍保存在CSV。

- zurich_city_07_a/SpikeFPN原始：EMA MAE 0.081248，lag3 0.092363；Q 85.66%→86.68%，TP差+5/FP差+16，最长丢失8→12查询；MAE更差但Q更高=True。
- zurich_city_07_a/SpikeFPN适配：EMA MAE 0.077053，lag3 0.087521；Q 73.62%→78.45%，TP差+27/FP差+22，最长丢失11→7查询；MAE更差但Q更高=True。
- zurich_city_07_a/RVT流式：EMA MAE 0.108923，lag3 0.139929；Q 62.73%→72.52%，TP差+306/FP差+102，最长丢失90→56查询；MAE更差但Q更高=True。
- zurich_city_08_a/SpikeFPN原始：EMA MAE 0.078437，lag3 0.089088；Q 88.22%→92.21%，TP差+39/FP差+29，最长丢失5→6查询；MAE更差但Q更高=True。
- zurich_city_08_a/SpikeFPN适配：EMA MAE 0.075493，lag3 0.085670；Q 84.30%→89.94%，TP差+63/FP差+42，最长丢失10→10查询；MAE更差但Q更高=True。
- zurich_city_08_a/RVT流式：EMA MAE 0.104660，lag3 0.135198；Q 79.42%→86.28%，TP差+141/FP差+18，最长丢失26→14查询；MAE更差但Q更高=True。
- zurich_city_09_a/SpikeFPN原始：EMA MAE 0.078913，lag3 0.087529；Q 79.16%→87.27%，TP差+100/FP差+36，最长丢失8→5查询；MAE更差但Q更高=True。
- zurich_city_09_a/SpikeFPN适配：EMA MAE 0.077200，lag3 0.085469；Q 76.37%→81.03%，TP差+74/FP差+21，最长丢失5→5查询；MAE更差但Q更高=True。
- zurich_city_09_a/RVT流式：EMA MAE 0.109037，lag3 0.143592；Q 85.37%→89.50%，TP差+116/FP差+30，最长丢失21→20查询；MAE更差但Q更高=True。
- zurich_city_10_a/SpikeFPN原始：EMA MAE 0.078159，lag3 0.087072；Q 87.41%→91.99%，TP差+63/FP差+40，最长丢失6→6查询；MAE更差但Q更高=True。
- zurich_city_10_a/SpikeFPN适配：EMA MAE 0.076863，lag3 0.085422；Q 79.19%→84.81%，TP差+90/FP差+39，最长丢失19→19查询；MAE更差但Q更高=True。
- zurich_city_10_a/RVT流式：EMA MAE 0.106917，lag3 0.140752；Q 77.58%→82.85%，TP差+168/FP差+22，最长丢失34→28查询；MAE更差但Q更高=True。

## 活动量强参照：不自动归功于预测

同一模型、同一dense阈值和20%容量下，同时列当前活动、EMA历史活动、persistence绝对残差和EMA绝对残差。即使absolute优于current归一化，也不能跳过零预测／活动量对照来宣称预测有额外价值。

- zurich_city_07_a/SpikeFPN原始：current_activity Q 53.89%/FPPI 0.2560；ema_activity Q 28.48%/FPPI 0.0414；persistence/absolute Q 93.24%/FPPI 1.4389；ema/absolute Q 98.98%/FPPI 1.5114。
- zurich_city_07_a/SpikeFPN适配：current_activity Q 54.31%/FPPI 0.4003；ema_activity Q 30.34%/FPPI 0.0676；persistence/absolute Q 87.41%/FPPI 2.0386；ema/absolute Q 95.34%/FPPI 2.1677。
- zurich_city_07_a/RVT流式：current_activity Q 99.14%/FPPI 0.9876；ema_activity Q 90.36%/FPPI 0.5970；persistence/absolute Q 62.25%/FPPI 0.8351；ema/absolute Q 92.58%/FPPI 1.0331。
- zurich_city_08_a/SpikeFPN原始：current_activity Q 44.26%/FPPI 0.2820；ema_activity Q 23.36%/FPPI 0.0893；persistence/absolute Q 88.01%/FPPI 1.3997；ema/absolute Q 98.57%/FPPI 1.5213。
- zurich_city_08_a/SpikeFPN适配：current_activity Q 41.63%/FPPI 0.3506；ema_activity Q 24.63%/FPPI 0.1268；persistence/absolute Q 85.60%/FPPI 1.5938；ema/absolute Q 96.53%/FPPI 1.7374。
- zurich_city_08_a/RVT流式：current_activity Q 97.50%/FPPI 0.6481；ema_activity Q 83.38%/FPPI 0.4191；persistence/absolute Q 73.36%/FPPI 0.4450；ema/absolute Q 94.94%/FPPI 0.5420。
- zurich_city_09_a/SpikeFPN原始：current_activity Q 15.43%/FPPI 0.1256；ema_activity Q 1.11%/FPPI 0.0106；persistence/absolute Q 90.21%/FPPI 1.8155；ema/absolute Q 98.17%/FPPI 1.9150。
- zurich_city_09_a/SpikeFPN适配：current_activity Q 21.02%/FPPI 0.1884；ema_activity Q 3.73%/FPPI 0.0195；persistence/absolute Q 85.88%/FPPI 1.8449；ema/absolute Q 94.65%/FPPI 1.9589。
- zurich_city_09_a/RVT流式：current_activity Q 97.54%/FPPI 0.5842；ema_activity Q 86.35%/FPPI 0.4358；persistence/absolute Q 83.71%/FPPI 0.3396；ema/absolute Q 95.26%/FPPI 0.4942。
- zurich_city_10_a/SpikeFPN原始：current_activity Q 50.64%/FPPI 0.2586；ema_activity Q 30.04%/FPPI 0.0769；persistence/absolute Q 91.56%/FPPI 2.1912；ema/absolute Q 98.93%/FPPI 2.2825。
- zurich_city_10_a/SpikeFPN适配：current_activity Q 51.52%/FPPI 0.2295；ema_activity Q 32.39%/FPPI 0.0652；persistence/absolute Q 88.30%/FPPI 1.6536；ema/absolute Q 95.09%/FPPI 1.7679。
- zurich_city_10_a/RVT流式：current_activity Q 98.24%/FPPI 0.4294；ema_activity Q 90.90%/FPPI 0.3155；persistence/absolute Q 77.29%/FPPI 0.3251；ema/absolute Q 92.88%/FPPI 0.3968。

## 逐策略Thun校准后的实际新组误报

- zurich_city_07_a/SpikeFPN原始：current/absolute Q 85.66%/98.98%，FPPI 1.5804/1.5735，TP差+64/FP差-10。
- zurich_city_07_a/SpikeFPN适配：current/absolute Q 74.14%/95.34%，FPPI 2.1843/2.3168，TP差+120/FP差+192。
- zurich_city_07_a/RVT流式：current/absolute Q 65.10%/93.15%，FPPI 2.3727/1.7785，TP差+836/FP差-861。
- zurich_city_08_a/SpikeFPN原始：current/absolute Q 88.93%/98.57%，FPPI 1.5369/1.6054，TP差+84/FP差+53。
- zurich_city_08_a/SpikeFPN适配：current/absolute Q 84.65%/96.53%，FPPI 1.7413/1.8422，TP差+126/FP差+78。
- zurich_city_08_a/RVT流式：current/absolute Q 81.57%/95.94%，FPPI 1.0647/0.8525，TP差+245/FP差-164。
- zurich_city_09_a/SpikeFPN原始：current/absolute Q 80.03%/98.17%，FPPI 1.8660/2.0411，TP差+215/FP差+315。
- zurich_city_09_a/SpikeFPN适配：current/absolute Q 77.24%/94.71%，FPPI 1.8099/2.1273，TP差+278/FP差+571。
- zurich_city_09_a/RVT流式：current/absolute Q 87.73%/96.13%，FPPI 1.0578/0.9477，TP差+161/FP差-198。
- zurich_city_10_a/SpikeFPN原始：current/absolute Q 87.84%/98.93%，FPPI 2.1604/2.4072，TP差+144/FP差+568。
- zurich_city_10_a/SpikeFPN适配：current/absolute Q 79.70%/95.09%，FPPI 1.7010/1.8983，TP差+229/FP差+454。
- zurich_city_10_a/RVT流式：current/absolute Q 79.27%/93.84%，FPPI 0.8435/0.6788，TP差+406/FP差-379。

## 四个录制组的方向与范围

每个模型的四组分别报告，未把帧或三套模型当成独立科学样本；两个Spike检查点也不是两个架构。样本组数少，不提供帧独立p值或总体显著性／低概率安全界。

- SpikeFPN原始/ema/absolute：4组中有效4，Q差正/负/平为4/0/0；中位差+12.42pp，范围[+10.35,+19.01]pp，联合护栏通过0组。
- SpikeFPN原始/ema_lag3/current：4组中有效4，Q差正/负/平为4/0/0；中位差+4.29pp，范围[+1.02,+8.11]pp，联合护栏通过0组；具名lag排序反例4组。
- SpikeFPN适配/ema/absolute：4组中有效4，Q差正/负/平为4/0/0；中位差+17.09pp，范围[+12.23,+21.72]pp，联合护栏通过0组。
- SpikeFPN适配/ema_lag3/current：4组中有效4，Q差正/负/平为4/0/0；中位差+5.23pp，范围[+4.66,+5.64]pp，联合护栏通过0组；具名lag排序反例4组。
- RVT流式/ema/absolute：4组中有效4，Q差正/负/平为4/0/0；中位差+15.41pp，范围[+9.88,+29.86]pp，联合护栏通过0组。
- RVT流式/ema_lag3/current：4组中有效4，Q差正/负/平为4/0/0；中位差+6.07pp，范围[+4.13,+9.79]pp，联合护栏通过0组；具名lag排序反例4组。

## 核查、解释边界与论文状态

独立代数重建19641个模型—查询的完整评分（含校准集，warm-up用于状态）；独立排序／身份匹配／轨迹权重／删失区间核对通过6588项检查，覆盖2541444条条件—查询与197973个丢失段，校准最终NMS结果另行重算。共享观察到的描述子、坐标解码与上游NMS，不是独立传感器测量。
完整工作点、MAE、方向计数、轨迹权重、全GT与FP均归档。空dense或空GT分母记undefined，不按结果删除录制组。采样丢失的GT缺席、dense漏检、查询缺口及边界均为中断，不伪装恢复；跨度不等于连续物理盲区。
本轮只能检验冻结条件下的复核范围，不把“异质性”当作任何结果都能支持的假设。反方向和未重现的具名反例必须限制原解释。没有新增Risk/hybrid方法、匹配评价FP优越、硬件加速、神经形态能耗或闭环驾驶安全结论。
附图false_positive_tradeoffs_dense/policy在查看结果后增加，仅把已经冻结的20%工作点画为FPPI—保留率散点，不拟合前沿、不调阈值或改变主估计量；原报告生成源码与数值输入哈希保存于reporting_refinement.json。两幅图横轴为实际新组FPPI，不是计算量。
四个组来自同一公开DSEC训练集与城市，独立性和样本量仍有限；预训练、空间／状态协议与迁移误差影响跨架构解释。若根据本轮结果再调方法，四组已不能继续充当未触碰确认。当前main.tex未修改，本轮没有声称LaTeX编译。

## 图与完整数据

- [容量—保留率曲线](E:/SCI/experiments/confirmation_v01/confirmation_curves.png)；[SVG](E:/SCI/experiments/confirmation_v01/confirmation_curves.svg)。
- [MAE—保留率具名反例](E:/SCI/experiments/confirmation_v01/fidelity_utility_replication.png)；[SVG](E:/SCI/experiments/confirmation_v01/fidelity_utility_replication.svg)。此图须与误报图一起解释。
- [共同dense阈值下的实际误报权衡](E:/SCI/experiments/confirmation_v01/false_positive_tradeoffs_dense.png)；[SVG](E:/SCI/experiments/confirmation_v01/false_positive_tradeoffs_dense.svg)。
- [逐策略Thun冻结阈值下的实际误报权衡](E:/SCI/experiments/confirmation_v01/false_positive_tradeoffs_policy.png)；[SVG](E:/SCI/experiments/confirmation_v01/false_positive_tradeoffs_policy.svg)。
- [全部工作点CSV](E:/SCI/experiments/confirmation_v01/operating_points.csv)、[具名对照CSV](E:/SCI/experiments/confirmation_v01/primary_contrasts.csv)、[冻结设置](E:/SCI/code/configs/confirmation_v01.json)、[独立核对收据](E:/SCI/experiments/confirmation_v01/verification.json)。
