# 特征刷新/保持：旧校准片段的接口试验

原始SpikeFPN，旧Thun00，235输入时刻/225评测。两种门都使用之前完整观察下产生的因果掩码；选择后更新的是实际pre-smooth三内部步特征，未选位置保留上次刷新值，之后重跑原平滑/读出。
全开门235次全部精确等价；三项缓存状态契约单元测试通过。全部模型/分支仍执行，没有物理跳算。

以下数字是旧校准片段上的语义/工程比较，不是新录制组效果证据。相同置信度的FP可能不同；回顾性1FP点使用本片段标签，不能当部署阈值。

- dense / common_original_confidence：TP=184/866，FP=225，queries=225，threshold=0.7621667385101318。
- dense / retrospective_calibration_1FP：TP=184/866，FP=225，queries=225，threshold=0.7621667385101318。
- admit_ema / common_original_confidence：TP=181/866，FP=216，queries=225，threshold=0.7621667385101318。
- admit_ema / retrospective_calibration_1FP：TP=181/866，FP=225，queries=225，threshold=0.7547963261604309。
- admit_activity / common_original_confidence：TP=131/866，FP=36，queries=225，threshold=0.7621667385101318。
- admit_activity / retrospective_calibration_1FP：TP=135/866，FP=225，queries=225，threshold=0.544524073600769。
- reuse_ema / common_original_confidence：TP=178/866，FP=225，queries=225，threshold=0.7621667385101318。
- reuse_ema / retrospective_calibration_1FP：TP=179/866，FP=225，queries=225，threshold=0.7614784240722656。
- reuse_activity / common_original_confidence：TP=119/866，FP=481，queries=225，threshold=0.7621667385101318。
- reuse_activity / retrospective_calibration_1FP：TP=95/866，FP=176，queries=225，threshold=0.8485814332962036。

保持历史特征不是保留一个旧检测框；缓存位于平滑卷积之前，邻域和后续神经元仍参与计算。神经元按原协议每查询重置，缓存特征跨查询保持。未加入光流对齐、TTL或风险模块；年龄仅记录，不提供硬刷新上界。
该试验解决“代码接在什么位置、全开是否一致”问题；新数据上的效用/漂移/误报仍需后续独立实验。它不改变正在运行的SNN七组次级测试。
