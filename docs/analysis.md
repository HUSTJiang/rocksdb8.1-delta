
# 热点SST和delta_diag.log生成缓慢

能看出一些比较明确的原因。我的判断是：**问题主要不在 `delta_diag.log` 这个 logger 本身，而在 8.1 迁移后，触发 HotSST 生成和日志输出的那条“冷路径全量 scan + 后台 delta worker”整体变慢了**。

先说最直接的一点：**`delta/diag_log.cc` 两边实现基本一样**。两个提交里的 `DiagLogf()` 都是在互斥锁里 `fputs()` 之后立刻 `fflush()`，生命周期日志 `LifecycleLogf()` 也是同样模式；所以“8.1 版本打印慢”**不是**因为你迁移时把日志代码写得更慢了。它更像是：上游事件发生得更慢，于是文件里的行也长得更慢。

为什么说是上游事件慢？因为两边代码里，**热点初始化和 metadata scan 都是显式走 cold path 的全量范围扫描**：`ProcessPendingHotCuids()` / `ProcessPendingMetadataScans()` 都会设置 `read_opts.delta_full_scan = true`、`read_opts.skip_hot_path = true`，然后 `NewIterator(read_opts, cfh)`，再用 `Seek/Next` 把这一段完整扫过去。也就是说，**HotSST 何时生成、`delta_diag.log` 何时持续刷出内容，直接取决于这条冷路径 scan 的吞吐**。这在 10.10 和 8.1 两边都是如此。

而 10.10 这边，你的 delta patch 明确改到了 `db/arena_wrapped_db_iter.cc/h`，在 iterator 初始化时保留了 `hotspot_manager`，并且有 `async_io` 相关处理；对应地，8.1 这次迁移的提交文件列表里**没有** `arena_wrapped_db_iter.*`，整个 patch 里也**没有** `async_io` 相关改动痕迹。这个差异很重要：它说明 10.10 版 delta 是接在一个更现代的 iterator/refresh 路径上的，而 8.1 迁移版至少从这次提交看，**没有把这部分 scan 相关接线完整带过去**。

这件事和 RocksDB 官方文档也对得上。官方文档说明，iterator scan 的性能高度依赖 **automatic readahead / prefetch**，而开启 `ReadOptions.async_io` 后，Seek 和 Next 的 I/O 可以在后台并行预取，减少阻塞；RocksDB 后续 10.x 版本的发布说明里也连续加了 **multiscan / prefetch / async prefetch** 相关优化。换句话说，**同样一套 delta 逻辑挂在 RocksDB 10.10 上，本来就更容易把冷路径 scan 跑快；迁回 8.1 后，scan 变慢，HotSST 和日志自然都会“晚出来”**。这是我目前看到的最像主因的地方。

还有一个次要但很现实的因素：**delta 后台任务是低优先级、单线程默认串行跑的**。8.1 里 `MaybeScheduleDeltaWork()` 把 delta 任务丢到 `Env::Priority::LOW`，并且 `max_delta_threads` 默认是 `1`；更关键的是，8.1 的 `has_work` 判断里还把 `HasPendingGDCTFlush()` 也算进去了。也就是说，**同一个低优先级 delta worker** 不仅要做热点 init scan、metadata scan、partial merge，还可能被 GDCT flush/compact 这种 housekeeping 抢走时间片。10.10 这边同样是 LOW 线程池，但 `has_work` 里我看到的是 init / metadata / partial merge，没有把 `HasPendingGDCTFlush()` 单独作为调度条件。这个差异会让 8.1 更容易出现“热点文件和日志都在等后台线程”的现象。

所以我会把原因按优先级排成这样：

第一，**8.1 冷路径 scan 吞吐低于 10.10**，这是你现在这个现象最核心的解释。因为 HotSST 和 `delta_diag.log` 的增长，都是建立在那几个 cold-path full scan 跑完、跑快的基础上的。([GitHub][2])

第二，**8.1 迁移时没有把 10.10 那套更完整的 iterator/refresh/async-IO 接线一起迁下来**，至少从这两个 commit 的差异看是这样，这会直接放大 scan-heavy 路径的性能差。([GitHub][3])

第三，**8.1 的 delta worker 更容易被单线程低优先级调度和 GDCT housekeeping 拖慢**，所以“文件生成慢、日志打印慢”会被放大成非常明显的体感差异。([GitHub][5])

你现在最值得先做的，不是盯着 `diag_log.cc`，而是先验证这三个点：

1. 在 `ProcessPendingHotCuids()` 和 `ProcessPendingMetadataScans()` 里，给 `iter->Seek()` 到扫描结束这一段单独打时间戳。
2. 把 8.1 的 `delta_options.max_delta_threads` 临时调到 2 或 4，看 HotSST 与 `delta_diag.log` 的“出现速度”是否明显改善。
3. 临时把 8.1 的 `HasPendingGDCTFlush()` 从 delta 调度条件里拿掉，或者把 GDCT flush 拆到独立 worker，看热点初始化速度是否回升。
4. 对照 10.10，把 `arena_wrapped_db_iter` 那一层的接线补齐，重点核查 iterator refresh 和 async/readahead 相关路径。