#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include "delta/global_delete_count_table.h"
#include "delta/hot_data_buffer.h"
#include "delta/hot_index_table.h"
#include "delta/scan_frequency_table.h"
#include "rocksdb/options.h"
#include "rocksdb/rocksdb_namespace.h"
#include "rocksdb/slice.h"
#include "rocksdb/status.h"
#include "table/internal_iterator.h"

namespace ROCKSDB_NAMESPACE {

struct ScanContext {
  uint64_t current_cuid = 0;
  // 当前 cuid 已访问的文件 ID (FileNumber...)
  std::unordered_set<uint64_t> visited_phys_units;
};

// Scan-as-Compaction 策略
enum class ScanAsCompactionStrategy {
  kNoAction,
  kFullReplace,  // full scan
  kPartialMerge  // 部分合并：读取相关 snapshots + deltas + scan 数据，归并更新
};

// 异步 Partial Merge 任务
struct PartialMergePendingTask {
  uint64_t cuid;
  std::string scan_first_key;
  std::string scan_last_key;
  // 小 Scan 中精准抓取的 KV 对数据
  std::vector<std::pair<std::string, std::string>> scan_data;
};

class HotspotManager {
 public:
  // db_options: 用于初始化 SstFileWriter
  // data_dir: 生成 SST 文件存放的目录路径
  HotspotManager(const Options& db_options, const std::string& data_dir, const InternalKeyComparator* internal_comparator);

  ~HotspotManager() = default;

  // 返回 true 表示该 CUID 是热点
  // became_hot: 如果不为 nullptr，返回是否为首次成为热点
  bool RegisterScan(uint64_t cuid, bool is_full_scan,
                    bool* became_hot = nullptr);

  // 收集数据 (只有 RegisterScan 返回 true 时才调用此函数)
  bool BufferHotData(uint64_t cuid, const Slice& key, const Slice& value);

  // 为将要向 Buffer 写入 FullReplace 数据的 CUid 准备环境
  bool PrepareForFullReplace(uint64_t cuid);

  Status FlushBlockToSharedSST(
    std::shared_ptr<HotDataBlock> block,
    std::unordered_map<uint64_t, std::vector<DataSegment>>* output_segments);

  void TriggerBufferFlush(const char* source = "UNKNOWN",
                          uint64_t source_cuid = 0);

  bool ShouldTriggerScanAsCompaction(uint64_t cuid);

  // 根据扫描类型和涉及的 delta 数量，确定 Scan-as-Compaction 策略
  // is_full_scan: 是否为全版本扫描
  // scan_first_key/scan_last_key: 扫描的 key 范围（用于部分扫描的 delta 计算）
  ScanAsCompactionStrategy EvaluateScanAsCompactionStrategy(
      uint64_t cuid, bool is_full_scan, const std::string& scan_first_key = "",
      const std::string& scan_last_key = "",
      size_t* out_involved_delta_count = nullptr);

  // 计算给定 key 范围涉及的 delta 数量
  size_t CountInvolvedDeltas(uint64_t cuid, const std::string& first_key,
                             const std::string& last_key);

  void FinalizeScanAsCompaction(
      uint64_t cuid, const std::unordered_set<uint64_t>& visited_files,
      const std::string& scan_first_key, const std::string& scan_last_key);

  // 带策略的 Finalize 方法，支持不同的策略处理
  void FinalizeScanAsCompactionWithStrategy(
      uint64_t cuid, ScanAsCompactionStrategy strategy,
      const std::string& scan_first_key = "",
      const std::string& scan_last_key = "",
      const std::unordered_set<uint64_t>& visited_files = {},
      const std::vector<std::pair<std::string, std::string>>& scan_data = {});

  bool IsCuidDeleted(uint64_t cuid, SequenceNumber read_seqno) { return delete_table_.IsDeleted(cuid, read_seqno); }

  bool IsHot(uint64_t cuid);

  // 拦截 Delete 操作并持久化
  Status InterceptDelete(const Slice& key, SequenceNumber seq, bool sync);

  // 持久化 GDCT
  Status PersistDelete(uint64_t cuid, SequenceNumber seq, bool sync);

  // 恢复 GDCT
  void RecoverGDCT();

  // 定期合并与刷盘 gdct.log 文件 (背景线程调用)
  void CompactAndFlushGDCTLogIfNeeded();

  uint64_t ExtractCUID(const Slice& key);

  // compaction 结束之后更新delta表
  void UpdateCompactionDelta(uint64_t cuid,
                             const std::vector<uint64_t>& input_files,
                             uint64_t output_file_number,
                             const std::string& first_key,
                             const std::string& last_key);

  // CompactionIterator 使用 否应该跳过
  bool ShouldSkipObsoleteDelta(uint64_t cuid,
                               const std::vector<uint64_t>& input_files);

  // CompactionJob cleanup：obsolete
  void CleanUpMetadataAfterCompaction(
      const std::unordered_set<uint64_t>& involved_cuids,
      const std::vector<uint64_t>& input_files);

  std::string GenerateSstFileName(uint64_t cuid);

  void UpdateCompactionRefCount(
      uint64_t cuid, const std::vector<uint64_t>& input_files_verified,
      const std::vector<uint64_t>& output_files_verified) {
    delete_table_.ApplyCompactionChange(cuid, input_files_verified,
                                        output_files_verified);
  }

  void UpdateFlushRefCount(uint64_t cuid,
                          const std::vector<uint64_t>& input_phys_ids,
                          uint64_t output_file) {
    delete_table_.ApplyFlushChange(cuid, input_phys_ids, output_file);
  }

  bool GetHotIndexEntry(uint64_t cuid, HotIndexEntry* out_entry) {
    return index_table_.GetEntry(cuid, out_entry);
  }

  // 原子地复制 entry 并在 shared_lock 内 Ref 所有物理 snapshot 段
  // 返回的 out_entry->snapshot_segments 已预 Ref，调用方负责最终 Unref
  bool GetHotIndexEntryAndRefSnapshots(uint64_t cuid, HotIndexEntry* out_entry) {
    return index_table_.GetEntryAndRefSnapshots(cuid, out_entry);
  }

  InternalIterator* NewBufferIterator(uint64_t cuid,
                                      const InternalKeyComparator* icmp);

  // Getters for db_impl access
  const std::string& GetDataDir() const { return data_dir_; }
  std::shared_ptr<HotSstLifecycleManager> GetLifecycleManager() {
    return lifecycle_manager_;
  }

  HotIndexTable& GetIndexTable() { return index_table_; }

  GlobalDeleteCountTable& GetDeleteTable() { return delete_table_; }

  // 将 CUID 加入待初始化队列 (首次成为热点时调用)
  void EnqueueForInitScan(uint64_t cuid);

  // 获取并清空部分待初始化队列 (由 DBImpl 后台线程调用)
  std::vector<uint64_t> PopPendingInitCuids(size_t max_count = 32);

  // 检查是否有待处理的初始化任务
  bool HasPendingInitCuids() const;

  // 将 CUID 加入待补全元数据的扫描队列
  void EnqueueMetadataScan(uint64_t cuid);

  // 获取并部清空待补全元数据的扫描队列
  std::vector<uint64_t> PopPendingMetadataScans(size_t max_count = 32);

  // 检查是否有待处理的物理单元计数扫描任务
  bool HasPendingMetadataScans() const;

  // CUID Locking for background tasks (PartialMerge / InitScan)
  bool TryLockCuid(uint64_t cuid);
  void UnlockCuid(uint64_t cuid);

  void DebugDump(const std::string& label) {
    std::string dump_file =
        "/home/wam/HWKV/rocksdb-delta/build/a_test_db/a_mgr.log";
    index_table_.DumpToFile(dump_file, label);
  }

 private:
  Status InitGDCTLog();
  Status FlushGDCTLogBuffer();

 private:
  Options db_options_;
  std::string data_dir_;
  const InternalKeyComparator* internal_comparator_;

  std::shared_ptr<HotSstLifecycleManager> lifecycle_manager_;
  HotDataBuffer buffer_;
  HotIndexTable index_table_;

  ScanFrequencyTable frequency_table_;
  GlobalDeleteCountTable delete_table_;

  std::unique_ptr<WritableFile> gdct_log_writer_;
  mutable std::mutex gdct_log_mutex_;
  mutable std::mutex flush_mutex_;  // 新增：保证 TriggerBufferFlush 线程安全
  
  // Asynchronous extreme performance buffer for deletions
  std::vector<std::pair<uint64_t, SequenceNumber>> gdct_append_buffer_;
  std::mutex gdct_append_mutex_;
  std::atomic<uint64_t> pending_gdct_records_{0};
  std::atomic<uint64_t> last_gdct_flush_time_us_{0};
  std::atomic<uint64_t> last_gdct_compact_time_us_{0};

  // 暂存正在进行的 Scan 所生成的 SST 片段
  // FinalizeScanAsCompaction -> IndexTable
  std::unordered_map<uint64_t, std::vector<DataSegment>> pending_snapshots_;
  std::mutex pending_mutex_;
  // 待补全物理 ID 计数的扫描队列
  std::vector<uint64_t> pending_metadata_scans_;
  mutable std::mutex pending_metadata_mutex_;
  // 待初始化全量扫描的 CUID 队列
  std::vector<uint64_t> pending_init_cuids_;
  mutable std::mutex pending_init_mutex_;
  // Partial Merge 处理队列
  std::deque<PartialMergePendingTask> partial_merge_queue_;
  mutable std::mutex partial_merge_mutex_;
  // 正在进行的 CUID 操作 (防止并发)
  std::unordered_set<uint64_t> in_progress_cuids_;
  mutable std::mutex in_progress_mutex_;
  // PartialMerge 进行期间（BufferHotData ~ ReplaceOverlappingSegments），
  // 跨线程 TriggerBufferFlush 对该 CUID 产生的 SST 暂存于此
  std::unordered_map<uint64_t, std::vector<DataSegment>> pm_pending_snapshots_;
  // PM 活跃期间 PromoteSnapshot 成功消费的 SST 记录于此。
  // AtomicReplaceForPartialMerge 步骤① 需要保留这些 SST，不能将其移除。
  // (PromoteSnapshot 成功后 SST 已在 snapshot 中且正确 Ref，AtomicReplace 不额外 Ref/Unref)
  std::unordered_map<uint64_t, std::vector<DataSegment>> pm_promoted_snapshots_;
  std::mutex pm_pending_mutex_;  // 同一把锁保护 pm_pending 和 pm_promoted

 public:
  // 加入 Partial Merge 队列 (前台调用，不阻塞)
  void EnqueuePartialMerge(
      uint64_t cuid, const std::string& scan_first_key,
      const std::string& scan_last_key,
      const std::vector<std::pair<std::string, std::string>>& scan_data);

  // 检查是否有待处理的 Partial Merge 任务
  bool HasPendingPartialMerge() const;

  // 检查是否有待刷盘的 GDCT 日志 (轻量级原子检查)
  bool HasPendingGDCTFlush() const {
    return pending_gdct_records_.load(std::memory_order_relaxed) > 0;
  }

  // 取出一个待处理的 Partial Merge 任务 (供 db_impl 调用)
  bool PopPendingPartialMerge(PartialMergePendingTask* task);

  // 向 pm_pending_snapshots_ 注册该 cuid 的空列表
  void RegisterPmPending(uint64_t cuid);

  // 取出 pm_pending_snapshots_[cuid] 中当前积累的 SSTs，重置为空 vector 但不删除 cuid
  std::vector<DataSegment> SwapOutPmPending(uint64_t cuid) {
    std::lock_guard<std::mutex> lock(pm_pending_mutex_);
    auto it = pm_pending_snapshots_.find(cuid);
    if (it == pm_pending_snapshots_.end()) return {};
    auto result = std::move(it->second);
    it->second.clear();  // 重置为空，保留 key（entry 继续活跃）
    return result;
  }

  // 取出 pm_promoted_snapshots_[cuid] 并清空（PM 结束时调用，传给 AtomicReplace）
  std::vector<DataSegment> SwapOutPmPromoted(uint64_t cuid) {
    std::lock_guard<std::mutex> lock(pm_pending_mutex_);
    auto it = pm_promoted_snapshots_.find(cuid);
    if (it == pm_promoted_snapshots_.end()) return {};
    auto result = std::move(it->second);
    pm_promoted_snapshots_.erase(it);
    return result;
  }

  // 查询 buffer 中是否有该 CUID 的数据，以及数据的实际范围。
  // 用于为 AtomicReplaceForPartialMerge 提供准确的 -1 段范围，
  // 避免 -1 段声称覆盖已被 flush 到物理 SST 的范围。
  bool GetBufferBoundaryKeys(uint64_t cuid,
                             std::string* min_key,
                             std::string* max_key) {
    return buffer_.GetBoundaryKeys(cuid, min_key, max_key, internal_comparator_);
  }

  // 检查当前活跃 buffer 总大小是否达到 flush 阈值
  bool BufferExceedsThreshold() const {
    return buffer_.ExceedsThreshold();
  }

  void FinalizePmPendingSnapshots(uint64_t cuid);
};

}  // namespace ROCKSDB_NAMESPACE
