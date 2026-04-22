#pragma once

#include <chrono>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include <atomic>

#include "rocksdb/env.h"
#include "rocksdb/options.h"
#include "rocksdb/rocksdb_namespace.h"
#include "rocksdb/slice.h"
#include "rocksdb/status.h"
#include "table/internal_iterator.h"

namespace ROCKSDB_NAMESPACE {

struct HotEntry {
  uint64_t cuid;
  std::string key;
  std::string value;

  // 用于排序
  bool operator<(const HotEntry& other) const { return key < other.key; }
};

// buffer block
struct HotDataBlock {
  // 按 CUID 分桶存储 entries，消除 NewIterator 时的线性过滤
  std::unordered_map<uint64_t, std::vector<HotEntry>> buckets;
  size_t current_size_bytes = 0;
  
  // 记录每个 CUID 的边界，用于快速判断
  struct Boundary {
     std::string min_key;
     std::string max_key;
  };
  std::unordered_map<uint64_t, Boundary> bounds;

  void Add(uint64_t c, const Slice& k, const Slice& v) {
    std::string k_str = k.ToString();
    std::string v_str = v.ToString();
    buckets[c].push_back({c, k_str, v_str});
    current_size_bytes += (k_str.size() + v_str.size());

    auto it = bounds.find(c);
    if (it == bounds.end()) {
      bounds[c] = {k_str, k_str};
    } else {
      if (k_str < it->second.min_key) it->second.min_key = k_str;
      if (k_str > it->second.max_key) it->second.max_key = k_str;
    }
  }

  void Sort(const InternalKeyComparator* icmp);  // 对每个 Bucket 内部按 Key 排序

  void Clear() {
    buckets.clear();
    current_size_bytes = 0;
    bounds.clear();
  }
};

class HotDataBuffer {
 public:
  explicit HotDataBuffer(size_t threshold_bytes = 1 * 1024 * 1024, uint32_t num_shards = 32);

  // 将数据追加到对应 CUID 的 buffer 中
  // 如果 buffer 大小超过阈值，返回 true (need Flush)
  bool Append(uint64_t cuid, const Slice& key, const Slice& value);

  // Active Buffer -> Immutable Queue
  bool RotateBuffer(const InternalKeyComparator* icmp);
  std::shared_ptr<HotDataBlock> GetFrontBlockForFlush();
  void PopFrontBlockAfterFlush();

  size_t GetTotalBufferedSize() const { return total_buffered_size_; }

  bool GetBoundaryKeys(uint64_t cuid, std::string* min_key,
                       std::string* max_key,
                       const InternalKeyComparator* icmp);

  // 检查当前活跃数据（尚未 Rotate 的 active_block）是否超过阈值
  bool ExceedsThreshold() const {
    return total_active_size_.load(std::memory_order_relaxed) >= threshold_bytes_;
  }

  // for reading
  InternalIterator* NewIterator(uint64_t cuid,
                                const InternalKeyComparator* icmp);

 private:
  // 将 Append 操作分片，减少并发写入冲突
  struct Shard {
    mutable std::mutex mutex;
    std::shared_ptr<HotDataBlock> active_block;
    std::deque<std::shared_ptr<HotDataBlock>> immutable_queue;
    std::atomic<size_t> buffered_size{0};
  };

  Shard& GetShard(uint64_t cuid) {
    return shards_[cuid % shards_.size()];
  }

  size_t threshold_bytes_;
  std::atomic<size_t> total_buffered_size_{0};
  std::atomic<size_t> total_active_size_{0};  // 仅统计尚未 Rotate 的活跃数据
  
  mutable std::mutex global_queue_mutex_;
  std::deque<std::shared_ptr<HotDataBlock>> immutable_queue_;

  std::vector<Shard> shards_;
};

class HotSstLifecycleManager {
 public:
  HotSstLifecycleManager(const Options& options) : 
    env_(options.env), info_log_(options.info_log),
    last_dump_time_(std::chrono::steady_clock::now()) {}

  void RegisterFile(uint64_t file_number, const std::string& file_path,
                    const std::string& link_path);
  void Ref(uint64_t file_number);
  void Unref(uint64_t file_number);

  // 输出当前所有追踪文件的状态快照（标记潜在泄露）
  void DumpStatus();

 private:
  Env* env_;
  std::shared_ptr<Logger> info_log_;
  std::mutex mutex_;

  // ref 超过此阈值时立即输出 LC_ANOMALY
  static constexpr int kAnomalyRefThreshold = 50;
  // 最近一次 ref/unref 后静止超过此时间（秒）且 ref > 0 则标记为可疑
  static constexpr int kStaleRefThresholdSec = 300;
  // 定期快照间隔（秒）
  static constexpr int kDumpIntervalSec = 30;

  struct FileState {
    std::string file_path;
    std::string link_path;
    int ref_count;
    int max_ref_seen;
    std::chrono::steady_clock::time_point registered_at;
    std::chrono::steady_clock::time_point last_activity_at;
  };

  std::unordered_map<uint64_t, FileState> files_;
  std::chrono::steady_clock::time_point last_dump_time_;

  // 如果距上次快照超过 kDumpIntervalSec 秒则 dump（调用方已持有 mutex_）
  void MaybeDumpStatus();
};

}  // namespace ROCKSDB_NAMESPACE