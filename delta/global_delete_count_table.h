// delta/global_delete_count_table.h

#pragma once
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <atomic>
#include <vector>

#include "rocksdb/rocksdb_namespace.h"
#include "rocksdb/types.h"
#include "db/dbformat.h"


namespace ROCKSDB_NAMESPACE {

struct GDCTEntry {
  std::vector<uint64_t> tracked_phys_ids;
  int32_t ref_count = 0;
  // Use atomic 
  mutable std::atomic<SequenceNumber> deleted_at_seqno{kMaxSequenceNumber};

  int GetRefCount() const { return ref_count; }
};

class GlobalDeleteCountTable {
 public:
  explicit GlobalDeleteCountTable(size_t num_shards = 128);

  // 【Scan 阶段调用】
  // 增加引用计数 (当 Scan 发现一个新的 SST/Memtable 包含该 CUID 时调用)
  bool TrackPhysicalUnit(uint64_t cuid, uint64_t phys_id);

  void UntrackFiles(uint64_t cuid, const std::vector<uint64_t>& file_ids);
  // full scan 时重置引用计数
  void ResetTracking(uint64_t cuid);
  //  检查是否已经追踪了该 CUID
  bool IsTracked(uint64_t cuid) const;

  // 【Delete 阶段调用】
  // 使用 compare_exchange 实现极速标记。通过 newly_deleted 避免重复落盘。
  bool MarkDeleted(uint64_t cuid, SequenceNumber seq, bool* newly_deleted = nullptr);

  // 【Compaction/Read 阶段调用】
  // 检查是否已删除 (用于 MVCC 数据过滤)
  bool IsDeleted(uint64_t cuid, SequenceNumber read_seqno) const;

  // 获取所有已经被标记删除的记录（用于日志收缩）
  std::vector<std::pair<uint64_t, SequenceNumber>> GetAllDeletedCuids() const;

  int GetRefCount(uint64_t cuid) const;

  void ApplyCompactionChange(uint64_t cuid,
                             const std::vector<uint64_t>& input_files,
                             const std::vector<uint64_t>& output_files);

  void ApplyFlushChange(uint64_t cuid,
                        const std::vector<uint64_t>& input_phys_ids,
                        uint64_t output_file);

 private:
  struct Shard {
    mutable std::shared_mutex mutex;
    std::unordered_map<uint64_t, GDCTEntry> table;
  };

  Shard& GetShard(uint64_t cuid) {
    return shards_[cuid % shards_.size()];
  }

  const Shard& GetShard(uint64_t cuid) const {
    return shards_[cuid % shards_.size()];
  }

  std::vector<Shard> shards_;
};

}  // namespace ROCKSDB_NAMESPACE
