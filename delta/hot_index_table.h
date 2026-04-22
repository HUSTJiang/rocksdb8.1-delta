#pragma once

#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "delta/hot_data_buffer.h"
#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

struct DataSegment {
  uint64_t file_number;
  uint64_t file_size = 0;
  std::string first_key;
  std::string last_key;

  bool Valid() const { return !first_key.empty(); }
};

struct HotIndexEntry {
  std::vector<DataSegment> snapshot_segments;
  // std::vector<DataSegment> delta_segments;

  std::vector<DataSegment> deltas;
  // Scan-as-Compaction 标记
  std::vector<DataSegment> obsolete_deltas;

  bool HasSnapshot() const { return !snapshot_segments.empty(); }
};

class HotIndexTable {
 public:
  explicit HotIndexTable(
      const InternalKeyComparator* icmp,
      std::shared_ptr<HotSstLifecycleManager> lifecycle_manager,
      std::shared_ptr<Logger> info_log = nullptr,
      size_t num_shards = 128);
 
  void UpdateSnapshot(uint64_t cuid,
                      const std::vector<DataSegment>& new_segments);
  // scan-as-compaction -> add snapshots
  void AppendSnapshotSegment(uint64_t cuid, const DataSegment& segment);

  // 将内存中的 Snapshot (file_id = -1) 替换为真实 new_segment
  // 返回 true 表示成功找到并替换了 -1 段，false 表示未替换。
  // out_phys_overlap: 若非 nullptr，输出是否检测到与物理段重叠（但无 -1 段可替换）
  bool PromoteSnapshot(uint64_t cuid, const DataSegment& new_segment,
                       HotDataBuffer* buffer, const InternalKeyComparator* icmp,
                       bool* out_phys_overlap = nullptr);

  void AddDelta(uint64_t cuid, const DataSegment& segment);

  void MarkDeltasAsObsolete(uint64_t cuid,
                            const std::unordered_set<uint64_t>& visited_files);

  bool CheckAndRemoveObsoleteDeltas(uint64_t cuid,
                                    const std::vector<uint64_t>& input_files);

  bool GetEntry(uint64_t cuid, HotIndexEntry* entry) const;

  // 原子地复制 entry 并在 shared_lock 持有期间 Ref 所有物理 snapshot 段
  // 调用方负责最终 Unref（析构或 UnrefSegments）
  bool GetEntryAndRefSnapshots(uint64_t cuid, HotIndexEntry* out_entry) const;

  void RemoveCUID(uint64_t cuid);

  void UpdateDeltaIndex(uint64_t cuid, const std::vector<uint64_t>& input_files,
                        const DataSegment& new_delta);

  // 用于 CompactionIterator 决定是否 Skip
  bool IsDeltaObsolete(uint64_t cuid,
                       const std::vector<uint64_t>& input_files) const;

  // 用于清理cuid的 Obsolete Deltas
  void RemoveObsoleteDeltasForCUIDs(const std::unordered_set<uint64_t>& cuids,
                                    const std::vector<uint64_t>& input_files);

  // replace涉及的 Snapshot 或 Delta 段
  void ReplaceOverlappingSegments(
      uint64_t cuid, const DataSegment& new_segment,
      const std::vector<DataSegment>& obsolete_delta_segments);

  // PartialMerge 专用的无窗口原子替换方法。
  // 在单次 shard.mutex unique_lock
  void AtomicReplaceForPartialMerge(
      uint64_t cuid,
      const std::string& pm_range_first,
      const std::string& pm_range_last,
      const std::vector<DataSegment>& pm_sst_segs,
      const std::vector<DataSegment>& pm_promoted_segs,
      bool has_buf_data,
      const std::string& buf_min,
      const std::string& buf_max,
      const std::vector<DataSegment>& obsolete_delta_segments);

  // 统计与给定 key 范围重叠的 delta 数量
  size_t CountOverlappingDeltas(uint64_t cuid, const std::string& first_key,
                                const std::string& last_key) const;

  // kFullReplace
  void ClearAllForCuid(uint64_t cuid);

  // kPartialMerge: 获取与给定 key 范围重叠的 segments
  void GetOverlappingSegments(uint64_t cuid, const std::string& first_key,
                              const std::string& last_key,
                              std::vector<DataSegment>* overlapping_snapshots,
                              std::vector<DataSegment>* overlapping_deltas);

  void DumpToFile(const std::string& filename, const std::string& phase_label);

 private:
  struct Shard {
    mutable std::shared_mutex mutex;
    std::unordered_map<uint64_t, HotIndexEntry> table;
  };

  Shard& GetShard(uint64_t cuid) {
    return shards_[cuid % shards_.size()];
  }

  const Shard& GetShard(uint64_t cuid) const {
    return shards_[cuid % shards_.size()];
  }

  const InternalKeyComparator* icmp_;
  std::vector<Shard> shards_;
  std::shared_ptr<HotSstLifecycleManager> lifecycle_manager_;
  std::shared_ptr<Logger> info_log_;
};

}  // namespace ROCKSDB_NAMESPACE