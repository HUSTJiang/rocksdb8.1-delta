// delta/global_delete_count_table.cc

#include "delta/global_delete_count_table.h"

namespace ROCKSDB_NAMESPACE {

namespace {

bool InsertSortedUnique(std::vector<uint64_t>* ids, uint64_t id) {
  if (id == 0) {
    return false;
  }
  auto pos = std::lower_bound(ids->begin(), ids->end(), id);
  if (pos != ids->end() && *pos == id) {
    return false;
  }
  ids->insert(pos, id);
  return true;
}

bool EraseSortedIfPresent(std::vector<uint64_t>* ids, uint64_t id) {
  auto pos = std::lower_bound(ids->begin(), ids->end(), id);
  if (pos == ids->end() || *pos != id) {
    return false;
  }
  ids->erase(pos);
  return true;
}

}  // namespace

GlobalDeleteCountTable::GlobalDeleteCountTable(size_t num_shards)
    : shards_(num_shards) {}

bool GlobalDeleteCountTable::TrackPhysicalUnit(uint64_t cuid, uint64_t phys_id) {
  auto& shard = GetShard(cuid);
  std::unique_lock<std::shared_mutex> lock(shard.mutex);
  auto& entry = shard.table[cuid]; // Lazy Init
  
  if (InsertSortedUnique(&entry.tracked_phys_ids, phys_id)) {
    entry.ref_count++;
    return true;
  }
  return false;
}

void GlobalDeleteCountTable::ResetTracking(uint64_t cuid) {
    auto& shard = GetShard(cuid);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    auto it = shard.table.find(cuid);
    if (it != shard.table.end()) {
      it->second.ref_count = 0;
      it->second.tracked_phys_ids.clear();
    }
}

// void GlobalDeleteCountTable::UntrackPhysicalUnit(uint64_t cuid, uint64_t phys_id) {
//   std::unique_lock<std::shared_mutex> lock(mutex_);
//   auto it = table_.find(cuid);
//   if (it != table_.end()) {
//     it->second.tracked_phys_ids.erase(phys_id);
//     it->second.ref_count--;
//     // 如果计数归零且已标记删除的清理？
//     // if (it->second.is_deleted && it->second.tracked_phys_ids.empty()) {
//     //     table_.erase(it);
//     // }
//   }
// }

// 用于 L0Compaction 对 delete cuid 的清理
// void GlobalDeleteCountTable::UntrackFiles(uint64_t cuid, const std::vector<uint64_t>& file_ids) {
//   std::unique_lock<std::shared_mutex> lock(mutex_);
//   auto it = table_.find(cuid);
//   if (it != table_.end()) {
//     // 遍历本次 Compaction 的所有输入文件
//     for (uint64_t fid : file_ids) {
//       it->second.tracked_phys_ids.erase(fid);
//     }
//     // 检查是否归零且标记删除，如果是则清理条目
//     if (it->second.tracked_phys_ids.empty() && it->second.is_deleted) {
//       table_.erase(it);
//     }
//   }
// }

void GlobalDeleteCountTable::UntrackFiles(uint64_t cuid, const std::vector<uint64_t>& file_ids) {
  auto& shard = GetShard(cuid);
  std::unique_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it == shard.table.end()) return;

  auto& entry = it->second;
  auto& ids = entry.tracked_phys_ids; // 这是用于校验的 Vector

  for (uint64_t fid : file_ids) {
      if (EraseSortedIfPresent(&ids, fid)) {
          entry.ref_count--; // 同步扣减逻辑计数
      }
  }

  // 检查是否需要清理条目 (引用归零 且 标记删除)
  if (entry.ref_count <= 0 && entry.deleted_at_seqno.load(std::memory_order_relaxed) != kMaxSequenceNumber) {
      shard.table.erase(it);
  }
}

void GlobalDeleteCountTable::ApplyCompactionChange(
    uint64_t cuid, const std::vector<uint64_t>& input_files,
    const std::vector<uint64_t>& output_files) {
  auto& shard = GetShard(cuid);
  std::unique_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  bool inserted = false;
  if (it == shard.table.end()) {
    if (output_files.empty()) {
      return;
    }
    it = shard.table.emplace(std::piecewise_construct,
                             std::forward_as_tuple(cuid),
                             std::forward_as_tuple())
             .first;
    inserted = true;
  }
  auto& entry = it->second;

  for (uint64_t fid : input_files) {
    if (EraseSortedIfPresent(&entry.tracked_phys_ids, fid)) {
      entry.ref_count--;
    }
  }

  for (uint64_t fid : output_files) {
    if (InsertSortedUnique(&entry.tracked_phys_ids, fid)) {
      entry.ref_count++;
    }
  }

  // 2.3 检查清理条件：无文件引用 且 标记为删除
  if (entry.ref_count <= 0 && entry.deleted_at_seqno.load(std::memory_order_relaxed) != kMaxSequenceNumber) {
      shard.table.erase(it);
  } else if (inserted && entry.tracked_phys_ids.empty()) {
      shard.table.erase(it);
  }
}

void GlobalDeleteCountTable::ApplyFlushChange(uint64_t cuid,
                                              const std::vector<uint64_t>& input_phys_ids,
                                              uint64_t output_file) {
  auto& shard = GetShard(cuid);
  std::unique_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it == shard.table.end()) return;

  auto& entry = it->second;
  for (uint64_t phys_id : input_phys_ids) {
    if (EraseSortedIfPresent(&entry.tracked_phys_ids, phys_id)) {
      entry.ref_count--;
    }
  }

  if (InsertSortedUnique(&entry.tracked_phys_ids, output_file)) {
    entry.ref_count++;
  }

  if (entry.ref_count <= 0 &&
      entry.deleted_at_seqno.load(std::memory_order_relaxed) !=
          kMaxSequenceNumber) {
    shard.table.erase(it);
  }
}

bool GlobalDeleteCountTable::MarkDeleted(uint64_t cuid, SequenceNumber seq, bool* newly_deleted) {
  const auto& shard = GetShard(cuid);
  // atomic deleted_at_seqno shared_lock
  std::shared_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it != shard.table.end()) {
    SequenceNumber expected = kMaxSequenceNumber;
    bool exchanged = it->second.deleted_at_seqno.compare_exchange_strong(expected, seq, std::memory_order_acq_rel);
    if (newly_deleted) *newly_deleted = exchanged;
    return true; 
  }
  if (newly_deleted) *newly_deleted = false;
  return false;
}

bool GlobalDeleteCountTable::IsDeleted(uint64_t cuid, SequenceNumber read_seqno) const {
  const auto& shard = GetShard(cuid);
  std::shared_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);  
  if (it != shard.table.end()) {
    SequenceNumber at_seq = it->second.deleted_at_seqno.load(std::memory_order_acquire);
    return at_seq != kMaxSequenceNumber && at_seq <= read_seqno;
  }
  return false;
}

std::vector<std::pair<uint64_t, SequenceNumber>> GlobalDeleteCountTable::GetAllDeletedCuids() const {
  std::vector<std::pair<uint64_t, SequenceNumber>> result;
  for (size_t i = 0; i < shards_.size(); ++i) {
    const auto& shard = shards_[i];
    std::shared_lock<std::shared_mutex> lock(shard.mutex);
    for (const auto& kv : shard.table) {
      SequenceNumber seq = kv.second.deleted_at_seqno.load(std::memory_order_acquire);
      if (seq != kMaxSequenceNumber) {
        result.push_back({kv.first, seq});
      }
    }
  }
  return result;
}

int GlobalDeleteCountTable::GetRefCount(uint64_t cuid) const {
  const auto& shard = GetShard(cuid);
  std::shared_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it != shard.table.end()) {
    return it->second.GetRefCount();
  }
  return 0;
}

bool GlobalDeleteCountTable::IsTracked(uint64_t cuid) const {
  const auto& shard = GetShard(cuid);
  std::shared_lock<std::shared_mutex> lock(shard.mutex);
  return shard.table.find(cuid) != shard.table.end();
}


} // namespace ROCKSDB_NAMESPACE
