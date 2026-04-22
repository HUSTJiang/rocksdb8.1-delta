#include "delta/hot_data_buffer.h"
#include "delta/diag_log.h"
#include "delta/delta_perf_counters.h"
#include <algorithm>
#include "logging/logging.h"

namespace ROCKSDB_NAMESPACE {

void HotDataBlock::Sort(const InternalKeyComparator* icmp) {
  for (auto& pair : buckets) {
     std::sort(pair.second.begin(), pair.second.end(), 
               [icmp](const HotEntry& a, const HotEntry& b) {
                 return icmp->Compare(a.key, b.key) < 0;
               });
  }
}

HotDataBuffer::HotDataBuffer(size_t threshold_bytes, uint32_t num_shards)
    : threshold_bytes_(threshold_bytes), total_buffered_size_(0), shards_(num_shards) {
  for (size_t i = 0; i < shards_.size(); ++i) {
    shards_[i].active_block = std::make_shared<HotDataBlock>();
  }
}

bool HotDataBuffer::Append(uint64_t cuid, const Slice& key,
                           const Slice& value) {
  Shard& shard = GetShard(cuid);
  size_t entry_size = key.size() + value.size();
  
  {
    std::lock_guard<std::mutex> lock(shard.mutex);
    shard.active_block->Add(cuid, key, value);
    shard.buffered_size += entry_size;
  }
  
  total_buffered_size_.fetch_add(entry_size);
  // 使用 active 计数器触发 Flush，确保只有积攒够活跃数据才 Rotate
  size_t current_active = total_active_size_.fetch_add(entry_size) + entry_size;
  return current_active >= threshold_bytes_;
}

bool HotDataBuffer::RotateBuffer(const InternalKeyComparator* icmp) {
  bool has_data = false;

  // phase 1：将 active_block 替换为 new block，并转入暂存队列
  for (size_t i = 0; i < shards_.size(); ++i) {
    std::lock_guard<std::mutex> lock(shards_[i].mutex);
    if (shards_[i].active_block->buckets.empty()) continue;
    
    has_data = true;
    auto old_block = shards_[i].active_block;
    shards_[i].active_block = std::make_shared<HotDataBlock>();
    shards_[i].buffered_size = 0;
 
    shards_[i].immutable_queue.push_back(old_block);
  }
 
  if (!has_data) return false;
 
  // phase 2：stack combined_block，在双重锁的保护下，防止读取数据缺失
  auto combined_block = std::make_shared<HotDataBlock>();
  
  std::lock_guard<std::mutex> global_lock(global_queue_mutex_);
  
  // [DIAG] 记录合并开始
  // fprintf(stderr, "[DIAG_GAP] RotateBuffer Merge START. GlobalQ size: %zu\n", immutable_queue_.size());
  
  for (size_t i = 0; i < shards_.size(); ++i) {
    std::lock_guard<std::mutex> shard_lock(shards_[i].mutex);
    
    while (!shards_[i].immutable_queue.empty()) {
      auto old_block = shards_[i].immutable_queue.front();
      shards_[i].immutable_queue.pop_front();
      
      for (auto& pair : old_block->buckets) {
        uint64_t cuid = pair.first;
        auto& target_bucket = combined_block->buckets[cuid];
        
        if (target_bucket.empty()) {
          // Zero-Copy
          target_bucket = std::move(pair.second);
        } else {
          // 其他情况极低概率的合并补偿
          target_bucket.insert(target_bucket.end(), 
                               std::make_move_iterator(pair.second.begin()), 
                               std::make_move_iterator(pair.second.end()));
        }
        
        // 合并边界信息
        auto b_it = old_block->bounds.find(cuid);
        if (b_it != old_block->bounds.end()) {
          auto& target_bounds = combined_block->bounds[cuid];
          if (target_bounds.min_key.empty() || icmp->Compare(b_it->second.min_key, target_bounds.min_key) < 0) {
             target_bounds.min_key = b_it->second.min_key;
          }
          if (target_bounds.max_key.empty() || icmp->Compare(b_it->second.max_key, target_bounds.max_key) > 0) {
             target_bounds.max_key = b_it->second.max_key;
          }
        }
      }
      combined_block->current_size_bytes += old_block->current_size_bytes;
      total_active_size_.fetch_sub(old_block->current_size_bytes);
    }
  }

  // sort 确保 Reader 有序
  combined_block->Sort(icmp);
  immutable_queue_.push_back(combined_block);
  // [DIAG] 记录合并结束
  // fprintf(stderr, "[DIAG_GAP] RotateBuffer Merge END. Block added to GlobalQ.\n");
  return true;
}

std::shared_ptr<HotDataBlock> HotDataBuffer::GetFrontBlockForFlush() {
  std::lock_guard<std::mutex> lock(global_queue_mutex_);
  if (immutable_queue_.empty()) return nullptr;
  // fprintf(stderr, "[DIAG_GAP] GetFrontBlockForFlush called. GlobalQ size: %zu\n", immutable_queue_.size());
  return immutable_queue_.front();
}

void HotDataBuffer::PopFrontBlockAfterFlush() {
  std::shared_ptr<HotDataBlock> block;
  {
    std::lock_guard<std::mutex> lock(global_queue_mutex_);
    if (immutable_queue_.empty()) return;
    block = immutable_queue_.front();
    immutable_queue_.pop_front();
  }
  if (block) {
    total_buffered_size_.fetch_sub(block->current_size_bytes);
  }
}

bool HotDataBuffer::GetBoundaryKeys(uint64_t cuid, std::string* min_key,
                                    std::string* max_key,
                                    const InternalKeyComparator* icmp) {
  bool found = false;
  auto merge_bounds = [&](const HotDataBlock* block) {
    if (!block) return;
    auto it = block->bounds.find(cuid);
    if (it != block->bounds.end()) {
      if (!found) {
        *min_key = it->second.min_key;
        *max_key = it->second.max_key;
        found = true;
      } else {
        if (icmp->Compare(it->second.min_key, *min_key) < 0) *min_key = it->second.min_key;
        if (icmp->Compare(it->second.max_key, *max_key) > 0) *max_key = it->second.max_key;
      }
    }
  };

  Shard& shard = GetShard(cuid);

  // 先 global 后 shard
  std::unique_lock<std::mutex> global_lock(global_queue_mutex_);
  std::unique_lock<std::mutex> shard_lock(shard.mutex);

  // 1. 检查全局不可变队列
  for (const auto& block : immutable_queue_) {
    merge_bounds(block.get());
  }

  // 2. 检查特定分片的活跃块及其暂存队列
  for (const auto& block : shard.immutable_queue) {
    merge_bounds(block.get());
  }
  merge_bounds(shard.active_block.get());

  return found;
}

// --------------------- HotDataBuffer Iterator --------------------- //

class HotDataBufferIterator : public InternalIterator {
 public:
  explicit HotDataBufferIterator(std::vector<HotEntry>&& entries,
                                 const InternalKeyComparator* icmp)
      : entries_(std::move(entries)), idx_(0), icmp_(icmp) {
  }

  bool Valid() const override { return idx_ < entries_.size(); }
  void SeekToFirst() override { idx_ = 0; }
  void SeekToLast() override {
    idx_ = entries_.empty() ? 0 : entries_.size() - 1;
  }

  void Seek(const Slice& target) override {
    auto it = std::lower_bound(entries_.begin(), entries_.end(), target,
                               [this](const HotEntry& entry, const Slice& val) {
                                 return icmp_->Compare(entry.key, val) < 0;
                               });
    idx_ = std::distance(entries_.begin(), it);
  }

  void SeekForPrev(const Slice& target) override {
    Seek(target);
    if (!Valid()) SeekToLast();
    while (Valid() && icmp_->Compare(entries_[idx_].key, target) > 0) {
      Prev();
    }
  }

  void Next() override { if (idx_ < entries_.size()) idx_++; }
  void Prev() override { if (idx_ > 0) idx_--; else idx_ = entries_.size(); }

  Slice key() const override { return entries_[idx_].key; }
  Slice value() const override { return entries_[idx_].value; }
  Status status() const override { return Status::OK(); }
  uint64_t GetPhysicalId() override { return 0; }

 private:
  std::vector<HotEntry> entries_;
  size_t idx_;
  const InternalKeyComparator* icmp_;
};

InternalIterator* HotDataBuffer::NewIterator(
    uint64_t cuid, const InternalKeyComparator* icmp) {
  
  std::vector<HotEntry> filtered_entries;
  Shard& shard = GetShard(cuid);

  {
    std::unique_lock<std::mutex> global_lock(global_queue_mutex_);
    std::unique_lock<std::mutex> shard_lock(shard.mutex);

    // 1. Immutable
    for (const auto& block : immutable_queue_) {
      auto it = block->buckets.find(cuid);
      if (it != block->buckets.end()) {
        filtered_entries.insert(filtered_entries.end(), it->second.begin(), it->second.end());
      }
    }

    // 2. active + rotating 中的暂存块
    for (const auto& block : shard.immutable_queue) {
       auto it = block->buckets.find(cuid);
       if (it != block->buckets.end()) {
         filtered_entries.insert(filtered_entries.end(), it->second.begin(), it->second.end());
       }
    }
    auto it = shard.active_block->buckets.find(cuid);
    if (it != shard.active_block->buckets.end()) {
      filtered_entries.insert(filtered_entries.end(), it->second.begin(), it->second.end());
    }
  } // two lock ends

  // if (cuid == 1003) {
  //     if (filtered_entries.empty()) {
  //         Shard& s = GetShard(cuid);
  //         fprintf(stderr, "[DIAG_DATA] CUID 1003 EMPTY! ShardQ: %zu, GlobalQ: %zu, ActiveBuckets: %zu\n",
  //                 s.immutable_queue.size(), immutable_queue_.size(), s.active_block->buckets.size());
  //     } else {
  //         std::string first_hex = Slice(filtered_entries.front().key).ToString(true);
  //         std::string last_hex = Slice(filtered_entries.back().key).ToString(true);
  //         fprintf(stderr, "[DIAG_DATA] CUID 1003 size=%zu, FirstHex: %s\n", 
  //                 filtered_entries.size(), first_hex.substr(0, 60).c_str());
  //         fprintf(stderr, "[DIAG_DATA] CUID 1003 LastHex: %s\n", 
  //                 last_hex.substr(0, 60).c_str());
  //     }
  // }

  // 不同 block 之间可能重叠？
  g_buffer_iterator_build_count.fetch_add(1, std::memory_order_relaxed);
  g_buffer_iterator_rows_materialized.fetch_add(
      static_cast<uint64_t>(filtered_entries.size()), std::memory_order_relaxed);
  std::sort(filtered_entries.begin(), filtered_entries.end(),
            [icmp](const HotEntry& a, const HotEntry& b) {
              return icmp->Compare(Slice(a.key), Slice(b.key)) < 0;
            });
  // [DIAG] 排序自检
  // if (filtered_entries.size() > 1) {
  //   if (cuid == 1003) {
  //       fprintf(stderr, "[DIAG_KEY] NewIterator 1003 First Key Size: %zu\n", filtered_entries[0].key.size());
  //   }
  //   for (size_t i = 0; i < filtered_entries.size() - 1; ++i) {
  //     if (icmp->Compare(filtered_entries[i].key, filtered_entries[i+1].key) > 0) {
  //       fprintf(stderr, "[DIAG_SORT_ERROR] CUID %lu: Keys are NOT monotonic at index %zu!\n", cuid, i);
  //     }
  //   }
  // }

  // 需要进行重复检查，防止 dbiter 报错?
  // 性能考虑还是不要了？
  // auto last = std::unique(filtered_entries.begin(), filtered_entries.end(),
  //                         [icmp](const HotEntry& a, const HotEntry& b) {
  //                           return icmp->Compare(Slice(a.key), Slice(b.key)) == 0;
  //                         });
  // filtered_entries.erase(last, filtered_entries.end());

  return new HotDataBufferIterator(std::move(filtered_entries), icmp);
}


// SST Lifecycle Manager Implementation
void HotSstLifecycleManager::RegisterFile(uint64_t file_number, const std::string& file_path, const std::string& link_path) {
  if (file_number == static_cast<uint64_t>(-1)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto now = std::chrono::steady_clock::now();
  files_[file_number] = {file_path, link_path, 0, 0, now, now};
  LifecycleLogf("[LC_REGISTER] file=%lu path=%s link=%s ref=0\n",
               (unsigned long)file_number, file_path.c_str(), link_path.c_str());
  MaybeDumpStatus();
}

void HotSstLifecycleManager::Ref(uint64_t file_number) {
  if (file_number == static_cast<uint64_t>(-1)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = files_.find(file_number);
  if (it != files_.end()) {
    it->second.last_activity_at = std::chrono::steady_clock::now();
    int old_ref = it->second.ref_count++;
    if (it->second.ref_count > it->second.max_ref_seen)
      it->second.max_ref_seen = it->second.ref_count;
    LifecycleLogf("[LC_REF] file=%lu ref: %d -> %d\n",
                 (unsigned long)file_number, old_ref, it->second.ref_count);
    // 实时异常：ref 超过阈值
    if (it->second.ref_count == kAnomalyRefThreshold) {
      auto age = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - it->second.registered_at).count();
      LifecycleLogf("[LC_ANOMALY] file=%lu ref_count=%d reached threshold=%d "
                   "(age=%llds, max_ref=%d) -- possible ref leak!\n",
                   (unsigned long)file_number, it->second.ref_count,
                   kAnomalyRefThreshold, (long long)age, it->second.max_ref_seen);
    }
    MaybeDumpStatus();
  } else {
    LifecycleLogf("[LC_WARN] Ref on unregistered file=%lu\n",
                 (unsigned long)file_number);
  }
}

void HotSstLifecycleManager::Unref(uint64_t file_number) {
  if (file_number == static_cast<uint64_t>(-1)) return;
  std::string f_del, l_del;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = files_.find(file_number);
    if (it != files_.end()) {
      it->second.last_activity_at = std::chrono::steady_clock::now();
      int old_ref = it->second.ref_count;
      --it->second.ref_count;
      LifecycleLogf("[LC_UNREF] file=%lu ref: %d -> %d\n",
                   (unsigned long)file_number, old_ref, it->second.ref_count);
      if (it->second.ref_count <= 0) {
        f_del = it->second.file_path; l_del = it->second.link_path;
        auto age = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - it->second.registered_at).count();
        LifecycleLogf("[LC_DELETE] file=%lu DELETED (age=%llds max_ref=%d path=%s)\n",
                     (unsigned long)file_number, (long long)age,
                     it->second.max_ref_seen, f_del.c_str());
        files_.erase(it);
      }
    } else {
      LifecycleLogf("[LC_WARN] Unref on unregistered file=%lu\n",
                   (unsigned long)file_number);
    }
    MaybeDumpStatus();
  }
  if (!l_del.empty()) env_->DeleteFile(l_del);
  if (!f_del.empty()) env_->DeleteFile(f_del);
}

void HotSstLifecycleManager::DumpStatus() {
  std::lock_guard<std::mutex> lock(mutex_);
  MaybeDumpStatus();
}

void HotSstLifecycleManager::MaybeDumpStatus() {
  // 调用方已持有 mutex_
  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
      now - last_dump_time_).count();
  if (elapsed < kDumpIntervalSec) return;
  last_dump_time_ = now;

  LifecycleLogf("[LC_STATUS] ===== Periodic dump: %zu files tracked =====\n",
               files_.size());
  int anomaly_count = 0;
  for (const auto& kv : files_) {
    uint64_t fn = kv.first;
    const FileState& st = kv.second;
    auto age = std::chrono::duration_cast<std::chrono::seconds>(
        now - st.registered_at).count();
    auto idle = std::chrono::duration_cast<std::chrono::seconds>(
        now - st.last_activity_at).count();
    const char* anomaly = "";
    if (st.ref_count >= kAnomalyRefThreshold) {
      anomaly = "  <== [ANOMALY:HIGH_REF]";
      anomaly_count++;
    } else if (idle >= kStaleRefThresholdSec && st.ref_count > 0) {
      anomaly = "  <== [ANOMALY:STALE_WITH_REF]";
      anomaly_count++;
    }
    LifecycleLogf("[LC_STATUS]   file=%lu ref=%d max_ref=%d age=%llds idle=%llds%s\n",
                 (unsigned long)fn, st.ref_count, st.max_ref_seen,
                 (long long)age, (long long)idle, anomaly);
  }
  if (anomaly_count > 0) {
    LifecycleLogf("[LC_STATUS] WARNING: %d anomalous file(s) detected!\n",
                 anomaly_count);
  }
  LifecycleLogf("[LC_STATUS] ==============================================\n");
}

}  // namespace ROCKSDB_NAMESPACE