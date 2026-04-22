// delta/scan_frequency_table.cc

#include "delta/scan_frequency_table.h"

namespace ROCKSDB_NAMESPACE {

ScanFrequencyTable::ScanFrequencyTable(int threshold, int window_sec, size_t num_shards)
    : threshold_(threshold), window_sec_(window_sec), shards_(num_shards) {
  auto now = std::chrono::steady_clock::now();
  for (size_t i = 0; i < shards_.size(); ++i) {
    shards_[i].window_start_time = now;
  }
}

void ScanFrequencyTable::CheckAndRotateWindow(Shard& shard) {
  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
      now - shard.window_start_time).count();

  // 窗口轮转时，重置
  if (elapsed >= window_sec_) {
    shard.table.clear();
    shard.window_start_time = now;
  }
}

bool ScanFrequencyTable::IsHot(uint64_t cuid) const {
  const Shard& shard = GetShard(cuid);
  std::lock_guard<std::mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it != shard.table.end()) {
    return it->second.is_hot;
  }
  return false;
}

bool ScanFrequencyTable::RecordAndCheckHot(uint64_t cuid, bool* became_hot) {
  Shard& shard = GetShard(cuid);
  std::lock_guard<std::mutex> lock(shard.mutex);
  
  // 初始化输出参数
  if (became_hot) {
    *became_hot = false;
  }
  
  // 暂时不使用 window 策略
  // CheckAndRotateWindow(shard);

  FrequencyEntry& entry = shard.table[cuid];

  // 如果该CUid已经被标记为热点CUid，认为不需要再统计频率
  if (entry.is_hot) {
    entry.count = 0; 
    return true;
  }

  // 每发生一次Scan，计数器加1
  entry.count++;
  if (entry.count >= threshold_) {
    entry.is_hot = true;
    // 首次成为热点
    if (became_hot) {
      *became_hot = true;
    }
  }

  return entry.is_hot;
}

void ScanFrequencyTable::RemoveCUID(uint64_t cuid) {
  Shard& shard = GetShard(cuid);
  std::lock_guard<std::mutex> lock(shard.mutex);
  shard.table.erase(cuid);
}
} // namespace ROCKSDB_NAMESPACE