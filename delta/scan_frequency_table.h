// delta/scan_frequency_table.h

#pragma once
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <vector>
#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

struct FrequencyEntry {
  int count = 0;     
  bool is_hot = false; 
};

class ScanFrequencyTable {
 public:
  ScanFrequencyTable(int threshold = 200, int window_sec = 600, size_t num_shards = 128);

  // 返回值：当前是否为热点
  // became_hot：如果不为 nullptr，返回是否为首次成为热点
  bool RecordAndCheckHot(uint64_t cuid, bool* became_hot = nullptr);

  void RemoveCUID(uint64_t cuid);

  bool IsHot(uint64_t cuid) const;

 private:
  struct Shard {
    mutable std::mutex mutex;
    std::unordered_map<uint64_t, FrequencyEntry> table;
    std::chrono::steady_clock::time_point window_start_time;
  };

  Shard& GetShard(uint64_t cuid) {
    return shards_[cuid % shards_.size()];
  }

  const Shard& GetShard(uint64_t cuid) const {
    return shards_[cuid % shards_.size()];
  }

  void CheckAndRotateWindow(Shard& shard);

  int threshold_;
  int window_sec_;
  std::vector<Shard> shards_;
};

} // namespace ROCKSDB_NAMESPACE