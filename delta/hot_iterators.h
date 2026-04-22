#pragma once

#include <memory>
#include <vector>

#include "db/table_cache.h"
#include "db/version_edit.h"
#include "db/version_set.h"
#include "delta/hot_index_table.h"
#include "delta/hotspot_manager.h"
#include "options/cf_options.h"
#include "rocksdb/iterator.h"
#include "rocksdb/options.h"
#include "table/internal_iterator.h"

namespace ROCKSDB_NAMESPACE {

class HotDeltaIterator : public InternalIterator {
 public:
  // deltas: 数据段列表 (FileID, FirstKey, LastKey)
  // icmp: 内部 Key 比较器
  HotDeltaIterator(const std::vector<DataSegment>& deltas, uint64_t cuid,
                   TableCache* table_cache, const ReadOptions& read_options,
                   const FileOptions& file_options,
                   const InternalKeyComparator& icmp,
                   const MutableCFOptions& mutable_cf_options,
                   bool allow_unprepared_value);

  ~HotDeltaIterator() override;

  // InternalIterator 标准接口代理
  bool Valid() const override;
  void SeekToFirst() override;
  void SeekToLast() override;
  void Seek(const Slice& target) override;
  void SeekForPrev(const Slice& target) override;
  void Next() override;
  void Prev() override;
  Slice key() const override;
  Slice value() const override;
  Status status() const override;
  bool PrepareValue() override { return merging_iter_->PrepareValue(); }
  uint64_t GetPhysicalId() override;

 private:
  InternalIterator* merging_iter_;
  std::vector<DataSegment> deltas_;
  // [seg0_lower, seg0_upper, seg1_lower, seg1_upper, ...]
  std::vector<std::string> bounds_storage_;
  std::vector<Slice> bounds_slices_;
  // all ReadOptions (BlockBasedTableIterator
  std::vector<ReadOptions> read_options_storage_;
  uint64_t cuid_;
};

class HotSnapshotIterator : public InternalIterator {
 public:
  // 普通构造函数：内部对 segments 调用 RefSegments
  HotSnapshotIterator(const std::vector<DataSegment>& segments, uint64_t cuid,
                      HotspotManager* hotspot_manager, TableCache* table_cache,
                      const ReadOptions& read_options,
                      const FileOptions& file_options,
                      const InternalKeyComparator& icmp,
                      const MutableCFOptions& mutable_cf_options,
                      std::shared_ptr<HotSstLifecycleManager> lifecycle_manager);

  // 预 Ref 构造函数：调用方已通过 GetEntryAndRefSnapshots 完成 Ref，
  // 构造函数跳过内部 RefSegments，避免双重计数
  struct SegmentsPreRefed {};
  HotSnapshotIterator(SegmentsPreRefed,
                      const std::vector<DataSegment>& segments, uint64_t cuid,
                      HotspotManager* hotspot_manager, TableCache* table_cache,
                      const ReadOptions& read_options,
                      const FileOptions& file_options,
                      const InternalKeyComparator& icmp,
                      const MutableCFOptions& mutable_cf_options,
                      std::shared_ptr<HotSstLifecycleManager> lifecycle_manager);

  ~HotSnapshotIterator() override;

  bool Valid() const override;
  void SeekToFirst() override;
  void SeekToLast() override;
  void Seek(const Slice& target) override;
  void SeekForPrev(const Slice& target) override;
  void Next() override;
  void Prev() override;
  Slice key() const override;
  Slice value() const override;
  Status status() const override;
  uint64_t GetPhysicalId() override;

 private:
  enum class SegmentInitStatus {
    kSuccess,
    kError,
    kSnapshotChanged
  };

  // 根据 index 初始化 current_iter_
  // 返回状态标识SegmentInitStatus
  SegmentInitStatus InitIterForSegment(size_t index);

  // 根据当前位置重新同步到最新地图
  // 返回 true 表示成功定位（可能是 newdata/EOF）
  bool ReSyncToLatestSegments(const Slice& prev_key);


  // 切换到下一个 Segment
  void SwitchToNextSegment();
  void SwitchToPrevSegment();

  // 所有现在的snapshot segments
  std::vector<DataSegment> segments_;
  const uint64_t cuid_;
  HotspotManager* hotspot_manager_;
  TableCache* table_cache_;
  ReadOptions read_options_;
  FileOptions file_options_;
  const InternalKeyComparator& icmp_;
  const MutableCFOptions mutable_cf_options_;
  ReadOptions current_read_options_;

  // 当前正在使用的 Iterator (指向某个 SST 或 内存 Buffer)
  std::unique_ptr<InternalIterator> current_iter_;

  // 当前 Iterator 对应的 Segment 索引
  int current_segment_index_;
  std::string current_lower_bound_str_;
  std::string current_upper_bound_str_;
  Slice current_lower_bound_slice_;
  Slice current_upper_bound_slice_;

  // Ref/Unref 生命周期管理
  void RefSegments(const std::vector<DataSegment>& segs);
  void UnrefSegments(const std::vector<DataSegment>& segs);
  void LogSegmentExit(const char* reason);

  Status status_;

  // 持有 lifecycle_manager 保证 Iterator 生命期内文件不被删除
  std::shared_ptr<HotSstLifecycleManager> lifecycle_manager_;

  // Solution W: 防止无限重同步
  int resync_count_ = 0;
  static constexpr int kMaxResyncRetries = 3;

  // [DIAG] 调试计数：记录当前 segment 导出的数据量
  uint64_t current_segment_read_count_ = 0;

  // 复用 buffer 避免每次 Next() 堆分配（Plan-1 性能优化）
  std::string prev_key_buf_;
};

class DeltaSwitchingIterator : public InternalIterator {
 public:
  DeltaSwitchingIterator(Version* version, HotspotManager* hotspot_manager,
                         const ReadOptions& read_options,
                         const FileOptions& file_options,
                         const InternalKeyComparator& icmp,
                         const MutableCFOptions& mutable_cf_options,
                         Arena* arena);

  ~DeltaSwitchingIterator() override;

  bool Valid() const override;
  void SeekToFirst() override;
  void SeekToLast() override;
  void Seek(const Slice& target) override;
  void SeekForPrev(const Slice& target) override;
  void Next() override;
  void Prev() override;
  Slice key() const override;
  Slice value() const override;
  Status status() const override;
  bool PrepareValue() override;
  uint64_t GetPhysicalId() override;

 private:
  // 初始化冷数据迭代器 (Standard RocksDB Path)
  void InitColdIter();

  // 初始化热点数据迭代器 (Hot Optimized Path)
  bool InitHotIter(uint64_t cuid);

  void CheckAndSwitch(const Slice* target);

  Version* version_;
  HotspotManager* hotspot_manager_;
  ReadOptions read_options_;
  FileOptions file_options_;
  const InternalKeyComparator& icmp_;
  MutableCFOptions mutable_cf_options_;
  Arena* arena_;

  InternalIterator* current_iter_;

  // cold_iter_ 复用
  InternalIterator* cold_iter_;
  // hot_iter_ 切换 CUID 时需要重建
  InternalIterator* hot_iter_;

  uint64_t current_hot_cuid_;
  bool is_hot_mode_;

  // Plan-2: 标记当前热点 CUID 是否有活跃 delta，用于退出 dedup 循环
  bool has_active_deltas_ = false;

  // Buffer to store previous user key for deduplication to avoid heap allocations
  std::string prev_user_key_buf_;
};

}  // namespace ROCKSDB_NAMESPACE