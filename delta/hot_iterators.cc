#include "delta/hot_iterators.h"

#include <time.h>

#include <chrono>
#include <sstream>

#include "rocksdb/env.h"
#include "logging/logging.h"

#include "delta/delta_perf_counters.h"
#include "table/merging_iterator.h"
#include "util/cast_util.h"
#include "util/extract_cuid.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace ROCKSDB_NAMESPACE {

static FileMetaData MakeFileMetaFromSegment(const DataSegment& seg,
                                            bool is_delta = true) {
  FileMetaData meta;
  // file_number, path_id=0, file_size
  meta.fd = FileDescriptor(seg.file_number, 0, seg.file_size);
  // file_size = 0 对于 delta 来说会 Cache Hit，直接返回 FileReader
  if (seg.file_size <= 0 && !is_delta) {
    std::cerr << "[ERROR] Invalid file size " << seg.file_size
              << " for segment with file number " << seg.file_number
              << std::endl;
  }
  // internal key
  meta.smallest.DecodeFrom(seg.first_key);
  meta.largest.DecodeFrom(seg.last_key);
  return meta;
}

// Wraps a BlockBasedTableIterator to fix SeekToFirst() ignoring
// iterate_lower_bound. BlockBasedTableIterator::SeekToFirst() calls
// SeekImpl(nullptr) which starts from the physical file beginning without
// checking the lower bound. For split delta segments (e.g. file=132 covering
// both [0,54271] and [1197056,2518015]), SeekToFirst() on the
// [1197056,2518015] child would incorrectly return data from row 0.
// This wrapper redirects SeekToFirst() to Seek(lower_bound_ikey) instead.
class LowerBoundedSSTIterator : public InternalIterator {
 public:
  LowerBoundedSSTIterator(InternalIterator* inner,
                          const std::string& lower_bound_ikey)
      : inner_(inner), lower_bound_ikey_(lower_bound_ikey) {}

  ~LowerBoundedSSTIterator() override { delete inner_; }

  bool Valid() const override { return inner_->Valid(); }
  void SeekToFirst() override { inner_->Seek(Slice(lower_bound_ikey_)); }
  void SeekToLast() override { inner_->SeekToLast(); }
  void Seek(const Slice& target) override { inner_->Seek(target); }
  void SeekForPrev(const Slice& target) override {
    inner_->SeekForPrev(target);
  }
  void Next() override { inner_->Next(); }
  void Prev() override { inner_->Prev(); }
  Slice key() const override { return inner_->key(); }
  Slice value() const override { return inner_->value(); }
  Status status() const override { return inner_->status(); }
  bool PrepareValue() override { return inner_->PrepareValue(); }
  uint64_t GetPhysicalId() override { return inner_->GetPhysicalId(); }

 private:
  InternalIterator* inner_;
  std::string lower_bound_ikey_;
};

// ======================= HotDeltaIterator ==========================

HotDeltaIterator::HotDeltaIterator(const std::vector<DataSegment>& deltas,
                                   uint64_t cuid, TableCache* table_cache,
                                   const ReadOptions& read_options,
                                   const FileOptions& file_options,
                                   const InternalKeyComparator& icmp,
                                   const MutableCFOptions& mutable_cf_options,
                                   bool allow_unprepared_value)
    : merging_iter_(nullptr),
      deltas_(deltas),
      cuid_(cuid) {  // Copy deltas
  if (deltas_.empty()) {
    merging_iter_ = NewEmptyInternalIterator<Slice>(nullptr);
    return;
  }

  bounds_storage_.reserve(deltas_.size() * 2);
  bounds_slices_.reserve(deltas_.size() * 2);
  read_options_storage_.reserve(deltas_.size());

  std::vector<InternalIterator*> children;
  children.reserve(deltas_.size());

  // slices for bounds
  for (const auto& delta : deltas_) {
    bounds_storage_.push_back(ExtractUserKey(delta.first_key).ToString());
    std::string upper = ExtractUserKey(delta.last_key).ToString();
    upper.push_back('\0');  // Make it an exclusive bound
    bounds_storage_.push_back(upper);
  }
  for (size_t i = 0; i < bounds_storage_.size(); ++i) {
    bounds_slices_.emplace_back(bounds_storage_[i]);
  }

  for (size_t i = 0; i < deltas_.size(); ++i) {
    const auto& delta = deltas_[i];
    // ReadOptions
    read_options_storage_.emplace_back(read_options);
    ReadOptions& ro = read_options_storage_.back();
    ro.iterate_lower_bound = &bounds_slices_[i * 2];
    ro.iterate_upper_bound = &bounds_slices_[i * 2 + 1];

    FileMetaData meta = MakeFileMetaFromSegment(delta, true);

    InternalIterator* iter = table_cache->NewIterator(
        ro, file_options, icmp,  // InternalKeyComparator
        meta,                    // FileMetaData
        nullptr, mutable_cf_options.prefix_extractor, nullptr, nullptr,
        TableReaderCaller::kUserIterator, nullptr, false,
        0,  // L0
        0, nullptr, nullptr, allow_unprepared_value, nullptr);

    if (iter) {
      children.push_back(new LowerBoundedSSTIterator(iter, delta.first_key));
    }
  }

  merging_iter_ = NewMergingIterator(&icmp, children.data(),
                                     static_cast<int>(children.size()));
}

HotDeltaIterator::~HotDeltaIterator() {
  if (merging_iter_) {
    delete merging_iter_;
  }
}

// agent
bool HotDeltaIterator::Valid() const { return merging_iter_->Valid(); }
void HotDeltaIterator::SeekToFirst() { merging_iter_->SeekToFirst(); }
void HotDeltaIterator::SeekToLast() { merging_iter_->SeekToLast(); }
void HotDeltaIterator::Seek(const Slice& target) {
  merging_iter_->Seek(target);
}
void HotDeltaIterator::SeekForPrev(const Slice& target) {
  merging_iter_->SeekForPrev(target);
}
void HotDeltaIterator::Next() { merging_iter_->Next(); }
void HotDeltaIterator::Prev() { merging_iter_->Prev(); }
Slice HotDeltaIterator::key() const { return merging_iter_->key(); }
Slice HotDeltaIterator::value() const { return merging_iter_->value(); }
Status HotDeltaIterator::status() const { return merging_iter_->status(); }
uint64_t HotDeltaIterator::GetPhysicalId() {
  return merging_iter_->GetPhysicalId();
}

namespace {

class BoundedMemIterator : public InternalIterator {
 public:
  BoundedMemIterator(InternalIterator* iter, const std::string& lower_bound,
                     const std::string& upper_bound,
                     const InternalKeyComparator* icmp)
      : iter_(iter),
        lower_bound_(lower_bound),
        upper_bound_(upper_bound),
        icmp_(icmp) {
    UpdateValid();
  }

  ~BoundedMemIterator() override { delete iter_; }

  bool Valid() const override { return valid_; }

  void SeekToFirst() override {
    if (!lower_bound_.empty()) {
      iter_->Seek(lower_bound_);
    } else {
      iter_->SeekToFirst();
    }
    UpdateValid();
  }

  void SeekToLast() override {
    if (!upper_bound_.empty()) {
      iter_->SeekForPrev(upper_bound_);
    } else {
      iter_->SeekToLast();
    }
    UpdateValid();
  }

  void Seek(const Slice& target) override {
    if (!lower_bound_.empty() && icmp_->Compare(target, lower_bound_) < 0) {
      iter_->Seek(lower_bound_);
    } else if (!upper_bound_.empty() &&
               icmp_->Compare(target, upper_bound_) > 0) {
      valid_ = false;
      return;
    } else {
      iter_->Seek(target);
    }
    UpdateValid();
  }

  void SeekForPrev(const Slice& target) override {
    if (!upper_bound_.empty() && icmp_->Compare(target, upper_bound_) > 0) {
      iter_->SeekForPrev(upper_bound_);
    } else if (!lower_bound_.empty() &&
               icmp_->Compare(target, lower_bound_) < 0) {
      valid_ = false;
      return;
    } else {
      iter_->SeekForPrev(target);
    }
    UpdateValid();
  }

  void Next() override {
    iter_->Next();
    UpdateValid();
  }

  void Prev() override {
    iter_->Prev();
    UpdateValid();
  }

  Slice key() const override { return iter_->key(); }
  Slice value() const override { return iter_->value(); }
  Status status() const override { return iter_->status(); }

  bool PrepareValue() override { return iter_->PrepareValue(); }

  uint64_t GetPhysicalId() override { return iter_->GetPhysicalId(); }

 private:
  void UpdateValid() {
    valid_ = iter_->Valid() && IsWithinBounds(iter_->key());
  }

  bool IsWithinBounds(const Slice& k) const {
    if (!lower_bound_.empty() && icmp_->Compare(k, lower_bound_) < 0)
      return false;
    if (!upper_bound_.empty() && icmp_->Compare(k, upper_bound_) > 0)
      return false;
    return true;
  }

  InternalIterator* iter_;
  std::string lower_bound_;
  std::string upper_bound_;
  const InternalKeyComparator* icmp_;
  bool valid_ = false;
};

}  // namespace

// ====================== HotSnapshotIterator ============================

HotSnapshotIterator::HotSnapshotIterator(
    const std::vector<DataSegment>& segments, uint64_t cuid,
    HotspotManager* hotspot_manager, TableCache* table_cache,
    const ReadOptions& read_options, const FileOptions& file_options,
    const InternalKeyComparator& icmp,
    const MutableCFOptions& mutable_cf_options,
    std::shared_ptr<HotSstLifecycleManager> lifecycle_manager)
    : segments_(segments),
      cuid_(cuid),
      hotspot_manager_(hotspot_manager),
      table_cache_(table_cache),
      read_options_(read_options),
      file_options_(file_options),
      icmp_(icmp),
      mutable_cf_options_(mutable_cf_options),
      current_segment_index_(-1),
      status_(Status::OK()),
      lifecycle_manager_(lifecycle_manager),
      current_segment_read_count_(0) {
  // Ref all physical segments to prevent deletion during Iterator lifetime
  RefSegments(segments_);
}

// SegmentsPreRefed 构造函数：调用方已通过 GetEntryAndRefSnapshots
// 在 shared_lock 内完成 Ref，此处跳过 RefSegments 避免双重计数
HotSnapshotIterator::HotSnapshotIterator(
    SegmentsPreRefed,
    const std::vector<DataSegment>& segments, uint64_t cuid,
    HotspotManager* hotspot_manager, TableCache* table_cache,
    const ReadOptions& read_options, const FileOptions& file_options,
    const InternalKeyComparator& icmp,
    const MutableCFOptions& mutable_cf_options,
    std::shared_ptr<HotSstLifecycleManager> lifecycle_manager)
    : segments_(segments),
      cuid_(cuid),
      hotspot_manager_(hotspot_manager),
      table_cache_(table_cache),
      read_options_(read_options),
      file_options_(file_options),
      icmp_(icmp),
      mutable_cf_options_(mutable_cf_options),
      current_segment_index_(-1),
      status_(Status::OK()),
      lifecycle_manager_(lifecycle_manager),
      current_segment_read_count_(0) {
  // Segments are already Ref'd by the caller; do NOT call RefSegments here
}

HotSnapshotIterator::~HotSnapshotIterator() {
  // Unref all physical segments held by this Iterator
  UnrefSegments(segments_);
  // current_iter_ unique_ptr auto released
}

void HotSnapshotIterator::RefSegments(const std::vector<DataSegment>& segs) {
  if (!lifecycle_manager_) return;
  for (const auto& seg : segs) {
    if (seg.file_number != static_cast<uint64_t>(-1)) {
      lifecycle_manager_->Ref(seg.file_number);
    }
  }
}

void HotSnapshotIterator::UnrefSegments(const std::vector<DataSegment>& segs) {
  if (!lifecycle_manager_) return;
  for (const auto& seg : segs) {
    if (seg.file_number != static_cast<uint64_t>(-1)) {
      lifecycle_manager_->Unref(seg.file_number);
    }
  }
}

void HotSnapshotIterator::LogSegmentExit(const char* /*reason*/) {
  current_segment_read_count_ = 0;
}

HotSnapshotIterator::SegmentInitStatus HotSnapshotIterator::InitIterForSegment(
    size_t index) {
  if (index >= segments_.size()) {
    current_iter_.reset(nullptr);
    current_segment_index_ = -1;
    return SegmentInitStatus::kError;
  }

  const auto& seg = segments_[index];
  current_segment_index_ = static_cast<int>(index);

  if (seg.file_number == static_cast<uint64_t>(-1)) {
    // Case A: 内存 Buffer fileid -1
    InternalIterator* mem_iter =
        hotspot_manager_->NewBufferIterator(cuid_, &icmp_);

    // 检查 buffer 是否还有数据。如果为空，由于 SAC 机制，很可能是发生了 Flush
    // 导致 segments changed 
    mem_iter->Seek(seg.first_key);

    if (!mem_iter->Valid() ||
        icmp_.Compare(mem_iter->key(), seg.first_key) > 0) {
      HotIndexEntry latest_entry;
      if (hotspot_manager_->GetHotIndexEntry(cuid_, &latest_entry) &&
          latest_entry.HasSnapshot()) {
        bool snapshot_changed = false;
        if (segments_.size() != latest_entry.snapshot_segments.size()) {
          snapshot_changed = true;
        } else {
          for (size_t i = 0; i < segments_.size(); ++i) {
            if (segments_[i].file_number !=
                latest_entry.snapshot_segments[i].file_number) {
              snapshot_changed = true;
              break;
            }
          }
        }

        if (snapshot_changed) {
          delete mem_iter;
          // 不在这里更新 segments_，交给 Next() 循环处理 ReSync
          return SegmentInitStatus::kSnapshotChanged;
        }
      }
    }
    current_iter_.reset(new BoundedMemIterator(mem_iter, seg.first_key,
                                               seg.last_key, &icmp_));
    return SegmentInitStatus::kSuccess;
  } else {
    // Case B: 物理 SST
    FileMetaData meta = MakeFileMetaFromSegment(seg, false);

    current_read_options_ = read_options_;
    current_lower_bound_str_ = ExtractUserKey(seg.first_key).ToString();
    current_upper_bound_str_ = ExtractUserKey(seg.last_key).ToString();

    current_upper_bound_str_.push_back('\0');  // Make it an exclusive bound
    current_lower_bound_slice_ = Slice(current_lower_bound_str_);
    current_upper_bound_slice_ = Slice(current_upper_bound_str_);
    current_read_options_.iterate_lower_bound = &current_lower_bound_slice_;
    current_read_options_.iterate_upper_bound = &current_upper_bound_slice_;

    InternalIterator* iter = table_cache_->NewIterator(
        current_read_options_, file_options_, icmp_, meta, nullptr,
        mutable_cf_options_.prefix_extractor, nullptr, nullptr,
        TableReaderCaller::kUserIterator, nullptr, false,
        1,  // L1+
        0, nullptr, nullptr,
        false,  // allow_unprepared_value
        nullptr);

    current_iter_.reset(iter);

    // [DIAG] 防御性检查：如果文件打开失败，触发 ReSync
    if (iter && !iter->status().ok()) {
      fprintf(stderr,
              "[ERROR] CUID %lu: Failed to open SST file %lu in Segment %zu: %s\n",
              cuid_, seg.file_number, index,
              iter->status().ToString().c_str());
      current_iter_.reset(nullptr);
      current_segment_index_ = -1;
      return SegmentInitStatus::kSnapshotChanged;
    }

    return SegmentInitStatus::kSuccess;
  }
}

void HotSnapshotIterator::Seek(const Slice& target) {
  if (segments_.empty()) {
    current_iter_.reset(nullptr);
    return;
  }

  while (true) {
    // EndKey >= Target 的 Segment
    auto it = std::lower_bound(
        segments_.begin(), segments_.end(), target,
        [&](const DataSegment& seg, const Slice& val) {
          if (seg.last_key.empty()) {
            fprintf(stderr,
                    "Warning: Empty last_key in segment. This should not "
                    "happen.\n");
            return false;
          }
          // 比较 seg.last_key < val
          return icmp_.user_comparator()->Compare(ExtractUserKey(seg.last_key),
                                                  ExtractUserKey(val)) < 0;
        });

    size_t index = std::distance(segments_.begin(), it);

    if (it == segments_.end()) {
      current_iter_.reset(nullptr);
      current_segment_index_ = -1;
      return;
    }

    if (static_cast<int>(index) != current_segment_index_) {
      // [DIAG] Log jumping away from current segment
      LogSegmentExit("Interrupted by Seek");

      SegmentInitStatus status = InitIterForSegment(index);
      if (status == SegmentInitStatus::kSnapshotChanged) {
         // snapshots 已变，原子获取并 Ref 新段，释放旧段
        HotIndexEntry latest_entry;
        if (hotspot_manager_->GetHotIndexEntryAndRefSnapshots(cuid_, &latest_entry)) {
          std::vector<DataSegment> old_segs = std::move(segments_);
          segments_ = latest_entry.snapshot_segments;  // 已预 Ref
          UnrefSegments(old_segs);
          current_segment_index_ = -1;
          continue;  // 重新进行 lower_bound
        }
      } else if (status == SegmentInitStatus::kError) {
        current_iter_.reset(nullptr);
        current_segment_index_ = -1;
        return;
      }
    }

    break;
  }

  // segment seek
  if (current_iter_) {
    current_iter_->Seek(target);

    current_segment_read_count_ = 0;
    if (current_iter_->Valid()) {
        current_segment_read_count_ = 1;
    }

    if (!Valid()) {
      SwitchToNextSegment();
      if (current_iter_) {
        current_iter_->Seek(segments_[current_segment_index_].first_key);
      }
    }

  }
}

void HotSnapshotIterator::Next() {
  if (!current_iter_) return;

  // Plan-1: 复用成员 buffer，避免每次 Next() 触发堆分配
  prev_key_buf_.assign(key().data(), key().size());
  current_iter_->Next();

  int resync_count = 0;
  while (!Valid()) {
    // [DIAG] Current segment finished, log stats before moving away
    LogSegmentExit("End of Segment");

    if (++resync_count > 1000) {
        fprintf(stderr, "[ERROR] HotSnapshotIterator stuck in ReSync loop for CUID %lu. segments=%zu, idx=%d\n", 
                cuid_, segments_.size(), current_segment_index_);
        current_iter_.reset(nullptr);
        break;
    }

    int target_index = current_segment_index_ + 1;

    // 检查是否由于越界或者其他原因需要强制 ReSync
    if (target_index >= static_cast<int>(segments_.size())) {
      if (ReSyncToLatestSegments(Slice(prev_key_buf_))) {
         if (Valid()) break;
         // Relocated 到新图后，current_segment_index_ 已经更新
         // 从这个新的 index 继续检查，所以 continue
         continue; 
      }
      break; // EOF
    }

    // 尝试初始化下一段
    SegmentInitStatus status = InitIterForSegment(target_index);
    if (status == SegmentInitStatus::kSnapshotChanged) {
      if (ReSyncToLatestSegments(Slice(prev_key_buf_))) {
        if (Valid()) break;
        continue;
      }
      break;
    }
    else if (status == SegmentInitStatus::kError) {
      current_iter_.reset(nullptr);
      current_segment_index_ = -1;
      break;
    }

    // 正常跨段，来到新的 Seg 的起点
    if (current_iter_) {
      current_iter_->Seek(segments_[current_segment_index_].first_key);
      // Loop ends if Valid()
    }
  }

  // [DIAG] Increment count for the current record (landed or next)
  if (Valid()) {
      current_segment_read_count_++;
  }

#ifndef NDEBUG
  // Regression 检查：捕捉跨段/同步后的逻辑回退（仅 Debug 构建）
  if (Valid() && icmp_.user_comparator()->Compare(ExtractUserKey(Slice(prev_key_buf_)),
                                       ExtractUserKey(key())) > 0) {
    fprintf(stderr, "[FATAL] HotSnapshotIterator Regression detected!\n");
    fprintf(stderr, "Prev: %s\n", FormatKeyDisplay(prev_key_buf_).c_str());
    fprintf(stderr, "Curr: %s\n", FormatKeyDisplay(key()).c_str());
    for (size_t i = 0; i < segments_.size(); ++i) {
      fprintf(stderr, "Segment %zu: File %lu, Range [%s - %s]\n", i,
              segments_[i].file_number,
              FormatKeyDisplay(segments_[i].first_key).c_str(),
              FormatKeyDisplay(segments_[i].last_key).c_str());
    }
    // 强制失效以停止错误数据产出
    current_iter_.reset(nullptr);
  }
#endif
}

// 当 Reader 因后台 Flush/Promote 导致 segment 变化时
// 从最新索引重新获取段列表，并利用 prev_key 定位到正确的位置继续读
bool HotSnapshotIterator::ReSyncToLatestSegments(const Slice& prev_key) {
  HotIndexEntry latest_entry;
  // atomic Get+Ref：在 shared_lock 内完成 Ref，防止复制后
  // AtomicReplaceForPartialMerge 的锁外 Unref 把文件 GC 掉
  if (!hotspot_manager_->GetHotIndexEntryAndRefSnapshots(cuid_, &latest_entry) ||
      !latest_entry.HasSnapshot()) {
    // snapshot_segments 为空，GetEntryAndRefSnapshots 未 Ref 任何段
    return false;
  }

  // segments 没变 => EOF；释放预 Ref 后返回
  const auto& new_segs = latest_entry.snapshot_segments;
  if (!segments_.empty() && segments_.size() == new_segs.size()) {
    bool identical = true;
    for (size_t i = 0; i < segments_.size(); ++i) {
      if (segments_[i].file_number != new_segs[i].file_number ||
          segments_[i].first_key != new_segs[i].first_key ||
          segments_[i].last_key != new_segs[i].last_key) {
        identical = false;
        break;
      }
    }
    if (identical) {
      // 未采用新段，释放刚才预 Ref 的引用
      UnrefSegments(new_segs);
      return false;
    }
  }

  // 采用新段（已预 Ref），释放旧段引用
  std::vector<DataSegment> old_segments = std::move(segments_);
  segments_ = new_segs;
  // 不再调用 RefSegments(segments_)，new_segs 已由 GetHotIndexEntryAndRefSnapshots 预 Ref
  UnrefSegments(old_segments);

  //如果segments_空了 => EOF
  if (segments_.empty()) {
    current_iter_.reset(nullptr);
    current_segment_index_ = -1;
    return false;
  }

  // 寻找包含或紧邻 prev_key 的段
  auto it = std::lower_bound(
      segments_.begin(), segments_.end(), prev_key,
      [&](const DataSegment& seg, const Slice& val) {
        return icmp_.user_comparator()->Compare(ExtractUserKey(seg.last_key),
                                                ExtractUserKey(val)) < 0;
      });

  if (it == segments_.end()) {
    current_iter_.reset(nullptr);
    current_segment_index_ = static_cast<int>(segments_.size());
    return false;  // 定位到了 all segments 之后 -> EOF
  }

  size_t new_index = std::distance(segments_.begin(), it);
  current_segment_index_ = -1;
  SegmentInitStatus status = InitIterForSegment(new_index);

  if (status != SegmentInitStatus::kSuccess) {
    // 如果又是 snapshot changed，由外部 Next() 循环再次重试
    return true;
  }

  if (!current_iter_) return true;

  // 定位到 prev_key 及其之后
  current_iter_->Seek(prev_key);

  /*
  if (cuid_ == 1003) {
    fprintf(stderr,
            "[DIAG_RESYNC] CUID %lu: Relocated to Seg %d. Valid: %d, "
            "Landed: %s\n",
            cuid_, current_segment_index_,
            current_iter_->Valid(),
            current_iter_->Valid()
                ? FormatKeyDisplay(current_iter_->key()).c_str()
                : "N/A");
  }
  */

  return true; // 地图已经更新，无论当前段是否 Valid 都要返回 true 让外层继续
}

void HotSnapshotIterator::SwitchToNextSegment() {
  int target_index = current_segment_index_ + 1;
  int retry = 0;
  while (true) {
    if (++retry > 20) break;
    SegmentInitStatus status = InitIterForSegment(target_index);
    if (status == SegmentInitStatus::kSnapshotChanged) {
      HotIndexEntry latest_entry;
      if (hotspot_manager_->GetHotIndexEntryAndRefSnapshots(cuid_, &latest_entry)) {
        std::vector<DataSegment> old_segs = std::move(segments_);
        segments_ = latest_entry.snapshot_segments;  // 已预 Ref
        UnrefSegments(old_segs);
        continue;
      }
    }
    break;
  }
}

bool HotSnapshotIterator::Valid() const {
  // if (!current_iter_ || !current_iter_->Valid()) {
  //   return false;
  // }

  // // 边界检查：需要在 Segment 范围内
  // const auto& seg = segments_[current_segment_index_];
  // bool ret = true;
  // if (!seg.last_key.empty()) {
  //   if (icmp_.user_comparator()->Compare(ExtractUserKey(current_iter_->key()),
  //                                        ExtractUserKey(seg.last_key)) > 0) {
  //     ret = false;
  //   }
  // }
  // return ret;
  // 内部迭代器已通过 iterate_upper_bound（SST）或 BoundedMemIterator 的
  // IsWithinBounds() 强制执行段边界，此处无需重复比较（Plan-1 优化）。
  return current_iter_ != nullptr && current_iter_->Valid();
}

Slice HotSnapshotIterator::key() const { return current_iter_->key(); }
Slice HotSnapshotIterator::value() const { return current_iter_->value(); }
Status HotSnapshotIterator::status() const {
  if (!status_.ok()) return status_;
  if (current_iter_) return current_iter_->status();
  return Status::OK();
}

void HotSnapshotIterator::SeekToFirst() {
  // [DIAG] Reset before moving to start
  LogSegmentExit("SeekToFirst");

  int retry = 0;
  while (true) {
    if (++retry > 20) break;
    SegmentInitStatus status = InitIterForSegment(0);

    if (status == SegmentInitStatus::kSnapshotChanged) {
        HotIndexEntry latest_entry;
        if (hotspot_manager_->GetHotIndexEntryAndRefSnapshots(cuid_, &latest_entry)) {
            std::vector<DataSegment> old_segs = std::move(segments_);
            segments_ = latest_entry.snapshot_segments;  // 已预 Ref
            UnrefSegments(old_segs);
            continue;
        }
    }
    break;
  }

  if (current_iter_) {
    current_iter_->Seek(segments_[current_segment_index_].first_key);

    if (current_iter_->Valid()) {
        current_segment_read_count_ = 1;
    }
  }
}

void HotSnapshotIterator::SeekToLast() {
  // [DIAG] Reset before moving to end
  LogSegmentExit("SeekToLast");

  int retry = 0;
  while (true) {
    if (++retry > 20) break;
    SegmentInitStatus status = InitIterForSegment(segments_.size() - 1);

    if (status == SegmentInitStatus::kSnapshotChanged) {
        HotIndexEntry latest_entry;
        if (hotspot_manager_->GetHotIndexEntryAndRefSnapshots(cuid_, &latest_entry)) {
            std::vector<DataSegment> old_segs = std::move(segments_);
            segments_ = latest_entry.snapshot_segments;  // 已预 Ref
            UnrefSegments(old_segs);
            continue;
        }
    }
    break;
  }
  if (current_iter_) {
    if (segments_[current_segment_index_].file_number ==
        static_cast<uint64_t>(-1)) {
      current_iter_->SeekForPrev(segments_[current_segment_index_].last_key);
    } else {
      current_iter_->SeekToLast();
    }
  }
  // [DIAG] Initialize count for the new position
  if (Valid()) {
      current_segment_read_count_ = 1;
  }
}

void HotSnapshotIterator::Prev() {
  if (!current_iter_) return;
  current_iter_->Prev();
  if (!Valid()) {
    LogSegmentExit("End of Segment (Backward)");
    SwitchToPrevSegment();
    if (Valid()) {
      if (segments_[current_segment_index_].file_number ==
          static_cast<uint64_t>(-1)) {
        current_iter_->SeekForPrev(segments_[current_segment_index_].last_key);
      } else {
        current_iter_->SeekToLast();
      }
      // [DIAG] Initialize count for new segment
      if (Valid()) {
          current_segment_read_count_ = 1;
      }
    }
  } else {
    // [DIAG] Increment count for same segment
    current_segment_read_count_++;
  }
}

void HotSnapshotIterator::SwitchToPrevSegment() {
  // [DIAG] Reset before moving
  LogSegmentExit("SwitchToPrevSegment");

  if (current_segment_index_ > 0) {
    int target_index = current_segment_index_ - 1;
    int retry = 0;
    while (true) {
      if (++retry > 20) break;
      SegmentInitStatus status = InitIterForSegment(target_index);

      if (status == SegmentInitStatus::kSnapshotChanged) {
        HotIndexEntry latest_entry;
        if (hotspot_manager_->GetHotIndexEntryAndRefSnapshots(cuid_, &latest_entry)) {
          std::vector<DataSegment> old_segs = std::move(segments_);
          segments_ = latest_entry.snapshot_segments;  // 已预 Ref
          UnrefSegments(old_segs);
          continue;
        }
      }
      break;
    }
  } else {
    current_iter_.reset(nullptr);
    current_segment_index_ = -1;
  }
}

void HotSnapshotIterator::SeekForPrev(const Slice& target) {
  // simplified
  Seek(target);
  if (!Valid()) {
    SeekToLast();
  }
  while (Valid() && icmp_.Compare(key(), target) > 0) {
    Prev();
  }
}

uint64_t HotSnapshotIterator::GetPhysicalId() {
  if (current_iter_) return current_iter_->GetPhysicalId();
  return 0;
}

// ===================================================================
// DeltaSwitchingIterator Implementation
// ===================================================================

DeltaSwitchingIterator::DeltaSwitchingIterator(
    Version* version, HotspotManager* hotspot_manager,
    const ReadOptions& read_options, const FileOptions& file_options,
    const InternalKeyComparator& icmp,
    const MutableCFOptions& mutable_cf_options, Arena* arena)
    : version_(version),
      hotspot_manager_(hotspot_manager),
      read_options_(read_options),
      file_options_(file_options),
      icmp_(icmp),
      mutable_cf_options_(mutable_cf_options),
      arena_(arena),
      current_iter_(nullptr),
      cold_iter_(nullptr),
      hot_iter_(nullptr),
      current_hot_cuid_(0),
      is_hot_mode_(false) {}

DeltaSwitchingIterator::~DeltaSwitchingIterator() {
  if (hot_iter_) {
    delete hot_iter_;
  }
  // rocksdb 会在 arena 中分配内存，不需要手动删除
  if (cold_iter_) {
    if (arena_) {
      cold_iter_->~InternalIterator();
    } else {
      delete cold_iter_;
    }
  }
}

void DeltaSwitchingIterator::InitColdIter() {
  if (cold_iter_) return;

  // L0~Ln 所有文件的 MergingIterator
  // Arena=nullptr, skip_filters=false
  MergeIteratorBuilder builder(&icmp_, arena_);
  version_->AddIterators(read_options_, file_options_, &builder,
                         /*allow_unprepared*/ false);
  // get MergingIterator
  cold_iter_ = builder.Finish();

  if (!cold_iter_) {
    cold_iter_ = NewEmptyInternalIterator<Slice>(arena_);
  }
}

bool DeltaSwitchingIterator::InitHotIter(uint64_t cuid) {
  if (hot_iter_ && current_hot_cuid_ == cuid) return true;

  if (hot_iter_) {
    delete hot_iter_;
    hot_iter_ = nullptr;
  }

  // 1. 获取元数据，并在 shared_lock 持有期间原子地 Ref 所有物理 snapshot 段
  // 防止 GetEntry → Ref 之间 AtomicReplaceForPartialMerge Unref 到 0 的竞态
  HotIndexEntry entry;
  if (!hotspot_manager_->GetHotIndexEntryAndRefSnapshots(cuid, &entry)) {
    if (read_options_.enable_delta_diag_logging) {
      fprintf(stderr, "[DIAG_HOT_PATH] CUID %lu: InitHotIter failed - No Index Entry found\n", cuid);
    }
    return false;
  }

  if (!entry.HasSnapshot()) {
    // Snapshot 为空，预 Ref 也为空（nothing to Unref）
    if (read_options_.enable_delta_diag_logging) {
      fprintf(stderr, "[DIAG_HOT_PATH] CUID %lu: InitHotIter failed - No Snapshot segments yet (Initial scan might be in progress)\n", cuid);
    }
    return false;
  }

  // snapshot_segments 已预 Ref
  InternalIterator* snapshot_iter =
      new HotSnapshotIterator(HotSnapshotIterator::SegmentsPreRefed{},
                              entry.snapshot_segments, cuid, hotspot_manager_,
                              version_->cfd()->table_cache(), read_options_,
                              file_options_, icmp_, mutable_cf_options_,
                              hotspot_manager_->GetLifecycleManager());

  // Plan-3: 当 deltas 为空时直接使用 snapshot_iter，跳过 MergingIterator 的调度开销
  has_active_deltas_ = !entry.deltas.empty();
  if (has_active_deltas_) {
    InternalIterator* delta_iter = new HotDeltaIterator(
        entry.deltas, cuid, version_->cfd()->table_cache(), read_options_,
        file_options_, icmp_, mutable_cf_options_, false);
    InternalIterator* children[] = {delta_iter, snapshot_iter};
    hot_iter_ = NewMergingIterator(&icmp_, children, 2);
  } else {
    // 无 delta 段：直接用 snapshot 迭代器，避免 MergingIterator 优先队列开销
    hot_iter_ = snapshot_iter;
  }
  current_hot_cuid_ = cuid;
  return true;
}

void DeltaSwitchingIterator::Seek(const Slice& target) {
  uint64_t cuid = hotspot_manager_->ExtractCUID(target);
  if (read_options_.skip_hot_path) {
    InitColdIter();
    current_iter_ = cold_iter_;
    is_hot_mode_ = false;
  }
  // 否则，不论是点查还是全扫描(delta_full_scan)，只要是热点，全部走热点路径
  else if (cuid != 0 && hotspot_manager_->IsHot(cuid)) {
    g_hot_path_seek_attempts.fetch_add(1, std::memory_order_relaxed);
    if (InitHotIter(cuid)) {
      current_iter_ = hot_iter_;
      is_hot_mode_ = true;
      g_hot_path_seek_success.fetch_add(1, std::memory_order_relaxed);
    } else {
      // 热点索引尚未建立（如 Init Scan 还在进行中），回退到 Cold Path
      g_hot_path_fallbacks.fetch_add(1, std::memory_order_relaxed);
      if (read_options_.enable_delta_diag_logging) {
        fprintf(stderr, "[DIAG_HOT_PATH] CUID %lu: IsHot=true but InitHotIter failed. Falling back to Cold Path.\n", cuid);
      }
      InitColdIter();
      current_iter_ = cold_iter_;
      is_hot_mode_ = false;
    }
  }
  // 既不是 skip_hot_path 也不是热点，默认走冷路径
  else {
    if (cuid != 0 && read_options_.enable_delta_diag_logging) {
        fprintf(stderr, "[DIAG_HOT_PATH] CUID %lu: IsHot=false. Using Cold Path.\n", cuid);
    }
    InitColdIter();
    current_iter_ = cold_iter_;
    is_hot_mode_ = false;
  }

  if (current_iter_) {
    current_iter_->Seek(target);
  }
}

// 全表扫描或未知方向，强制回退到 Cold Mode
void DeltaSwitchingIterator::SeekToFirst() {
  if (read_options_.skip_hot_path) {
    InitColdIter();
    current_iter_ = cold_iter_;
    is_hot_mode_ = false;
  }
  // InitColdIter();
  // current_iter_ = cold_iter_;
  // is_hot_mode_ = false;
  if (current_iter_) current_iter_->SeekToFirst();
}

void DeltaSwitchingIterator::SeekToLast() {
  InitColdIter();
  current_iter_ = cold_iter_;
  is_hot_mode_ = false;
  if (current_iter_) current_iter_->SeekToLast();
}

bool DeltaSwitchingIterator::Valid() const {
  return current_iter_ && current_iter_->Valid();
}
// void DeltaSwitchingIterator::Next() {
//   if (current_iter_) current_iter_->Next();
// }

void DeltaSwitchingIterator::Next() {
  if (!current_iter_) return;

  // Plan-2: 只有当前 CUID 有活跃 delta 时，才需要 dedup 循环
  // （delta 为空时 hot_iter_ 直接是 HotSnapshotIterator，不会产生重复 key）
  if (is_hot_mode_ && has_active_deltas_ && current_iter_->Valid()) {
    Slice cur_key = current_iter_->key();
    Slice prev_user_key = ExtractUserKey(cur_key);
    prev_user_key_buf_.assign(prev_user_key.data(), prev_user_key.size());

    current_iter_->Next();

    // 跳过 <= 上一条 UserKey 的所有条目，否则FindNextUserEntryInternalImpl会
    // 出现错误Assertion `!skipping_saved_key ||
    // CompareKeyForSkip(ikey_.user_key, saved_key_.GetUserKey()) > 0' failed.
    while (current_iter_->Valid()) {
      Slice cur_user_key = ExtractUserKey(current_iter_->key());
      int cmp = icmp_.user_comparator()->Compare(cur_user_key,
                                                 Slice(prev_user_key_buf_));

      if (cmp > 0) {
        break;
      }
      current_iter_->Next();
    }
  } else {
    current_iter_->Next();
  }
}

void DeltaSwitchingIterator::Prev() {
  if (current_iter_) current_iter_->Prev();
}
void DeltaSwitchingIterator::SeekForPrev(const Slice& target) {
  // 逻辑同 Seek
  Seek(target);
  if (!Valid()) SeekToLast();
  while (Valid() && icmp_.Compare(key(), target) > 0) Prev();
}

Slice DeltaSwitchingIterator::key() const { return current_iter_->key(); }
Slice DeltaSwitchingIterator::value() const { return current_iter_->value(); }
Status DeltaSwitchingIterator::status() const {
  if (current_iter_) return current_iter_->status();
  return Status::OK();
}
bool DeltaSwitchingIterator::PrepareValue() {
  if (current_iter_) return current_iter_->PrepareValue();
  return false;
}
uint64_t DeltaSwitchingIterator::GetPhysicalId() {
  if (current_iter_) return current_iter_->GetPhysicalId();
  return 0;
}

}  // namespace ROCKSDB_NAMESPACE
