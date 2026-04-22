#include "delta/hot_index_table.h"

#include "delta/diag_log.h"
#include "util/extract_cuid.h"

namespace ROCKSDB_NAMESPACE {

HotIndexTable::HotIndexTable(
    const InternalKeyComparator* icmp,
    std::shared_ptr<HotSstLifecycleManager> lifecycle_manager,
    std::shared_ptr<Logger> info_log, size_t num_shards)
    : icmp_(icmp),
      shards_(num_shards),
      lifecycle_manager_(lifecycle_manager),
      info_log_(info_log) {}

// Helper to extract row ID from key for gap detection
static uint64_t ExtractRowID(const std::string& key) {
  if (key.size() < 34) return 0;
  try {
    return std::stoull(key.substr(24, 10));
  } catch (...) {
    return 0;
  }
}

static bool DetectSnapshotGap(uint64_t cuid, const std::vector<DataSegment>& segments, const char* context) {
  if (segments.size() < 2) return false;
  for (size_t i = 0; i + 1 < segments.size(); ++i) {
    uint64_t last_id = ExtractRowID(segments[i].last_key);
    uint64_t next_id = ExtractRowID(segments[i+1].first_key);
    if (last_id != 0 && next_id != 0 && last_id + 1 != next_id && last_id < next_id) {
      DiagLogf("[DIAG_GAP_DETECTED] CUID %lu [%s] GAP between Seg%zu and Seg%zu! "
              "Last (File %lu): %lu, Next (File %lu): %lu. Missing: %lu\n",
              cuid, context, i, i+1, segments[i].file_number, last_id,
              segments[i+1].file_number, next_id, last_id + 1);
      for (size_t j = 0; j < segments.size(); ++j) {
        DiagLogf("  Seg%zu: File %lu, Range [%s, %s]\n", j, segments[j].file_number,
                FormatKeyDisplay(segments[j].first_key).c_str(),
                FormatKeyDisplay(segments[j].last_key).c_str());
      }
      return true;
    }
  }
  return false;
}

static bool HasActualDataInBuffer(uint64_t cuid, const std::string& start,
                                  const std::string& end, HotDataBuffer* buffer,
                                  const InternalKeyComparator* icmp) {
  if (icmp->Compare(start, end) >= 0) return false;

  std::unique_ptr<InternalIterator> iter(buffer->NewIterator(cuid, icmp));
  iter->Seek(start);

  // 跳过边界行本身：start 通常是相邻 segment 的 last_key，已由该 segment 的 SST 覆盖。
  // 若 Seek 落在 start 本身，需前进一步，否则 HasActualDataInBuffer 会因边界行
  // 而对 gap 区间 (start, end) 产生假阳性，导致空洞 remnant -1 segment 被错误保留。
  if (iter->Valid() && icmp->Compare(iter->key(), start) == 0) {
    iter->Next();
  }

  // 如果找到的第一个 Key 已经超过了范围终点，则说明区间内无数据
  return iter->Valid() && icmp->Compare(iter->key(), end) < 0;
}

void HotIndexTable::UpdateSnapshot(
    uint64_t cuid, const std::vector<DataSegment>& new_segments) {
  std::vector<DataSegment> segments_to_unref;
  {
    Shard& shard = GetShard(cuid);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    auto& entry = shard.table[cuid];
    // Old Snapshot
    if (entry.HasSnapshot()) {
      // 检测新 snapshot 是否有 GAP，有则打印完整上下文
      if (DetectSnapshotGap(cuid, new_segments, "UpdateSnapshot")) {
        DiagLogf("[DIAG][UpdateSnapshot] CUID %lu: %zu old segs -> %zu new segs (GAP above)\n",
                cuid, entry.snapshot_segments.size(), new_segments.size());
        for (size_t i = 0; i < entry.snapshot_segments.size(); ++i) {
          DiagLogf("  old[%zu] file=%lu [%s - %s]\n", i,
                  entry.snapshot_segments[i].file_number,
                  FormatKeyDisplay(entry.snapshot_segments[i].first_key).c_str(),
                  FormatKeyDisplay(entry.snapshot_segments[i].last_key).c_str());
        }
        for (size_t i = 0; i < new_segments.size(); ++i) {
          DiagLogf("  new[%zu] file=%lu [%s - %s]\n", i,
                  new_segments[i].file_number,
                  FormatKeyDisplay(new_segments[i].first_key).c_str(),
                  FormatKeyDisplay(new_segments[i].last_key).c_str());
        }
      }
      segments_to_unref.insert(segments_to_unref.end(),
                               entry.snapshot_segments.begin(),
                               entry.snapshot_segments.end());
    }
    entry.snapshot_segments = new_segments;
  }

  if (lifecycle_manager_) {
    for (const auto& seg : new_segments) {
      lifecycle_manager_->Ref(seg.file_number);
    }
    for (const auto& seg : segments_to_unref) {
      lifecycle_manager_->Unref(seg.file_number);
    }
  }
}

void HotIndexTable::AppendSnapshotSegment(uint64_t cuid,
                                          const DataSegment& segment) {
  DiagLogf("[DIAG][AppendSnapshotSegment] CUID %lu: file=%lu [%s - %s]\n",
          cuid, segment.file_number,
          FormatKeyDisplay(segment.first_key).c_str(),
          FormatKeyDisplay(segment.last_key).c_str());
  {
    Shard& shard = GetShard(cuid);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    auto& segments = shard.table[cuid].snapshot_segments;
    // 使用排序插入而非 push_back。
    // 当 FinalizePmPendingSnapshots 有多个 pm_pending SST 时，
    // 第一个 SST 消耗了 -1 段（right remnant 被 discarded），后续 SST
    // 找不到 -1 段而走此路径。若 snapshot 中还有 PM 范围之后的段（Seg-C），
    // 直接 push_back 会使 SST 排在 Seg-C 之后，造成乱序。
    auto pos = std::lower_bound(
        segments.begin(), segments.end(), segment,
        [this](const DataSegment& a, const DataSegment& b) {
          return icmp_->Compare(a.first_key, b.first_key) < 0;
        });
    segments.insert(pos, segment);
  }

  if (lifecycle_manager_) {
    lifecycle_manager_->Ref(segment.file_number);
  }
}

bool HotIndexTable::PromoteSnapshot(uint64_t cuid,
                                    const DataSegment& new_segment,
                                    HotDataBuffer* buffer,
                                    const InternalKeyComparator* icmp,
                                    bool* out_phys_overlap) {
  Shard& shard = GetShard(cuid);
  std::unique_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it == shard.table.end()) {
    return false;
  }

  auto& entry = it->second;

  bool found_mem_overlap = false;   // 是否找到了可替换的 -1 内存段
  bool found_phys_overlap = false;  // 是否有物理段与新 SST 重叠
  int replacement_count = 0;        // 记录替换物理段数量，用于正确 Ref
  std::vector<DataSegment> next_segments;
  std::vector<DataSegment> clipped_mem_segments; // 记录被裁减的 mem 段

  std::string new_start_user = ExtractUserKey(new_segment.first_key).ToString();
  std::string new_end_user = ExtractUserKey(new_segment.last_key).ToString();

  for (const auto& seg : entry.snapshot_segments) {
    std::string seg_start_user = ExtractUserKey(seg.first_key).ToString();
    std::string seg_end_user = ExtractUserKey(seg.last_key).ToString();

    // 检查区间重叠 (UserKey)
    bool is_left =
        (icmp->user_comparator()->Compare(seg_end_user, new_start_user) < 0);
    bool is_right =
        (icmp->user_comparator()->Compare(seg_start_user, new_end_user) > 0);
    bool overlaps = !is_left && !is_right;

    if (overlaps) {
      if (seg.file_number != static_cast<uint64_t>(-1)) {
        // [IMPORTANT]: NEVER cut a physical snapshot segment.
        // Physical segments contain dense historical data that may be needed by
        // older MVCC readers. The new SST from buffer flush may be sparse
        // (cross-scan contamination) and must not destroy dense data.
        next_segments.push_back(seg);
        found_phys_overlap = true;
        continue;
      }

      found_mem_overlap = true;  // found a -1 segment to replace
      clipped_mem_segments.push_back(seg);

      // 只针对 Memory 段 (-1) 进行切分
      // 1. 左侧剩余: [seg.first, new.first)
      if (icmp->Compare(seg.first_key, new_segment.first_key) < 0) {
        bool keep = HasActualDataInBuffer(cuid, seg.first_key,
                                          new_segment.first_key, buffer, icmp);
        if (keep) {
          DataSegment left = seg;
          left.last_key = new_segment.first_key;
          next_segments.push_back(left);
        } else {
          DiagLogf("[DIAG_CLIP_LEFT] CUID %lu: Mem Seg left remnant discarded! Range: [%s - %s). Keep = false\n",
                  cuid, FormatKeyDisplay(seg.first_key).c_str(), FormatKeyDisplay(new_segment.first_key).c_str());
        }
      }
      // 2. 右侧剩余: [new.last, seg.last)
      if (icmp->Compare(seg.last_key, new_segment.last_key) > 0) {
        bool keep = HasActualDataInBuffer(cuid, new_segment.last_key,
                                          seg.last_key, buffer, icmp);
        if (keep) {
          DataSegment right = seg;
          right.first_key = new_segment.last_key;
          next_segments.push_back(right);
          DiagLogf("[DIAG][PromoteSnapshot] CUID %lu: RIGHT remnant KEPT [%s - %s)\n",
                  cuid, FormatKeyDisplay(new_segment.last_key).c_str(),
                  FormatKeyDisplay(seg.last_key).c_str());
        } else {
          DiagLogf("[DIAG_CLIP_RIGHT] CUID %lu: Mem Seg right remnant discarded! Range: [%s - %s). Keep = false\n",
                  cuid, FormatKeyDisplay(new_segment.last_key).c_str(), FormatKeyDisplay(seg.last_key).c_str());
        }
      }

      // 3. 填入重叠部分的物理段，严格 clip 到原 mem segment 的边界
      DataSegment replacement = new_segment;
      if (icmp->Compare(seg.first_key, replacement.first_key) > 0) {
        replacement.first_key = seg.first_key;
      }
      if (icmp->Compare(seg.last_key, replacement.last_key) < 0) {
        replacement.last_key = seg.last_key;
      }
      next_segments.push_back(replacement);
      replacement_count++;

      if (false) {
        fprintf(
            stderr,
            "[DIAG_PROMOTE] CUID %lu: Mem Segment [%s - %s] replaced by SST "
            "%lu clipped to [%s - %s]\n",
            cuid, FormatKeyDisplay(seg.first_key).c_str(),
            FormatKeyDisplay(seg.last_key).c_str(), new_segment.file_number,
            FormatKeyDisplay(replacement.first_key).c_str(),
            FormatKeyDisplay(replacement.last_key).c_str());
      }
    } else {
      next_segments.push_back(seg);
    }
  }

  if (found_mem_overlap) {
    std::sort(next_segments.begin(), next_segments.end(),
              [icmp](const DataSegment& a, const DataSegment& b) {
                int cmp = icmp->Compare(a.first_key, b.first_key);
                if (cmp != 0) return cmp < 0;
                return icmp->Compare(a.last_key, b.last_key) < 0;
              });

    // GAP 检测：排序后检测是否有间隙
    if (DetectSnapshotGap(cuid, next_segments, "PromoteSnapshot")) {
      DiagLogf("[DIAG][PromoteSnapshot] CUID %lu: GAP! newSST file=%lu [%s - %s]. Clipped mem segs=%zu\n",
              cuid, new_segment.file_number,
              FormatKeyDisplay(new_segment.first_key).c_str(),
              FormatKeyDisplay(new_segment.last_key).c_str(),
              clipped_mem_segments.size());
      for (size_t i = 0; i < clipped_mem_segments.size(); ++i) {
        DiagLogf("  clipped_mem[%zu] [%s - %s]\n", i,
                FormatKeyDisplay(clipped_mem_segments[i].first_key).c_str(),
                FormatKeyDisplay(clipped_mem_segments[i].last_key).c_str());
      }
    }

    entry.snapshot_segments = std::move(next_segments);

    if (lifecycle_manager_) {
      // [FIX] 移除 -1 段合并后，可能有多个 -1 段同时被一个 SST 覆盖，
      // 每个 replacement 都持有同一个 file_number，需要 Ref 相应次数。
      for (int i = 0; i < replacement_count; i++) {
        lifecycle_manager_->Ref(new_segment.file_number);
      }
    }
  } else if (found_phys_overlap) {
    // 新 SST 与物理段重叠但未被加入 snapshot：记录以便排查数据丢失
    DiagLogf("[DIAG][PromoteSnapshot] CUID %lu: PHYS_OVERLAP_ONLY newSST file=%lu [%s - %s] skipped (no -1 seg matched)\n",
            cuid, new_segment.file_number,
            FormatKeyDisplay(new_segment.first_key).c_str(),
            FormatKeyDisplay(new_segment.last_key).c_str());
  }

  if (out_phys_overlap) {
    *out_phys_overlap = found_phys_overlap;
  }
  // 只有真正替换了 -1 段才返回 true。
  // 仅物理段重叠时返回 false，由调用方决定走 pm_pending/pending 拦截。
  return found_mem_overlap;
}

void HotIndexTable::AddDelta(uint64_t cuid, const DataSegment& delta) {
  {
    Shard& shard = GetShard(cuid);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    shard.table[cuid].deltas.push_back(delta);
  }

  // if (lifecycle_manager_) {
  //   lifecycle_manager_->Ref(delta.file_number);
  // }
}

bool HotIndexTable::GetEntry(uint64_t cuid, HotIndexEntry* entry) const {
  const Shard& shard = GetShard(cuid);
  std::shared_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it != shard.table.end()) {
    if (entry) {
      *entry = it->second;
    }
    return true;
  }
  return false;
}

bool HotIndexTable::GetEntryAndRefSnapshots(uint64_t cuid,
                                            HotIndexEntry* out_entry) const {
  const Shard& shard = GetShard(cuid);
  std::shared_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it == shard.table.end()) return false;
  if (out_entry) {
    *out_entry = it->second;
    // 在 shared_lock 持有期间 Ref 所有物理段：
    // 此时写路径的 unique_lock 无法同时持有，保证 Ref 发生在任何
    // AtomicReplaceForPartialMerge 的锁外 Unref 之前。
    if (lifecycle_manager_) {
      for (const auto& seg : out_entry->snapshot_segments) {
        if (seg.file_number != static_cast<uint64_t>(-1)) {
          lifecycle_manager_->Ref(seg.file_number);
        }
      }
    }
  }
  return true;
}

void HotIndexTable::MarkDeltasAsObsolete(
    uint64_t cuid, const std::unordered_set<uint64_t>& visited_files) {
  Shard& shard = GetShard(cuid);
  std::unique_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it == shard.table.end()) return;

  auto& entry = it->second;
  if (visited_files.empty()) return;

  // 从 entry.deltas 中匹配并移入 obsolete_deltas
  std::unordered_set<uint64_t> matched_files;
  for (auto d_it = entry.deltas.begin(); d_it != entry.deltas.end();) {
    if (visited_files.count(d_it->file_number)) {
      matched_files.insert(d_it->file_number);
      DiagLogf("[DIAG_MARK_OBSOLETE] CUID %lu: delta file=%lu [%s - %s]"
              " moved to obsolete_deltas\n",
              cuid, d_it->file_number,
              FormatKeyDisplay(d_it->first_key).c_str(),
              FormatKeyDisplay(d_it->last_key).c_str());
      entry.obsolete_deltas.push_back(*d_it);
      d_it = entry.deltas.erase(d_it);
    } else {
      ++d_it;
    }
  }

  // 对于 visited_files 中不在 deltas 中找到的文件
  // 可能是因为第一次 full scan 之前生成的 SST，直接标记为 obsolete
  // TIPS & TODO：是否存在问题？
  for (uint64_t fid : visited_files) {
    if (matched_files.count(fid)) continue;
    DataSegment dummy;
    dummy.file_number = fid;
    DiagLogf("[DIAG_MARK_OBSOLETE] CUID %lu: visited file=%lu NOT in deltas"
            " -> dummy obsolete entry (pre-hotspot SST)\n",
            cuid, fid);
    entry.obsolete_deltas.push_back(dummy);
  }

  size_t dummy_count = visited_files.size() - matched_files.size();
  DiagLogf("[DIAG_MARK_OBSOLETE] CUID %lu: summary: visited=%zu"
          " matched_from_deltas=%zu dummy_added=%zu total_obsolete_now=%zu\n",
          cuid, visited_files.size(), matched_files.size(), dummy_count,
          entry.obsolete_deltas.size());
}

// d)
// 若遇到热点CUid，检查其热点索引表，若发现Deltas列表中对应的该段数据已被标记为
// Obsolete， 则直接跳过该段数据，并删除对应的Deltas记录。 Deprecated：不能在
// compaction 时直接清理
bool HotIndexTable::CheckAndRemoveObsoleteDeltas(
    uint64_t cuid, const std::vector<uint64_t>& input_files) {
  Shard& shard = GetShard(cuid);
  std::unique_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it == shard.table.end()) return false;

  auto& entry = it->second;
  if (entry.obsolete_deltas.empty()) return false;

  bool found_obsolete = false;

  for (auto d_it = entry.obsolete_deltas.begin();
       d_it != entry.obsolete_deltas.end();) {
    bool is_input_file = false;
    for (uint64_t fid : input_files) {
      if (d_it->file_number == fid) {
        is_input_file = true;
        break;
      }
    }
    if (is_input_file) {
      found_obsolete = true;
      d_it = entry.obsolete_deltas.erase(d_it);
    } else {
      ++d_it;
    }
  }
  return found_obsolete;
}

bool HotIndexTable::IsDeltaObsolete(
    uint64_t cuid, const std::vector<uint64_t>& input_files) const {
  const Shard& shard = GetShard(cuid);
  std::shared_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it == shard.table.end()) return false;

  const auto& entry = it->second;
  if (entry.obsolete_deltas.empty()) return false;

  for (uint64_t fid : input_files) {
    bool found_in_obsolete = false;
    for (const auto& obs : entry.obsolete_deltas) {
      if (obs.file_number == fid) {
        found_in_obsolete = true;
        break;
      }
    }
    if (!found_in_obsolete) continue;

    // 保守检查：如果同一 file_number 在 active deltas 中仍有段，
    // 说明该文件包含多个段且只有部分被 PM 覆盖，不能跳过。
    bool still_active = false;
    for (const auto& d : entry.deltas) {
      if (d.file_number == fid) {
        still_active = true;
        break;
      }
    }
    if (still_active) {
      DiagLogf("[DIAG_OBSOLETE_GUARD] cuid=%lu file=%lu: in obsolete_deltas BUT still has active segment(s) in deltas "
               "(Plan C multi-segment guard). L0 Compaction will NOT skip this file.\n",
               cuid, fid);
      return false;
    }
    return true;
  }
  return false;
}

void HotIndexTable::RemoveObsoleteDeltasForCUIDs(
    const std::unordered_set<uint64_t>& cuids,
    const std::vector<uint64_t>& input_files) {
  std::unordered_set<uint64_t> input_files_set(input_files.begin(),
                                               input_files.end());

  for (uint64_t cuid : cuids) {
    Shard& shard = GetShard(cuid);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    auto entry_it = shard.table.find(cuid);
    if (entry_it == shard.table.end()) continue;

    auto& entry = entry_it->second;

    // 1. 移除 deltas 中的匹配文件
    for (auto it = entry.deltas.begin(); it != entry.deltas.end();) {
      if (input_files_set.count(it->file_number)) {
        it = entry.deltas.erase(it);
      } else {
        ++it;
      }
    }

    // 2. 移除 obsolete_deltas 中的匹配文件
    for (auto it = entry.obsolete_deltas.begin();
         it != entry.obsolete_deltas.end();) {
      if (input_files_set.count(it->file_number)) {
        it = entry.obsolete_deltas.erase(it);
      } else {
        ++it;
      }
    }
  }
}

// L0Compaction更新 Delta 索引
// remove delta and add new data
void HotIndexTable::UpdateDeltaIndex(uint64_t cuid,
                                     const std::vector<uint64_t>& input_files,
                                     const DataSegment& new_delta) {
  Shard& shard = GetShard(cuid);
  std::unique_lock<std::shared_mutex> lock(shard.mutex);
  auto& entry = shard.table[cuid];

  // 清理本次 Compaction 涉及的文件
  for (auto it = entry.deltas.begin(); it != entry.deltas.end();) {
    bool is_input = false;
    for (uint64_t fid : input_files) {
      if (it->file_number == fid) {
        is_input = true;
        break;
      }
    }

    if (is_input) {
      it = entry.deltas.erase(it);
    } else {
      ++it;
    }
  }

  // double check obsolete_deltas
  for (auto it = entry.obsolete_deltas.begin();
       it != entry.obsolete_deltas.end();) {
    bool is_input = false;
    for (uint64_t fid : input_files) {
      if (it->file_number == fid) {
        is_input = true;
        break;
      }
    }
    if (is_input) {
      it = entry.obsolete_deltas.erase(it);
    } else {
      ++it;
    }
  }

  // 加入 new delta index
  entry.deltas.push_back(new_delta);
}

void HotIndexTable::RemoveCUID(uint64_t cuid) {
  std::vector<DataSegment> segments_to_unref;
  {
    Shard& shard = GetShard(cuid);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    auto it = shard.table.find(cuid);
    if (it != shard.table.end()) {
      // Snapshot 片段
      if (it->second.HasSnapshot()) {
        segments_to_unref.insert(segments_to_unref.end(),
                                 it->second.snapshot_segments.begin(),
                                 it->second.snapshot_segments.end());
      }
      shard.table.erase(it);
    }
  }

  // Unref
  if (lifecycle_manager_) {
    for (const auto& seg : segments_to_unref) {
      lifecycle_manager_->Unref(seg.file_number);
    }
  }
}

void HotIndexTable::ReplaceOverlappingSegments(
    uint64_t cuid, const DataSegment& new_segment,
    const std::vector<DataSegment>& obsolete_delta_segments) {
  std::vector<DataSegment> segments_to_unref;

  {
    Shard& shard = GetShard(cuid);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    auto& entry = shard.table[cuid];
    auto& snapshots = entry.snapshot_segments;

    std::vector<DataSegment> next_segments;

    std::string new_start_user =
        ExtractUserKey(new_segment.first_key).ToString();
    std::string new_end_user = ExtractUserKey(new_segment.last_key).ToString();

    // 判定与new snapshot相关的snapshot -> !(End < New.Start || Start > New.End)
    // 先将原先的 snapshots 进行分割+排序
    for (auto it = snapshots.begin(); it != snapshots.end(); ++it) {
      std::string seg_start_user = ExtractUserKey(it->first_key).ToString();
      std::string seg_end_user = ExtractUserKey(it->last_key).ToString();

      bool is_left =
          (icmp_->user_comparator()->Compare(seg_end_user, new_start_user) < 0);
      bool is_right =
          (icmp_->user_comparator()->Compare(seg_start_user, new_end_user) > 0);

      if (!is_left && !is_right) {
        // 重叠的情况: 无论物理段还是内存段，都保留不重叠的边缘部分（防止 scope
        // 缩小导致数据丢失）
        // 1. 左侧剩余部分
        if (icmp_->user_comparator()->Compare(seg_start_user, new_start_user) <
            0) {
          DataSegment left_seg = *it;
          left_seg.last_key = new_segment.first_key;
          next_segments.push_back(left_seg);
          if (it->file_number != static_cast<uint64_t>(-1) &&
              lifecycle_manager_) {
            lifecycle_manager_->Ref(it->file_number);
          }
        }
        // 2. 右侧剩余部分
        if (icmp_->user_comparator()->Compare(seg_end_user, new_end_user) > 0) {
          DataSegment right_seg = *it;
          right_seg.first_key = new_segment.last_key;
          next_segments.push_back(right_seg);
          if (it->file_number != static_cast<uint64_t>(-1) &&
              lifecycle_manager_) {
            lifecycle_manager_->Ref(it->file_number);
          }
        }

        // 原来的段如果包含物理文件，则需要放入 unref 列表
        if (it->file_number != static_cast<uint64_t>(-1)) {
          segments_to_unref.push_back(*it);
        }
      } else {
        // 不重叠，保留原样
        next_segments.push_back(*it);
      }
    }

    // 检测单 Key 分段的来源
    if (icmp_->user_comparator()->Compare(
            ExtractUserKey(new_segment.first_key),
            ExtractUserKey(new_segment.last_key)) == 0) {
      DiagLogf("[DIAG_SINGLE_KEY_SEG] CUID %lu: New single-key segment [%s] "
              "added.\n",
              cuid, FormatKeyDisplay(new_segment.first_key).c_str());
    }

    next_segments.push_back(new_segment);

    // 重新排序 (使用 InternalKeyComparator)
    std::sort(next_segments.begin(), next_segments.end(),
              [this](const DataSegment& a, const DataSegment& b) {
                int cmp = icmp_->Compare(a.first_key, b.first_key);
                if (cmp != 0) return cmp < 0;
                return icmp_->Compare(a.last_key, b.last_key) < 0;
              });
    
    if (DetectSnapshotGap(cuid, next_segments, "ReplaceOverlappingSegments")) {
      DiagLogf("[DIAG][ReplaceOverlapping] CUID %lu: GAP! new={file=%lu [%s - %s]}\n",
              cuid, new_segment.file_number,
              FormatKeyDisplay(new_segment.first_key).c_str(),
              FormatKeyDisplay(new_segment.last_key).c_str());
      for (size_t i = 0; i < next_segments.size(); ++i) {
        DiagLogf("  result[%zu] file=%lu [%s - %s]\n", i,
                next_segments[i].file_number,
                FormatKeyDisplay(next_segments[i].first_key).c_str(),
                FormatKeyDisplay(next_segments[i].last_key).c_str());
      }
      for (size_t j = 0; j < entry.deltas.size(); ++j) {
        DiagLogf("  delta[%zu] file=%lu [%s - %s]\n", j,
                entry.deltas[j].file_number,
                FormatKeyDisplay(entry.deltas[j].first_key).c_str(),
                FormatKeyDisplay(entry.deltas[j].last_key).c_str());
      }
    }
    // DetectSnapshotGap(cuid, snapshots, "ReplaceOverlappingSegments");

    snapshots = std::move(next_segments);

    // 插入这个新的 buffer snapshot，重新寻找插入位置进行合并相邻的内存段 (-1)
    auto insert_pos = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const DataSegment& s) {
          return s.first_key == new_segment.first_key &&
                 s.last_key == new_segment.last_key &&
                 s.file_number == static_cast<uint64_t>(-1);
        });

    // 合并相邻的内存段 (-1)，因为之前切分+排序插入过了
    // 直接检查前后buffer segment，如果存在就合并成一个更大的 segment
    // MARK：多个 buffer segment 的情况？
    if (insert_pos != snapshots.end() &&
        insert_pos->file_number == static_cast<uint64_t>(-1)) {
      // 检查后一个
      auto next = std::next(insert_pos);
      if (next != snapshots.end()) {
        if (next->file_number == static_cast<uint64_t>(-1)) {
          insert_pos->last_key = next->last_key;
          snapshots.erase(next);
        }
      }
      // 检查前一个
      if (insert_pos != snapshots.begin()) {
        auto prev = std::prev(insert_pos);
        if (prev->file_number == static_cast<uint64_t>(-1)) {
          prev->last_key = insert_pos->last_key;
          snapshots.erase(insert_pos);
        }
      }
    }

    if (!obsolete_delta_segments.empty()) {
      auto& deltas = entry.deltas;

      // 段级精确匹配：只移入与 overlapping_deltas 精确匹配的段
      for (const auto& obs_seg : obsolete_delta_segments) {
        for (auto it = deltas.begin(); it != deltas.end();) {
          if (it->file_number == obs_seg.file_number &&
              it->first_key == obs_seg.first_key &&
              it->last_key == obs_seg.last_key) {
            entry.obsolete_deltas.push_back(*it);
            it = deltas.erase(it);
          } else {
            ++it;
          }
        }
      }
    }
  }  // Unlock

  if (lifecycle_manager_) {
    if (new_segment.file_number != static_cast<uint64_t>(-1)) {
      lifecycle_manager_->Ref(new_segment.file_number);
    }
    for (const auto& seg : segments_to_unref) {
      lifecycle_manager_->Unref(seg.file_number);
    }
  }
}

void HotIndexTable::AtomicReplaceForPartialMerge(
    uint64_t cuid,
    const std::string& pm_range_first,
    const std::string& pm_range_last,
    const std::vector<DataSegment>& pm_sst_segs,
    const std::vector<DataSegment>& pm_promoted_segs,
    bool has_buf_data,
    const std::string& buf_min,
    const std::string& buf_max,
    const std::vector<DataSegment>& obsolete_delta_segments) {
  // 在 shard lock 外收集待 Unref 的旧段（避免锁内调用 lifecycle_manager_→死锁风险）
  std::vector<DataSegment> segments_to_unref;

  {
    Shard& shard = GetShard(cuid);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    auto& entry = shard.table[cuid];
    auto& snapshots = entry.snapshot_segments;

    std::string pm_start_user = ExtractUserKey(pm_range_first).ToString();
    std::string pm_end_user   = ExtractUserKey(pm_range_last).ToString();

    // 构建 pm_promoted 文件号集合：这些 SST 已在 PromoteSnapshot 时正确放入 snapshot，
    // 步骤① 不应将其移除（它们本身就是此次 PM 产生的正确数据）。
    std::unordered_set<uint64_t> pm_promoted_file_nums;
    for (const auto& seg : pm_promoted_segs) {
      pm_promoted_file_nums.insert(seg.file_number);
    }
    if (!pm_promoted_segs.empty()) {
      DiagLogf("[DIAG][AtomicReplace] CUID %lu: pm_promoted_segs=%zu files to preserve:",
              cuid, pm_promoted_segs.size());
      for (const auto& seg : pm_promoted_segs) {
        DiagLogf(" %lu[%s-%s]", seg.file_number,
                FormatKeyDisplay(seg.first_key).c_str(),
                FormatKeyDisplay(seg.last_key).c_str());
      }
      DiagLogf("\n");
    }

    std::vector<DataSegment> next_segments;

    // ① 裁剪与 PM 输出范围重叠的旧 snapshot 段（逻辑与 ReplaceOverlappingSegments 相同）
    for (const auto& seg : snapshots) {
      std::string seg_start = ExtractUserKey(seg.first_key).ToString();
      std::string seg_end   = ExtractUserKey(seg.last_key).ToString();

      bool is_left  = (icmp_->user_comparator()->Compare(seg_end,   pm_start_user) < 0);
      bool is_right = (icmp_->user_comparator()->Compare(seg_start, pm_end_user)   > 0);

      if (!is_left && !is_right) {
        // [FIX-v2] pm_promoted SST 被 PromoteSnapshot 以截断范围（old -1 段边界）写入 snapshot，
        // 但其物理完整范围更宽（存于 pm_promoted_segs）。此处跳过截断的旧条目，
        // 在步骤 ②-c 中以完整物理范围重新插入，同时切割 buffer 以消除重叠。
        // 不加入 segments_to_unref：文件保留来自 PromoteSnapshot 的 Ref，在 ②-c 中转移。
        if (seg.file_number != static_cast<uint64_t>(-1) &&
            pm_promoted_file_nums.count(seg.file_number)) {
          DiagLogf("[DIAG][AtomicReplace] CUID %lu: Seg=[%s - %s] is pm_promoted, DEFERRED to step-2c (full range)\n",
                  cuid, FormatKeyDisplay(seg.first_key).c_str(),
                  FormatKeyDisplay(seg.last_key).c_str());
          continue;
        }

        // 与 PM 范围重叠：保留不重叠的边缘部分
        DiagLogf("[DIAG][AtomicReplace] CUID %lu: Seg Overlap detected. Seg=[%s - %s] PM=[%s - %s]\n",
                cuid, FormatKeyDisplay(seg.first_key).c_str(), FormatKeyDisplay(seg.last_key).c_str(),
                FormatKeyDisplay(pm_range_first).c_str(), FormatKeyDisplay(pm_range_last).c_str());
        // 1. 左侧剩余
        if (icmp_->user_comparator()->Compare(seg_start, pm_start_user) < 0) {
          DataSegment left = seg;
          left.last_key = pm_range_first;
          next_segments.push_back(left);
          DiagLogf("  \u2192 Created LEFT remnant: [%s - %s]\n",
                  FormatKeyDisplay(left.first_key).c_str(), FormatKeyDisplay(left.last_key).c_str());
          // 左残余与原始段共享文件，额外 Ref
          if (seg.file_number != static_cast<uint64_t>(-1) && lifecycle_manager_) {
            lifecycle_manager_->Ref(seg.file_number);
          }
        }
        // 2. 右侧剩余
        if (icmp_->user_comparator()->Compare(seg_end, pm_end_user) > 0) {
          DataSegment right = seg;
          right.first_key = pm_range_last;
          next_segments.push_back(right);
          DiagLogf("  \u2192 Created RIGHT remnant: [%s - %s]\n",
                  FormatKeyDisplay(right.first_key).c_str(), FormatKeyDisplay(right.last_key).c_str());
          if (seg.file_number != static_cast<uint64_t>(-1) && lifecycle_manager_) {
            lifecycle_manager_->Ref(seg.file_number);
          }
        }
        // 原始段收集待 Unref（物理段需要减少 snapshot Ref）
        if (seg.file_number != static_cast<uint64_t>(-1)) {
          segments_to_unref.push_back(seg);
        }
      } else {
        // 不重叠：原样保留
        next_segments.push_back(seg);
      }
    }

    // ②-prep: 若 buffer 有剩余数据，先将 buf_seg (-1) 插入 next_segments。
    // 后续步骤②会将 pm_sst_segs 的物理 SST 与此 -1 段进行裁剪，自动消除重叠。
    // [FIX] 之前 buf_seg 在步骤②之后插入，导致它与已插入的 pm_sst_segs 物理段重叠，
    //       后续 PromoteSnapshot 无法消除这种重叠，最终产生 Regression。
    if (has_buf_data) {
      DataSegment buf_seg;
      buf_seg.file_number = static_cast<uint64_t>(-1);
      buf_seg.first_key   = buf_min;
      buf_seg.last_key    = buf_max;
      
      // 裁剪到 pm_range 内
      if (icmp_->user_comparator()->Compare(ExtractUserKey(buf_seg.first_key), pm_start_user) < 0) {
        buf_seg.first_key = pm_range_first;
      }
      if (icmp_->user_comparator()->Compare(ExtractUserKey(buf_seg.last_key), pm_end_user) > 0) {
        buf_seg.last_key = pm_range_last;
      }

      if (icmp_->user_comparator()->Compare(ExtractUserKey(buf_seg.first_key), 
                                            ExtractUserKey(buf_seg.last_key)) <= 0) {
        DiagLogf("[DIAG][AtomicReplace] CUID %lu: Pre-inserting Buffer Segment (File -1) [%s - %s] (will be split by pm_sst_segs)\n",
                cuid, FormatKeyDisplay(buf_seg.first_key).c_str(), FormatKeyDisplay(buf_seg.last_key).c_str());
        next_segments.push_back(buf_seg);
      }
    }

    // ② 处理 pm_sst_segs（pm_pending 期间 flush 产生的 SST）
    // pm_sst_seg 可能同时包含：
    //   (a) pm_range 范围外的旧 -1 段数据（InitScan 或上次 PM 的尾部）
    //   (b) pm_range 范围内的本次 PM 输出数据（中途 flush 的部分）
    // 两种数据需要不同处理方式：
    //   (a) → 像 PromoteSnapshot 一样 clip 到 -1 段边界，替换旧 -1 段
    //   (b) → clip 到 pm_range，填充 buf_seg 左侧空隙
    //
    // [FIX] 之前只有 (b)（pm_range clip），导致 (a) 部分被 Unref 丢弃，
    //       旧 -1 段指向的 buffer 数据已被 PopFrontBlock 移走 → 数据丢失。

    // 收集 pm_sst 的 file_number，用于 ②-b 区分「原始物理段」和「本轮 pm_sst 插入的物理段」
    // 后者可以被后续 pm_sst_seg 裁剪，前者不可修改。
    // 处理场景：多次 flush 产生重叠的 pm_sst_segs（例如 [0,4000] 和 [2400,6000]）
    std::unordered_set<uint64_t> pm_sst_file_numbers;
    for (const auto& sst : pm_sst_segs) {
      pm_sst_file_numbers.insert(sst.file_number);
    }

    for (const auto& sst_seg : pm_sst_segs) {
      std::string sst_start_user = ExtractUserKey(sst_seg.first_key).ToString();
      std::string sst_end_user   = ExtractUserKey(sst_seg.last_key).ToString();

      bool extends_left  = icmp_->user_comparator()->Compare(sst_start_user, pm_start_user) < 0;
      bool extends_right = icmp_->user_comparator()->Compare(sst_end_user,   pm_end_user)   > 0;

      // ②-a: 处理超出 pm_range 的部分（PromoteSnapshot 逻辑）
      // 将超出部分 clip 到每个重叠 -1 段的边界，精确替换旧 -1 段。
      // 物理段不受影响。每个 replacement 都需要额外 Ref。
      auto promote_out_of_range = [&](const DataSegment& portion) {
        std::string por_start = ExtractUserKey(portion.first_key).ToString();
        std::string por_end   = ExtractUserKey(portion.last_key).ToString();

        std::vector<DataSegment> after;
        for (const auto& seg : next_segments) {
          // 原始物理段（非 -1 且非 pm_sst 来源）：不可修改
          bool is_original_phys = (seg.file_number != static_cast<uint64_t>(-1) &&
                                   pm_sst_file_numbers.find(seg.file_number) == pm_sst_file_numbers.end());
          if (is_original_phys) {
            after.push_back(seg);
            continue;
          }
          std::string seg_s = ExtractUserKey(seg.first_key).ToString();
          std::string seg_e = ExtractUserKey(seg.last_key).ToString();
          bool il = icmp_->user_comparator()->Compare(seg_e, por_start) < 0;
          bool ir = icmp_->user_comparator()->Compare(seg_s, por_end)   > 0;
          if (il || ir) {
            after.push_back(seg);
          } else {
            // 重叠：用 SST 替换，保留左/右残余
            if (icmp_->user_comparator()->Compare(seg_s, por_start) < 0) {
              DataSegment left = seg;
              left.last_key = portion.first_key;
              after.push_back(left);
              if (seg.file_number != static_cast<uint64_t>(-1) && lifecycle_manager_) {
                lifecycle_manager_->Ref(seg.file_number);
              }
            }
            if (icmp_->user_comparator()->Compare(seg_e, por_end) > 0) {
              DataSegment right = seg;
              right.first_key = portion.last_key;
              after.push_back(right);
              if (seg.file_number != static_cast<uint64_t>(-1) && lifecycle_manager_) {
                lifecycle_manager_->Ref(seg.file_number);
              }
            }
            // 被替换的 pm_sst 物理段需要 Unref
            if (seg.file_number != static_cast<uint64_t>(-1) && lifecycle_manager_) {
              lifecycle_manager_->Unref(seg.file_number);
            }
            // 创建 replacement：clip SST 到段的边界
            DataSegment rep = portion;
            if (icmp_->user_comparator()->Compare(seg_s, por_start) > 0) {
              rep.first_key = seg.first_key;
            }
            if (icmp_->user_comparator()->Compare(seg_e, por_end) < 0) {
              rep.last_key = seg.last_key;
            }
            after.push_back(rep);
            // 每个 replacement 需要 1 个 Ref（pm_pending 的 Ref 留给 ②-b 使用）
            if (lifecycle_manager_) {
              lifecycle_manager_->Ref(sst_seg.file_number);
            }
            DiagLogf("[DIAG][AtomicReplace] CUID %lu: ②-a Out-of-range SST file=%lu "
                    "replaced seg [%s - %s] → SST clipped [%s - %s]\n",
                    cuid, sst_seg.file_number,
                    FormatKeyDisplay(seg.first_key).c_str(),
                    FormatKeyDisplay(seg.last_key).c_str(),
                    FormatKeyDisplay(rep.first_key).c_str(),
                    FormatKeyDisplay(rep.last_key).c_str());
          }

        }
        next_segments = std::move(after);
      };

      if (extends_left) {
        DataSegment left_portion = sst_seg;
        left_portion.last_key = pm_range_first;
        // 确认范围有效
        if (icmp_->user_comparator()->Compare(
                ExtractUserKey(left_portion.first_key),
                ExtractUserKey(left_portion.last_key)) <= 0) {
          promote_out_of_range(left_portion);
        }
      }
      if (extends_right) {
        DataSegment right_portion = sst_seg;
        right_portion.first_key = pm_range_last;
        if (icmp_->user_comparator()->Compare(
                ExtractUserKey(right_portion.first_key),
                ExtractUserKey(right_portion.last_key)) <= 0) {
          promote_out_of_range(right_portion);
        }
      }

      // ②-b: 处理 pm_range 内的部分（原有逻辑）
      // clip 到 pm_range，替换 pm_range 内的 -1 段（buf_seg 及 step① 残余）
      DataSegment clipped_seg = sst_seg;
      if (icmp_->user_comparator()->Compare(
              ExtractUserKey(clipped_seg.first_key), pm_start_user) < 0) {
        clipped_seg.first_key = pm_range_first;
      }
      if (icmp_->user_comparator()->Compare(
              ExtractUserKey(clipped_seg.last_key), pm_end_user) > 0) {
        clipped_seg.last_key = pm_range_last;
      }

      // Clip 后若范围变为空（SST 完全位于 pm_range 之外），释放 pm_pending Ref 并跳过
      if (icmp_->user_comparator()->Compare(
              ExtractUserKey(clipped_seg.first_key),
              ExtractUserKey(clipped_seg.last_key)) > 0) {
        if (lifecycle_manager_) {
          lifecycle_manager_->Unref(sst_seg.file_number);
        }
        continue;
      }

      DiagLogf("[DIAG][AtomicReplace] CUID %lu: ②-b In-range pm_sst_seg file=%lu [%s - %s]\n",
              cuid, clipped_seg.file_number,
              FormatKeyDisplay(clipped_seg.first_key).c_str(),
              FormatKeyDisplay(clipped_seg.last_key).c_str());

      std::string sst_start = ExtractUserKey(clipped_seg.first_key).ToString();
      std::string sst_end   = ExtractUserKey(clipped_seg.last_key).ToString();

      std::vector<DataSegment> after_promote;
      for (const auto& seg : next_segments) {
        // 原始 snapshot 物理段（非 -1 且非 pm_sst 来源）：不可修改
        bool is_original_phys = (seg.file_number != static_cast<uint64_t>(-1) &&
                                 pm_sst_file_numbers.find(seg.file_number) == pm_sst_file_numbers.end());
        if (is_original_phys) {
          after_promote.push_back(seg);
          continue;
        }
        // -1 段 或 本轮 pm_sst 插入的物理段：可被后续 pm_sst_seg 裁剪
        std::string seg_start = ExtractUserKey(seg.first_key).ToString();
        std::string seg_end   = ExtractUserKey(seg.last_key).ToString();
        bool is_left  = icmp_->user_comparator()->Compare(seg_end,   sst_start) < 0;
        bool is_right = icmp_->user_comparator()->Compare(seg_start, sst_end)   > 0;
        if (is_left || is_right) {
          after_promote.push_back(seg);
        } else {
          // 重叠：裁剪，保留左/右残余
          if (icmp_->user_comparator()->Compare(seg_start, sst_start) < 0) {
            DataSegment left = seg;
            left.last_key = clipped_seg.first_key;
            after_promote.push_back(left);
            // pm_sst 物理段被裁剪：残余保留原 Ref，新 clip 区域需额外 Ref
            if (seg.file_number != static_cast<uint64_t>(-1) && lifecycle_manager_) {
              lifecycle_manager_->Ref(seg.file_number);
            }
          }
          if (icmp_->user_comparator()->Compare(seg_end, sst_end) > 0) {
            DataSegment right = seg;
            right.first_key = clipped_seg.last_key;
            after_promote.push_back(right);
            if (seg.file_number != static_cast<uint64_t>(-1) && lifecycle_manager_) {
              lifecycle_manager_->Ref(seg.file_number);
            }
          }
          // 被替换的 pm_sst 物理段需要 Unref
          if (seg.file_number != static_cast<uint64_t>(-1) && lifecycle_manager_) {
            lifecycle_manager_->Unref(seg.file_number);
          }
        }
      }
      next_segments = std::move(after_promote);
      // pm_pending 的 1 个 Ref 转移为 snapshot Ref（不额外 Ref）
      next_segments.push_back(clipped_seg);
    }

    // ②-c: 以完整物理范围插入 pm_promoted SSTs，并切割 buffer/-1 段以消除重叠。
    // PromoteSnapshot 已设置了 Ref；这里将其「转移」给新条目（不额外 Ref/Unref）。
    for (const auto& sst_seg : pm_promoted_segs) {
      std::string prom_start = ExtractUserKey(sst_seg.first_key).ToString();
      std::string prom_end   = ExtractUserKey(sst_seg.last_key).ToString();

      // 将 pm_promoted 完整范围裁剪到 pm_range（防止超出 PM 范围影响外部段）
      DataSegment clipped_prom = sst_seg;
      if (icmp_->user_comparator()->Compare(prom_start, pm_start_user) < 0) {
        clipped_prom.first_key = pm_range_first;
        prom_start = pm_start_user;
      }
      if (icmp_->user_comparator()->Compare(prom_end, pm_end_user) > 0) {
        clipped_prom.last_key = pm_range_last;
        prom_end = pm_end_user;
      }

      std::vector<DataSegment> after_cut;
      for (const auto& seg : next_segments) {
        // 原始物理段（非 -1、非 pm_sst、非 pm_promoted）：不可被切割
        bool is_orig = (seg.file_number != static_cast<uint64_t>(-1) &&
                        pm_sst_file_numbers.find(seg.file_number) == pm_sst_file_numbers.end() &&
                        pm_promoted_file_nums.find(seg.file_number) == pm_promoted_file_nums.end());
        if (is_orig) {
          after_cut.push_back(seg);
          continue;
        }
        std::string seg_s = ExtractUserKey(seg.first_key).ToString();
        std::string seg_e = ExtractUserKey(seg.last_key).ToString();
        bool il = icmp_->user_comparator()->Compare(seg_e, prom_start) < 0;
        bool ir = icmp_->user_comparator()->Compare(seg_s, prom_end)   > 0;
        if (il || ir) {
          after_cut.push_back(seg);
        } else {
          // 重叠：保留 pm_promoted 范围之外的残余
          if (icmp_->user_comparator()->Compare(seg_s, prom_start) < 0) {
            DataSegment left = seg;
            left.last_key = clipped_prom.first_key;
            after_cut.push_back(left);
            if (seg.file_number != static_cast<uint64_t>(-1) && lifecycle_manager_) {
              lifecycle_manager_->Ref(seg.file_number);
            }
          }
          if (icmp_->user_comparator()->Compare(seg_e, prom_end) > 0) {
            DataSegment right = seg;
            right.first_key = clipped_prom.last_key;
            after_cut.push_back(right);
            if (seg.file_number != static_cast<uint64_t>(-1) && lifecycle_manager_) {
              lifecycle_manager_->Ref(seg.file_number);
            }
          }
          // pm_sst 物理段被切走时需 Unref；-1 段不需要
          if (seg.file_number != static_cast<uint64_t>(-1) && lifecycle_manager_) {
            lifecycle_manager_->Unref(seg.file_number);
          }
        }
      }
      next_segments = std::move(after_cut);
      // 以完整（已裁剪到 pm_range 的）范围插入 pm_promoted SST
      next_segments.push_back(clipped_prom);
      DiagLogf("[DIAG][AtomicReplace] CUID %lu: step-2c pm_promoted file=%lu [%s - %s] "
              "inserted with full range (clipped to pm_range)\n",
              cuid, sst_seg.file_number,
              FormatKeyDisplay(clipped_prom.first_key).c_str(),
              FormatKeyDisplay(clipped_prom.last_key).c_str());
      // Degenerate segment check: first_key should be <= last_key
      if (icmp_->Compare(clipped_prom.first_key, clipped_prom.last_key) > 0) {
        DiagLogf("[DIAG_DEGENERATE_SEG] *** CRITICAL *** CUID %lu: step-2c produced "
                "degenerate segment file=%lu first_key=[%s] > last_key=[%s]. "
                "pm_range=[%s - %s] sst_seg=[%s - %s]. "
                "This segment is invalid and will cause scan errors!\n",
                cuid, sst_seg.file_number,
                FormatKeyDisplay(clipped_prom.first_key).c_str(),
                FormatKeyDisplay(clipped_prom.last_key).c_str(),
                FormatKeyDisplay(pm_range_first).c_str(),
                FormatKeyDisplay(pm_range_last).c_str(),
                FormatKeyDisplay(sst_seg.first_key).c_str(),
                FormatKeyDisplay(sst_seg.last_key).c_str());
      }
    }

    // ④ 排序
    std::sort(next_segments.begin(), next_segments.end(),
              [this](const DataSegment& a, const DataSegment& b) {
                int cmp = icmp_->Compare(a.first_key, b.first_key);
                if (cmp != 0) return cmp < 0;
                return icmp_->Compare(a.last_key, b.last_key) < 0;
              });


    // [FIX] 不再合并相邻 -1 段。步骤 ②-a 已正确处理了旧 -1 段的替换，
    // 不同来源的 -1 段保持独立，可确保每个被其对应的 flush SST 精确替换。


    // GAP 诊断
    if (DetectSnapshotGap(cuid, next_segments, "AtomicReplaceForPartialMerge")) {
      DiagLogf("[DIAG][AtomicReplace] CUID %lu: GAP! pm_range=[%s - %s] "
              "has_buf=%d pm_ssts=%zu\n",
              cuid,
              FormatKeyDisplay(pm_range_first).c_str(),
              FormatKeyDisplay(pm_range_last).c_str(),
              (int)has_buf_data, pm_sst_segs.size());
    }

    snapshots = std::move(next_segments);

    // ⑤ 处理 obsolete_delta_segments（段级精确匹配）
    // 只移入与 overlapping_deltas 精确匹配的段，同 file_number 的其他段保留在 active deltas
    if (!obsolete_delta_segments.empty()) {
      auto& deltas = entry.deltas;
      for (const auto& obs_seg : obsolete_delta_segments) {
        for (auto it = deltas.begin(); it != deltas.end();) {
          if (it->file_number == obs_seg.file_number &&
              it->first_key == obs_seg.first_key &&
              it->last_key == obs_seg.last_key) {
            entry.obsolete_deltas.push_back(*it);
            it = deltas.erase(it);
          } else {
            ++it;
          }
        }
      }
    }
  }  // shard.mutex 

  // ⑥ 锁外释放旧段持有的 snapshot Ref（不影响新 snapshot 的可见性）
  if (lifecycle_manager_) {
    for (const auto& seg : segments_to_unref) {
      lifecycle_manager_->Unref(seg.file_number);
    }
  }
}

size_t HotIndexTable::CountOverlappingDeltas(
    uint64_t cuid, const std::string& first_key,
    const std::string& last_key) const {
  const Shard& shard = GetShard(cuid);
  std::shared_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it == shard.table.end()) return 0;

  const auto& entry = it->second;
  size_t count = 0;

  for (const auto& delta : entry.deltas) {
    // Key 范围重叠判断: !(delta.last < first || delta.first > last)
    bool is_left =
        (icmp_->user_comparator()->Compare(ExtractUserKey(delta.last_key),
                                           ExtractUserKey(first_key)) < 0);
    bool is_right =
        (icmp_->user_comparator()->Compare(ExtractUserKey(delta.first_key),
                                           ExtractUserKey(last_key)) > 0);
    if (!is_left && !is_right) {
      count++;
    }
  }
  return count;
}

void HotIndexTable::ClearAllForCuid(uint64_t cuid) {
  std::vector<DataSegment> segments_to_unref;
  {
    Shard& shard = GetShard(cuid);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    auto it = shard.table.find(cuid);
    if (it == shard.table.end()) return;

    auto& entry = it->second;

    // 收集需要 Unref 的 Snapshot 段
    for (const auto& seg : entry.snapshot_segments) {
      if (seg.file_number != static_cast<uint64_t>(-1)) {
        segments_to_unref.push_back(seg);
      }
    }

    // 清空所有数据
    entry.snapshot_segments.clear();
    entry.deltas.clear();
    entry.obsolete_deltas.clear();
  }

  // Unref lifecycle
  if (lifecycle_manager_) {
    for (const auto& seg : segments_to_unref) {
      lifecycle_manager_->Unref(seg.file_number);
    }
  }
}

void HotIndexTable::GetOverlappingSegments(
    uint64_t cuid, const std::string& first_key, const std::string& last_key,
    std::vector<DataSegment>* overlapping_snapshots,
    std::vector<DataSegment>* overlapping_deltas) {
  const Shard& shard = GetShard(cuid);
  std::shared_lock<std::shared_mutex> lock(shard.mutex);
  auto it = shard.table.find(cuid);
  if (it == shard.table.end()) return;

  const auto& entry = it->second;

  // 收集重叠的 snapshot segments
  for (const auto& seg : entry.snapshot_segments) {
    // !(seg.last < first || seg.first > last)
    bool is_left =
        (icmp_->user_comparator()->Compare(ExtractUserKey(seg.last_key),
                                           ExtractUserKey(first_key)) < 0);
    bool is_right =
        (icmp_->user_comparator()->Compare(ExtractUserKey(seg.first_key),
                                           ExtractUserKey(last_key)) > 0);
    if (!is_left && !is_right) {
      overlapping_snapshots->push_back(seg);
    }
  }

  // 收集重叠的 delta segments
  for (const auto& seg : entry.deltas) {
    bool is_left =
        (icmp_->user_comparator()->Compare(ExtractUserKey(seg.last_key),
                                           ExtractUserKey(first_key)) < 0);
    bool is_right =
        (icmp_->user_comparator()->Compare(ExtractUserKey(seg.first_key),
                                           ExtractUserKey(last_key)) > 0);
    if (!is_left && !is_right) {
      overlapping_deltas->push_back(seg);
    }
  }
}

void HotIndexTable::DumpToFile(const std::string& filename,
                               const std::string& phase_label) {
  std::ofstream outfile;
  outfile.open(filename, std::ios_base::app);

  if (!outfile.is_open()) {
    std::cerr << "Error: Unable to open dump file: " << filename << std::endl;
    return;
  }

  outfile << "========================================" << std::endl;
  outfile << "DUMP PHASE: " << phase_label << std::endl;

  size_t total_cuids = 0;
  for (size_t i = 0; i < shards_.size(); ++i) {
    std::shared_lock<std::shared_mutex> lock(shards_[i].mutex);
    total_cuids += shards_[i].table.size();
  }
  outfile << "Total Tracked CUIDs: " << total_cuids << std::endl;

  for (size_t i = 0; i < shards_.size(); ++i) {
    std::shared_lock<std::shared_mutex> lock(shards_[i].mutex);
    for (const auto& kv : shards_[i].table) {
      uint64_t cuid = kv.first;
      const HotIndexEntry& entry = kv.second;

      outfile << "CUID: " << cuid << std::endl;

      if (entry.snapshot_segments.empty()) {
        outfile << "  [Snapshot]: None" << std::endl;
      } else {
        outfile << "  [Snapshot]: " << entry.snapshot_segments.size()
                << " segments" << std::endl;
        for (const auto& seg : entry.snapshot_segments) {
          outfile << "    -> FileID: " << (int64_t)seg.file_number
                  << " | FirstKey: " << seg.first_key
                  << " | LastKey: " << seg.last_key;
          if (seg.file_number == (uint64_t)-1) outfile << " (Mem)";
          outfile << std::endl;
        }
      }

      if (entry.deltas.empty()) {
        outfile << "  [Deltas]: None" << std::endl;
      } else {
        outfile << "  [Deltas]: " << entry.deltas.size() << " segments"
                << std::endl;
        for (const auto& seg : entry.deltas) {
          outfile << "    -> FileID: " << seg.file_number
                  << " | FirstKey: " << seg.first_key
                  << " | LastKey: " << seg.last_key << std::endl;
        }
      }

      if (!entry.obsolete_deltas.empty()) {
        outfile << "  [Obsolete]: " << entry.obsolete_deltas.size()
                << " pending cleanup" << std::endl;
        for (const auto& seg : entry.obsolete_deltas) {
          outfile << "    -> FileID: " << seg.file_number << " (Obs)"
                  << std::endl;
        }
      }
    }
  }
  outfile << "----------------------------------------" << std::endl
          << std::endl;
  outfile.close();
}

}  // namespace ROCKSDB_NAMESPACE
