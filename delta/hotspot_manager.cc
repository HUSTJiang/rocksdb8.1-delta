//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "delta/hotspot_manager.h"

#include <chrono>
#include <cinttypes>
#include <sstream>

#include "delta/diag_log.h"

#include "db/dbformat.h"
#include "logging/logging.h"
#include "port/port.h"
#include "rocksdb/env.h"
#include "rocksdb/sst_file_writer.h"
#include "util/coding.h"
#include "util/extract_cuid.h"

namespace ROCKSDB_NAMESPACE {

HotspotManager::HotspotManager(const Options& db_options,
                               const std::string& data_dir,
                               const InternalKeyComparator* internal_comparator)
    : db_options_(db_options),
      data_dir_(data_dir),
      internal_comparator_(internal_comparator),
      lifecycle_manager_(std::make_shared<HotSstLifecycleManager>(db_options)),
      buffer_(db_options_.delta_options.hot_data_buffer_threshold_bytes, db_options_.delta_options.hot_data_buffer_shards),
      index_table_(internal_comparator, lifecycle_manager_, db_options.info_log,
                   db_options.delta_options.sharding_count),
      frequency_table_(db_options_.delta_options.hotspot_scan_threshold,
                       db_options_.delta_options.hotspot_scan_window_sec,
                       db_options_.delta_options.sharding_count),
      delete_table_(db_options_.delta_options.sharding_count) {
  db_options_.env->CreateDirIfMissing(data_dir_);

  // 恢复之前因为宕机等原因遗留的 GDCT 逻辑删除记录
  RecoverGDCT();
  // 初始化持久化日志 AppendableWriter
  InitGDCTLog();
  // 独立诊断日志文件（关键事件持久化，防止终端滚动丢失）
  DiagLogOpen((data_dir_ + "/delta_diag.log").c_str());
  DiagLogf("[DIAG_HOTSPOT_MANAGER_INIT] data_dir=%s sharding_count=%u\n",
           data_dir_.c_str(), db_options_.delta_options.sharding_count);
  // HotSST 生命周期独立日志（Register/Ref/Unref 事件）
  LifecycleLogOpen((data_dir_ + "/hotsst_lifecycle.log").c_str());
}

uint64_t HotspotManager::ExtractCUID(const Slice& key) {
  // TODO: 根据实际的 Key Schema提取 cuid，这里先假设一波
  if (key.size() < 24) {
    return 0;
  }

  const unsigned char* p =
      reinterpret_cast<const unsigned char*>(key.data()) + 16;

  // Big-Endian Decoding
  uint64_t cuid = (static_cast<uint64_t>(p[0]) << 56) |
                  (static_cast<uint64_t>(p[1]) << 48) |
                  (static_cast<uint64_t>(p[2]) << 40) |
                  (static_cast<uint64_t>(p[3]) << 32) |
                  (static_cast<uint64_t>(p[4]) << 24) |
                  (static_cast<uint64_t>(p[5]) << 16) |
                  (static_cast<uint64_t>(p[6]) << 8) |
                  (static_cast<uint64_t>(p[7]));

  return cuid;
}

bool HotspotManager::RegisterScan(uint64_t cuid, bool is_full_scan,
                                  bool* became_hot) {
  if (cuid == 0) {
    if (became_hot) *became_hot = false;
    return false;
  }

  bool first_time_hot = false;
  bool is_hot = frequency_table_.RecordAndCheckHot(cuid, &first_time_hot);

  // 返回是否首次成为热点
  if (became_hot) {
    *became_hot = first_time_hot;
  }

  if (is_hot && !first_time_hot) {
    // We do not eagerly initialize pending_snapshots_ here anymore.
    // It should be explicitly initialized by PrepareForFullReplace when a
    // FullScan really starts buffering.
  }

  if (is_full_scan) {
    delete_table_.ResetTracking(cuid);
  }
  return is_hot;
}

// Helper to extract row ID from key for gap detection
static uint64_t ExtractRowID(const std::string& key) {
  if (key.size() < 34) return 0;
  try {
    return std::stoull(key.substr(24, 10));
  } catch (...) {
    return 0;
  }
}

static std::unordered_map<uint64_t, uint64_t> diag_last_appended_row_id;
static std::mutex diag_append_mutex;

static std::unordered_map<uint64_t, uint64_t> diag_last_flushed_row_id;

bool HotspotManager::BufferHotData(uint64_t cuid, const Slice& key,
                                   const Slice& value) {
  uint64_t curr_id = ExtractRowID(key.ToString());
  if (curr_id != 0) {
    std::lock_guard<std::mutex> lock(diag_append_mutex);
    auto it = diag_last_appended_row_id.find(cuid);
    if (it != diag_last_appended_row_id.end() && it->second != 0) {
      if (it->second + 1 != curr_id && it->second < curr_id && curr_id - it->second < 50000) {
        DiagLogf("[DIAG_APPEND_GAP] CUID %lu GAP detected on append! "
                "Prev: %lu, Curr: %lu. Missing: %lu\n",
                cuid, it->second, curr_id, it->second + 1);
      }
    }
    diag_last_appended_row_id[cuid] = curr_id;
  }

  return buffer_.Append(cuid, key, value);
}

bool HotspotManager::PrepareForFullReplace(uint64_t cuid) {
  // PopPendingInitCuids 已经调用了 TryLockCuid，不需要调用 TryLockCuid
  std::lock_guard<std::mutex> lock(pending_mutex_);
  if (pending_snapshots_.find(cuid) == pending_snapshots_.end()) {
    pending_snapshots_[cuid] = std::vector<DataSegment>();
  }
  return true;
}

Status HotspotManager::InterceptDelete(const Slice& key, SequenceNumber seq,
                                       bool sync) {
  uint64_t cuid = ExtractCUID(key);
  if (cuid == 0) return Status::NotSupported("CUID not found");

  bool newly_deleted = false;
  // 在 GDCT 中查询是否该 cuid 已经被追踪为热点
  bool tracked = delete_table_.MarkDeleted(cuid, seq, &newly_deleted);

  if (tracked) {
    if (newly_deleted) {
      // 第一次 delete
      return PersistDelete(cuid, seq, sync);
    }
    return Status::OK();
  }
  // CUID 不在热点管理范围内
  return Status::NotSupported("Not a hot CUID");
}

Status HotspotManager::InitGDCTLog() {
  std::lock_guard<std::mutex> lock(gdct_log_mutex_);
  if (gdct_log_writer_) return Status::OK();

  std::string log_dir = data_dir_ + "/hot_shared_";
  db_options_.env->CreateDirIfMissing(log_dir);
  std::string log_path = log_dir + "/gdct.log";

  EnvOptions env_options;
  Status s = db_options_.env->ReopenWritableFile(log_path, &gdct_log_writer_,
                                                 env_options);
  if (s.IsNotSupported()) {
    std::cout << "[HotspotManager] ReopenWritableFile is not supported, using "
                 "NewWritableFile"
              << std::endl;
    // In a real environment fallback would go here depending on FS
  }
  if (!s.ok()) {
    ROCKS_LOG_ERROR(db_options_.info_log,
                    "[HotspotManager] Failed to open GDCT log file: %s",
                    s.ToString().c_str());
  }
  return s;
}

Status HotspotManager::FlushGDCTLogBuffer() {
  if (pending_gdct_records_.load(std::memory_order_relaxed) == 0) {
    return Status::OK();
  }

  std::vector<std::pair<uint64_t, SequenceNumber>> local_buffer;
  {
    std::lock_guard<std::mutex> lock(gdct_append_mutex_);
    if (gdct_append_buffer_.empty()) {
      pending_gdct_records_.store(0, std::memory_order_relaxed);
      return Status::OK();
    }
    local_buffer = std::move(gdct_append_buffer_);
    pending_gdct_records_.store(0, std::memory_order_relaxed);
  }

  last_gdct_flush_time_us_.store(db_options_.env->NowMicros(),
                                 std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(gdct_log_mutex_);
  if (!gdct_log_writer_) {
    InitGDCTLog();
    if (!gdct_log_writer_) return Status::IOError("No GDCT log writer");
  }

  Status s;
  for (const auto& record : local_buffer) {
    char buffer[16];
    EncodeFixed64(buffer, record.first);
    EncodeFixed64(buffer + 8, record.second);
    s = gdct_log_writer_->Append(Slice(buffer, 16));
    if (!s.ok()) break;
  }

  if (s.ok()) {
    s = gdct_log_writer_->Sync();
  }
  return s;
}

Status HotspotManager::PersistDelete(uint64_t cuid, SequenceNumber seq,
                                     bool sync) {
  if (sync) {
    // 同步写
    FlushGDCTLogBuffer();  // 先刷掉积压的
    std::lock_guard<std::mutex> lock(gdct_log_mutex_);
    if (!gdct_log_writer_) InitGDCTLog();
    if (!gdct_log_writer_) return Status::IOError("No GDCT log writer");

    char buffer[16];
    EncodeFixed64(buffer, cuid);
    EncodeFixed64(buffer + 8, seq);
    Status s = gdct_log_writer_->Append(Slice(buffer, 16));
    if (s.ok()) s = gdct_log_writer_->Sync();
    return s;
  } else {
    // 异步写 加到内存缓冲
    std::lock_guard<std::mutex> lock(gdct_append_mutex_);
    gdct_append_buffer_.push_back({cuid, seq});
    pending_gdct_records_.fetch_add(1, std::memory_order_relaxed);
    return Status::OK();
  }
}

void HotspotManager::RecoverGDCT() {
  std::string log_dir = data_dir_ + "/hot_shared_";
  std::string log_path = log_dir + "/gdct.log";
  std::unique_ptr<SequentialFile> file;
  EnvOptions env_options;

  Status s = db_options_.env->NewSequentialFile(log_path, &file, env_options);
  if (!s.ok()) {
    if (s.IsNotFound()) {
      return;  // 没有日志文件是正常的
    }
    ROCKS_LOG_WARN(db_options_.info_log,
                   "[HotspotManager] Failed to open GDCT log for recovery: %s",
                   s.ToString().c_str());
    return;
  }

  size_t recovered_count = 0;
  char buffer[16];
  Slice result;

  while (true) {
    s = file->Read(16, &result, buffer);
    if (!s.ok() || result.size() < 16) {
      break;
    }

    uint64_t cuid = DecodeFixed64(result.data());
    SequenceNumber seq = DecodeFixed64(result.data() + 8);

    delete_table_.MarkDeleted(cuid, seq);
    recovered_count++;
  }

  ROCKS_LOG_INFO(
      db_options_.info_log,
      "[HotspotManager] Recovered %zu CUID delete records from GDCT log.",
      recovered_count);
}

void HotspotManager::CompactAndFlushGDCTLogIfNeeded() {
  uint64_t now = db_options_.env->NowMicros();

  // 没有积压数据 && time interval >= 30s
  if (pending_gdct_records_.load(std::memory_order_relaxed) <
          db_options_.delta_options.gdct_flush_threshold_records &&
      now - last_gdct_flush_time_us_.load(std::memory_order_relaxed) <
          db_options_.delta_options.gdct_flush_interval_us) {
    return;
  }
  FlushGDCTLogBuffer();
  // 检查日志重写 (Compact) && time interval >= 600s
  if (now - last_gdct_compact_time_us_.load(std::memory_order_relaxed) <
      db_options_.delta_options.gdct_compact_interval_us) {
    return;
  }
  last_gdct_compact_time_us_.store(now, std::memory_order_relaxed);

  std::string log_dir = data_dir_ + "/hot_shared_";
  std::string log_path = log_dir + "/gdct.log";

  uint64_t file_size = 0;
  Status s = db_options_.env->GetFileSize(log_path, &file_size);
  if (!s.ok() ||
      file_size < db_options_.delta_options.gdct_log_compact_size) {
    return;
  }

  ROCKS_LOG_INFO(db_options_.info_log,
                 "[HotspotManager] GDCT log size %" PRIu64
                 " bytes, starting compaction.",
                 file_size);

  std::string new_log_path = log_dir + "/gdct.log.new";
  std::unique_ptr<WritableFile> new_writer;
  EnvOptions env_options;
  s = db_options_.env->NewWritableFile(new_log_path, &new_writer, env_options);
  if (!s.ok()) return;

  auto deleted_cuids = delete_table_.GetAllDeletedCuids();

  for (const auto& pair : deleted_cuids) {
    char buffer[16];
    EncodeFixed64(buffer, pair.first);
    EncodeFixed64(buffer + 8, pair.second);
    s = new_writer->Append(Slice(buffer, 16));
    if (!s.ok()) break;
  }

  if (s.ok()) s = new_writer->Sync();
  if (s.ok())
    s = new_writer->Close();
  else
    new_writer->Close();

  if (s.ok()) {
    std::lock_guard<std::mutex> lock(gdct_log_mutex_);
    if (gdct_log_writer_) {
      gdct_log_writer_->Close();
      gdct_log_writer_.reset();
    }
    s = db_options_.env->RenameFile(new_log_path, log_path);
    if (s.ok()) {
      db_options_.env->ReopenWritableFile(log_path, &gdct_log_writer_,
                                          env_options);
      ROCKS_LOG_INFO(db_options_.info_log,
                     "[HotspotManager] Successfully compacted GDCT log.");
    }
  }
}

std::string HotspotManager::GenerateSstFileName(uint64_t cuid) {
  auto now = std::chrono::system_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                       now.time_since_epoch())
                       .count();

  std::stringstream ss;
  ss << data_dir_ << "/hot_" << cuid << "_" << timestamp << ".sst";
  return ss.str();
}

bool HotspotManager::ShouldTriggerScanAsCompaction(uint64_t cuid) {
  HotIndexEntry entry;
  if (!index_table_.GetEntry(cuid, &entry)) {
    // a)	当前热点CUid无Snapshot。
    return true;
  }
  // b)已有Snapshot，且新增的Deltas片段数量超过阈值。
  if (!entry.HasSnapshot() ||
      entry.deltas.size() >
          db_options_.delta_options.sac_delta_count_threshold) {
    return true;
  }
  return false;
}

Status HotspotManager::FlushBlockToSharedSST(
    std::shared_ptr<HotDataBlock> block,
    std::unordered_map<uint64_t, std::vector<DataSegment>>* output_segments) {
  if (!block || block->buckets.empty()) return Status::OK();

  // keysort: already in RotateBuffer()

  EnvOptions env_options;
  SstFileWriter sst_writer(env_options, db_options_);

  // 文件名使用时间戳
  auto now = std::chrono::system_clock::now();
  uint64_t file_number = std::chrono::duration_cast<std::chrono::microseconds>(
                             now.time_since_epoch())
                             .count();
  std::string file_path =
      data_dir_ + "/hot_shared_" + std::to_string(file_number) + ".sst";

  Status s = sst_writer.Open(file_path);
  if (!s.ok()) return s;

  // link file，让 snapshot 的 iterator 能够正确找到文件
  // link_path = db_root / <file_number>.sst
  // data_dir_ = dbname_ / hotspot_data
  std::string db_root = data_dir_;
  size_t last_slash = db_root.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    db_root = db_root.substr(0, last_slash);
  }
  std::string link_path = db_root + "/" + std::to_string(file_number) + ".sst";

  s = db_options_.env->LinkFile(file_path, link_path);
  if (!s.ok()) {
    ROCKS_LOG_ERROR(db_options_.info_log,
                    "[HotspotManager] Failed to create link %s -> %s: %s",
                    file_path.c_str(), link_path.c_str(), s.ToString().c_str());
  }

  lifecycle_manager_->RegisterFile(file_number, file_path, link_path);

  // 3. 按 CUID 顺序写入数据
  // DataBlock 内部已经按 Bucket 组织，按 CUID 升序遍历 Bucket
  std::vector<uint64_t> cuids;
  for (const auto& pair : block->buckets) {
    cuids.push_back(pair.first);
  }
  std::sort(cuids.begin(), cuids.end());

  for (uint64_t current_cuid : cuids) {
    const auto& bucket_entries = block->buckets[current_cuid];
    if (bucket_entries.empty()) continue;

    const HotEntry& first_entry = bucket_entries[0];
    std::string last_written_key_in_segment;
    int written_count = 0;

    uint64_t first_id = ExtractRowID(first_entry.key);
    uint64_t last_id = ExtractRowID(bucket_entries.back().key);
    
    auto it = diag_last_flushed_row_id.find(current_cuid);
    if (it != diag_last_flushed_row_id.end()) {
      if (it->second != 0 && first_id != 0 && it->second + 1 != first_id && it->second < first_id) {
         DiagLogf("[DIAG_FLUSH_ACROSS_GAP] CUID %lu GAP detected across flushes! "
                 "Last flushed: %lu, This flush starts at: %lu. Missing: %lu\n",
                 current_cuid, it->second, first_id, it->second + 1);
      }
    }
    diag_last_flushed_row_id[current_cuid] = last_id;

    // 记录每个写入的 entry 的 InternalKey 和 row_id，用于 gap 检测分裂
    struct WrittenEntry {
      std::string internal_key;
      uint64_t row_id;
    };
    std::vector<WrittenEntry> written_entries;

    bool is_first_entry = true;
    for (const HotEntry& entry : bucket_entries) {
      uint64_t curr_id = ExtractRowID(entry.key);

      Slice entry_key(entry.key);
      Slice current_user_key_slice = ExtractUserKey(entry_key);
      std::string current_user_key = current_user_key_slice.ToString();

      // 去重
      if (!is_first_entry &&
          current_user_key == last_written_key_in_segment) {
        continue;
      }

      Slice entry_value(entry.value);
      s = sst_writer.Put(current_user_key_slice, entry_value);
      if (!s.ok()) return s;

      last_written_key_in_segment = current_user_key;
      written_entries.push_back({entry.key, curr_id});
      written_count++;
      is_first_entry = false;
    }

    if (written_count > 0) {
      // Gap 检测分裂：扫描相邻 row_id，gap >= 1 则拆为多个 segment
      std::vector<DataSegment>& cuid_segments = (*output_segments)[current_cuid];
      size_t seg_start_idx = 0;
      for (size_t i = 1; i < written_entries.size(); ++i) {
        uint64_t prev_rid = written_entries[i - 1].row_id;
        uint64_t curr_rid = written_entries[i].row_id;
        if (prev_rid != 0 && curr_rid != 0 &&
            curr_rid > prev_rid + 1) {
          // Gap detected: close current segment [seg_start_idx, i-1]
          DataSegment segment;
          segment.file_number = file_number;
          segment.first_key = written_entries[seg_start_idx].internal_key;
          segment.last_key = written_entries[i - 1].internal_key;
          cuid_segments.push_back(segment);
          DiagLogf("[DIAG_FLUSH_GAP_SPLIT] CUID %lu: Split at gap! "
                  "prev_rid=%lu curr_rid=%lu gap=%lu. "
                  "Segment [%s - %s] (%zu entries)\n",
                  current_cuid, prev_rid, curr_rid,
                  curr_rid - prev_rid - 1,
                  FormatKeyDisplay(segment.first_key).c_str(),
                  FormatKeyDisplay(segment.last_key).c_str(),
                  i - seg_start_idx);
          seg_start_idx = i;
        }
      }
      // Close last (or only) segment
      DataSegment last_segment;
      last_segment.file_number = file_number;
      last_segment.first_key = written_entries[seg_start_idx].internal_key;
      last_segment.last_key = written_entries.back().internal_key;
      cuid_segments.push_back(last_segment);

      if (cuid_segments.size() > 1) {
        DiagLogf("[DIAG_FLUSH_GAP_SPLIT] CUID %lu: Total %zu segments from %d entries\n",
                current_cuid, cuid_segments.size(), written_count);
        // Dump current snapshot & delta state so post-mortem can see what rows
        // are supposed to exist in SST vs. buffer when this gap formed.
        HotIndexEntry _snap_entry;
        if (index_table_.GetEntry(current_cuid, &_snap_entry)) {
          DiagLogf("[DIAG_FLUSH_GAP_SPLIT_SNAP] CUID %lu: at flush time: "
                   "%zu snapshot_segs, %zu deltas, %zu obsolete_deltas\n",
                   current_cuid,
                   _snap_entry.snapshot_segments.size(),
                   _snap_entry.deltas.size(),
                   _snap_entry.obsolete_deltas.size());
          for (size_t _si = 0; _si < _snap_entry.snapshot_segments.size(); _si++) {
            const auto& _s = _snap_entry.snapshot_segments[_si];
            DiagLogf("[DIAG_FLUSH_GAP_SPLIT_SNAP] CUID %lu:   snap[%zu] file=%lu [%s - %s]\n",
                     current_cuid, _si, _s.file_number,
                     FormatKeyDisplay(_s.first_key).c_str(),
                     FormatKeyDisplay(_s.last_key).c_str());
          }
          for (size_t _di = 0; _di < _snap_entry.deltas.size(); _di++) {
            const auto& _d = _snap_entry.deltas[_di];
            DiagLogf("[DIAG_FLUSH_GAP_SPLIT_SNAP] CUID %lu:   delta[%zu] file=%lu [%s - %s]\n",
                     current_cuid, _di, _d.file_number,
                     FormatKeyDisplay(_d.first_key).c_str(),
                     FormatKeyDisplay(_d.last_key).c_str());
          }
        }
      }

      // [DIAG] detect dedup loss during flush
      size_t bucket_size = bucket_entries.size();
      if (bucket_size != static_cast<size_t>(written_count)) {
        DiagLogf("[DIAG_FLUSH_DEDUP] CUID %lu: bucket_entries=%zu "
                "written_count=%d dedup_dropped=%zu\n",
                current_cuid, bucket_size, written_count,
                bucket_size - written_count);
      }
    }
  }

  // 4. Finish
  ExternalSstFileInfo file_info;
  s = sst_writer.Finish(&file_info);
  if (!s.ok()) {
    return s;
  }

  uint64_t actual_file_size = file_info.file_size;
  for (auto& kv : *output_segments) {
    for (auto& seg : kv.second) {
      seg.file_size = actual_file_size;
    }
  }

  ROCKS_LOG_INFO(db_options_.info_log,
                 "[HotspotManager] Flushed Shared SST: %s, CUIDs: %zu",
                 file_path.c_str(), output_segments->size());

  return Status::OK();
}

void HotspotManager::TriggerBufferFlush(const char* source,
                                        uint64_t source_cuid) {
  (void)source;
  (void)source_cuid;
  // 防止多次 flush 
  std::lock_guard<std::mutex> flush_lock(flush_mutex_);
  // 轮转 Buffer，传入 InternalKeyComparator 保证排序严格有序
  if (!buffer_.RotateBuffer(internal_comparator_)) {
    return;
  }

  // 提取待刷盘 Block
  // 先获取 block 执行刷盘，成功后再从队列移除
  // 防止 reader 再 flush 过程中看不到数据
  ROCKS_LOG_DEBUG(db_options_.info_log,
                  "[HotspotManager] Buffer rotated. Preparing to flush block.");
  auto block = buffer_.GetFrontBlockForFlush();
  while (block) {
    std::string block_min_key;
    std::string block_max_key;
    bool has_block_range = false;
    for (const auto& kv : block->bounds) {
      const auto& b = kv.second;
      if (b.min_key.empty() || b.max_key.empty()) {
        continue;
      }
      if (!has_block_range) {
        block_min_key = b.min_key;
        block_max_key = b.max_key;
        has_block_range = true;
      } else {
        if (internal_comparator_->Compare(b.min_key, block_min_key) < 0) {
          block_min_key = b.min_key;
        }
        if (internal_comparator_->Compare(b.max_key, block_max_key) > 0) {
          block_max_key = b.max_key;
        }
      }
    }


    std::unordered_map<uint64_t, std::vector<DataSegment>> new_segments;
    // to share SST
    Status s = FlushBlockToSharedSST(std::move(block), &new_segments);

    if (s.ok()) {
      for (const auto& kv : new_segments) {
        uint64_t cuid = kv.first;
        // 如果 FlushBlockToSharedSST gap-split 产生了多个段，记录当前 flush 上下文
        if (kv.second.size() > 1) {
          const char* split_ctx = "normal";
          {
            std::lock_guard<std::mutex> pm_lk(pm_pending_mutex_);
            if (pm_pending_snapshots_.count(cuid)) split_ctx = "PM";
          }
          if (split_ctx[0] == 'n') {  // still "normal"
            std::lock_guard<std::mutex> lk(pending_mutex_);
            if (pending_snapshots_.count(cuid)) split_ctx = "InitScan";
          }
          DiagLogf("[DIAG_FLUSH_GAP_SPLIT_CTX] CUID %lu: %zu segments from "
                  "gap-split (ctx=%s)\n",
                  cuid, kv.second.size(), split_ctx);
        }
        for (const DataSegment& real_segment : kv.second) {

          // 注意，这里必须要先调用 PromoteSnapshot
          // 否则这次 PM 之前的其他 PM 的 initscan 的 scan 数据对应 -1 segment
          // 不能正确地被替换导致正在进行的 scan 读取不到数据
          bool phys_overlap = false;
          bool promoted = index_table_.PromoteSnapshot(
              cuid, real_segment, &buffer_, internal_comparator_, &phys_overlap);

          // [DIAG] log every PromoteSnapshot result for hot CUIDs
          if (frequency_table_.IsHot(cuid)) {
            DiagLogf("[DIAG_FLUSH_PROMOTE] CUID %lu: SST file=%lu [%s - %s] "
                    "promoted=%d phys_overlap=%d source=%s\n",
                    cuid, real_segment.file_number,
                    FormatKeyDisplay(real_segment.first_key).c_str(),
                    FormatKeyDisplay(real_segment.last_key).c_str(),
                    (int)promoted, (int)phys_overlap, source);
          }

          // [FIX] 若 PromoteSnapshot 成功，且该 CUID 有活跃 PM，
          // 将此 SST 记录到 pm_promoted_snapshots_，供 AtomicReplaceForPartialMerge 保留。
          // AtomicReplace 步骤① 会移除 pm_range 内所有旧 segment，
          // 若不记录则刚 promoted 的 SST 也会被移除，导致 GAP。
          // 注意：PromoteSnapshot 已为此 SST Ref，这里不额外 Ref。
          if (promoted) {
            std::lock_guard<std::mutex> pm_lock(pm_pending_mutex_);
            if (pm_pending_snapshots_.count(cuid)) {
              pm_promoted_snapshots_[cuid].push_back(real_segment);
              DiagLogf("[DIAG_FLUSH_PROMOTE_TRACKED] CUID %lu: SST file=%lu [%s - %s]"
                      " → pm_promoted_snapshots (preserved by AtomicReplace)\n",
                      cuid, real_segment.file_number,
                      FormatKeyDisplay(real_segment.first_key).c_str(),
                      FormatKeyDisplay(real_segment.last_key).c_str());
            }
          }

          // 【1】PartialMerge
          // -1 段尚未建立（或仅有物理段重叠），将 SST 存入 pm_pending 并 Ref
          // 在 FinalizePmPendingSnapshots / AtomicReplace 统一处理
          if (!promoted) {
            bool intercepted = false;
            {
              std::lock_guard<std::mutex> lock(pm_pending_mutex_);
              auto pm_it = pm_pending_snapshots_.find(cuid);
              if (pm_it != pm_pending_snapshots_.end()) {
                pm_it->second.push_back(real_segment);
                lifecycle_manager_->Ref(real_segment.file_number);
                intercepted = true;
                DiagLogf("[DIAG_FLUSH_INTERCEPT] CUID %lu: SST file=%lu → pm_pending "
                        "(pm_pending_count=%zu)\n",
                        cuid, real_segment.file_number, pm_it->second.size());
              }
            }

            // 【2】Init Scan pending【热点 CUID 初始化】
            // Init Scan 持有 CUID Lock，snapshot_segments 必为空，
            // PromoteSnapshot 调用后一定返回 false（无 -1 段可替换）。
            // 与 PM pending 保持相同模式：存入 pending + Ref + continue，
            // 等 FinalizeScanAsCompaction 统一构建初始 snapshot。
            if (!intercepted) {
              std::lock_guard<std::mutex> lock2(pending_mutex_);
              auto it = pending_snapshots_.find(cuid);
              if (it != pending_snapshots_.end()) {
                it->second.push_back(real_segment);
                lifecycle_manager_->Ref(real_segment.file_number);
                intercepted = true;
                DiagLogf("[DIAG_FLUSH_INTERCEPT] CUID %lu: SST file=%lu → init_pending "
                        "(pending_count=%zu)\n",
                        cuid, real_segment.file_number, it->second.size());
              }
            }

            if (!intercepted) {
              if (phys_overlap) {
                // 仅物理段重叠，无活跃 PM/InitScan：SST 与已有物理段冗余，
                // 安全跳过（数据已在物理段中），不执行 AppendSnapshotSegment
                // 以免产生重叠段
                DiagLogf("[DIAG][TriggerBufferFlush] CUID %lu: SST file=%lu "
                        "[%s - %s] has PHYS_OVERLAP only, no active PM/InitScan. "
                        "Skipping (data already in physical segment).\n",
                        cuid, real_segment.file_number,
                        FormatKeyDisplay(real_segment.first_key).c_str(),
                        FormatKeyDisplay(real_segment.last_key).c_str());
              } else {
                // 无 -1 段也无任何活跃 Scan，强制 Append 兜底
                ROCKS_LOG_WARN(db_options_.info_log,
                              "[HotspotManager] No active scan or {-1} segment "
                              "for CUID %" PRIu64
                              ". Appending snapshot segment directly.",
                              cuid);
                index_table_.AppendSnapshotSegment(cuid, real_segment);
              }
            } 
          }
        } // end for each segment in vector
      }
    } else {
      ROCKS_LOG_ERROR(db_options_.info_log,
                      "[HotspotManager] Failed to flush shared block: %s",
                      s.ToString().c_str());
    }
    // 刷盘和index更新完成之后再移除
    buffer_.PopFrontBlockAfterFlush();
    // 【DIAG】Flush 完成后检查是否有悬空 -1 段（buffer 已无法支撑的内存段）
    if (s.ok()) {
      for (const auto& kv : new_segments) {
        uint64_t flushed_cuid = kv.first;
        if (!frequency_table_.IsHot(flushed_cuid)) continue;
        HotIndexEntry entry_chk;
        index_table_.GetEntry(flushed_cuid, &entry_chk);
        std::string chk_buf_min, chk_buf_max;
        bool chk_has_buf = buffer_.GetBoundaryKeys(
            flushed_cuid, &chk_buf_min, &chk_buf_max, internal_comparator_);
        for (const auto& seg : entry_chk.snapshot_segments) {
          if (seg.file_number != static_cast<uint64_t>(-1)) continue;
          // 检查 buffer 中是否仍有覆盖该 -1 段范围的数据
          bool seg_backed =
              chk_has_buf &&
              internal_comparator_->user_comparator()->Compare(
                  ExtractUserKey(chk_buf_max),
                  ExtractUserKey(seg.first_key)) >= 0 &&
              internal_comparator_->user_comparator()->Compare(
                  ExtractUserKey(chk_buf_min),
                  ExtractUserKey(seg.last_key)) <= 0;
          if (!seg_backed) {
            DiagLogf("[DIAG_WARN_ORPHAN_MEM_SEG] CUID %lu: "
                    "-1 segment [%s - %s] has NO buffer backing after flush! "
                    "(has_buf=%d buf_range=[%s - %s])\n",
                    flushed_cuid,
                    FormatKeyDisplay(seg.first_key).c_str(),
                    FormatKeyDisplay(seg.last_key).c_str(),
                    (int)chk_has_buf,
                    chk_has_buf ? FormatKeyDisplay(chk_buf_min).c_str() : "N/A",
                    chk_has_buf ? FormatKeyDisplay(chk_buf_max).c_str() : "N/A");
          }
        }
      }
    }
    block = buffer_.GetFrontBlockForFlush();
  }
}

void HotspotManager::FinalizeScanAsCompaction(
    uint64_t cuid, const std::unordered_set<uint64_t>& visited_files,
    const std::string& scan_first_key, const std::string& scan_last_key) {
  if (cuid == 0) return;
  auto unlock_guard = std::shared_ptr<void>( 
    nullptr, [this, cuid](void*) { this->UnlockCuid(cuid); });

  std::vector<DataSegment> final_segments;

  // A full scan can finish with the last hot range still sitting in the
  // in-memory buffer. Flush it here so initial snapshots materialize as SSTs
  // instead of only an in-memory tail segment.
  std::string pending_buf_min, pending_buf_max;
  if (buffer_.GetBoundaryKeys(cuid, &pending_buf_min, &pending_buf_max,
                              internal_comparator_)) {
    DiagLogf("[DIAG_FINALIZE_FLUSH] CUID %lu: flushing final buffered range [%s - %s]\n",
             cuid, FormatKeyDisplay(pending_buf_min).c_str(),
             FormatKeyDisplay(pending_buf_max).c_str());
    TriggerBufferFlush("FinalizeScanAsCompaction", cuid);
  }

  // Scan 过程中产生的 Pending SSTs【在 scan 的过程中flush的】
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_snapshots_.find(cuid);
    if (it != pending_snapshots_.end()) {
      final_segments = std::move(it->second);
      pending_snapshots_.erase(it);
    }
  }

  // 直接查询 buffer 是否还有该 CUID 的未 flush 数据？
  // 性能怎么样？
  std::string buf_min, buf_max;
  bool has_buffered_data = buffer_.GetBoundaryKeys(cuid, &buf_min, &buf_max,
                                                   internal_comparator_);

  // Init Scan 首次执行时 CUID 无先前 snapshot，两者同时为空说明该 CUID
  // 在 LSM 中根本没有数据
  if (final_segments.empty() && !has_buffered_data) {
    ROCKS_LOG_INFO(db_options_.info_log,
                   "[HotspotManager] FinalizeScanAsCompaction for CUID %" PRIu64
                   " skipped: no pending SSTs and no buffered data (empty CUID).",
                   cuid);
    return;
  }

  // ===== 第一步：先排序并消解 pending SST 之间的重叠 =====
  // 按道理来说，initial scan 的 sst 不应该重叠
  // 必须在创建 -1 尾段之前完成，否则 tail_min 会基于未排序的数据计算
  if (final_segments.size() > 1) {
    std::sort(final_segments.begin(), final_segments.end(),
              [this](const DataSegment& a, const DataSegment& b) {
                return internal_comparator_->Compare(a.first_key, b.first_key) < 0;
              });
    // 消解重叠：将前一个段的 last_key 裁剪到后一个段的 first_key
    for (size_t i = 0; i + 1 < final_segments.size(); ++i) {
      if (internal_comparator_->Compare(final_segments[i].last_key,
                                        final_segments[i + 1].first_key) > 0) {
        final_segments[i].last_key = final_segments[i + 1].first_key;
      }
    }
    // 移除裁剪后失效的空段 (first_key > last_key)
    for (auto it = final_segments.begin(); it != final_segments.end(); ) {
      if (!it->last_key.empty() && 
          internal_comparator_->Compare(it->first_key, it->last_key) > 0) {
        // 释放 pending 路径的 Ref，防止泄漏
        if (it->file_number != static_cast<uint64_t>(-1)) {
          lifecycle_manager_->Unref(it->file_number);
        }
        it = final_segments.erase(it);
      } else {
        ++it;
      }
    }
  }

  // 边界数据连续性检测：在追加 -1 尾段之前检查 pending SSTs 自身是否存在 GAP
  if (final_segments.size() > 1) {
    for (size_t i = 0; i + 1 < final_segments.size(); ++i) {
      const auto& fa = final_segments[i];
      const auto& fb = final_segments[i + 1];
      uint64_t rid_a = ExtractRowID(fa.last_key);
      uint64_t rid_b = ExtractRowID(fb.first_key);
      if (rid_a != 0 && rid_b != 0 && rid_a + 1 != rid_b) {
        const char* issue = (rid_a >= rid_b) ? "OVERLAP" : "GAP";
        DiagLogf("[DIAGNOSE_FATAL] FinalizeScanAsCompaction_PendingSSTs %s! "
                "CUID %lu Seg[%zu](file=%lu last_id=%lu) -> "
                "Seg[%zu](file=%lu first_id=%lu) missing=%lu\n",
                issue, cuid,
                i,     fa.file_number, rid_a,
                i + 1, fb.file_number, rid_b,
                (rid_a < rid_b) ? (rid_b - rid_a - 1) : 0);
        DiagLogf("  Seg[%zu] [%s - %s]\n", i,
                FormatKeyDisplay(fa.first_key).c_str(),
                FormatKeyDisplay(fa.last_key).c_str());
        DiagLogf("  Seg[%zu] [%s - %s]\n", i + 1,
                FormatKeyDisplay(fb.first_key).c_str(),
                FormatKeyDisplay(fb.last_key).c_str());
      }
    }
  }

  // ===== 第二步：计算 -1 尾段的起点并追加（如果 buffer 中还有剩余数据）=====
  // Init Scan 时 CUID 无先前 snapshot，所有 SST 都经过了 scan_first/last clip，
  // 因此 tail_min 必定 <= scan_last_key（clipping 已保证）
  std::string tail_min;
  if (final_segments.empty()) { // 还没有生成 SST 
    tail_min = scan_first_key;
  } else {
    // 基于排序+裁剪后的最后一个 SST 的 last_key
    tail_min = final_segments.back().last_key;
  }

  if (has_buffered_data) {
    // buffer 中还有未 flush 的数据（最后一个 SST flush 之后到 scan_last_key 的范围）
    DataSegment tail_segment;
    tail_segment.file_number = static_cast<uint64_t>(-1);  // 标记为内存段
    tail_segment.first_key = tail_min;
    tail_segment.last_key = scan_last_key;
    final_segments.push_back(tail_segment);
  } else if (final_segments.empty()) {
    // 无 buffer 数据且无 SST segments（逻辑上不应在此出现，已被前面的 early return 拦截）
    return;
  }
  // else: 全部数据已精确 flush 到 SST，最后 flush 点恰好等于 scan_last_key，
  // 无需额外 -1 尾段，直接使用 final_segments。

  // 【debug】验证 final_segments 数据连续性（含重叠和 GAP）
  if (final_segments.size() > 1) {
    for (size_t i = 0; i < final_segments.size() - 1; ++i) {
      const auto& s1 = final_segments[i];
      const auto& s2 = final_segments[i+1];
      uint64_t last_id  = ExtractRowID(s1.last_key);
      uint64_t next_id  = ExtractRowID(s2.first_key);
      // 连续性：相邻段的行 ID 应恰好相差 1
      if (last_id != 0 && next_id != 0 && last_id + 1 != next_id) {
        const char* issue = (last_id >= next_id) ? "OVERLAP" : "GAP";
        DiagLogf("[DIAGNOSE_FATAL] FinalizeScanAsCompaction %s detected! "
                "CUID %lu Seg[%zu](file=%lu last_id=%lu) -> "
                "Seg[%zu](file=%lu first_id=%lu) missing=%lu\n",
                issue, cuid,
                i,   s1.file_number, last_id,
                i+1, s2.file_number, next_id,
                (last_id < next_id) ? (next_id - last_id - 1) : 0);
        DiagLogf("  Seg[%zu] [%s - %s]\n",
                i,   FormatKeyDisplay(s1.first_key).c_str(),
                     FormatKeyDisplay(s1.last_key).c_str());
        DiagLogf("  Seg[%zu] [%s - %s]\n",
                i+1, FormatKeyDisplay(s2.first_key).c_str(),
                     FormatKeyDisplay(s2.last_key).c_str());
      }
    }
  }

  ROCKS_LOG_INFO(db_options_.info_log,
                 "[HotspotManager] FinalizeScanAsCompaction for CUID %" PRIu64
                 " updating snapshot with %zu segments.",
                 cuid, final_segments.size());

  // 建立这个 cuid 的 Snapshot
  index_table_.UpdateSnapshot(cuid, final_segments);

  // 释放 pending 路径的 Ref。
  // UpdateSnapshot 内部已经为 final_segments 中的每个 segment 做了 Ref(+1)，
  // 此处释放 pending 添加时的 Ref，使 ref_count 最终为 1
  for (const auto& seg : final_segments) {
    if (seg.file_number != static_cast<uint64_t>(-1)) {
      lifecycle_manager_->Unref(seg.file_number);
    }
  }
  // 在 snapshot 中已有数据，将这次 initialscan 经历的 rocksdb 文件标记为 obsolete
  index_table_.MarkDeltasAsObsolete(cuid, visited_files);
}

ScanAsCompactionStrategy HotspotManager::EvaluateScanAsCompactionStrategy(
    uint64_t cuid, bool is_full_scan, const std::string& scan_first_key,
    const std::string& scan_last_key, size_t* out_involved_delta_count) {
  // Init Scan（skip_hot_path=true，delta_full_scan=true）走 FullReplace 路径
  // 此时 CUID 必定无 snapshot，FinalizeScanAsCompaction 会建立初始 snapshot
  if (is_full_scan) {
    if (out_involved_delta_count) {
      *out_involved_delta_count = 0;
    }
    return ScanAsCompactionStrategy::kFullReplace;
  }

  // 如果没有 Snapshot 不进行 PartialMerge
  HotIndexEntry entry;
  if (!index_table_.GetEntry(cuid, &entry) || !entry.HasSnapshot()) {
    if (out_involved_delta_count) {
      *out_involved_delta_count = 0;
    }
    return ScanAsCompactionStrategy::kNoAction;
  }

  // 小 Scan: 查看涉及多少个 delta segments
  size_t delta_count = CountInvolvedDeltas(cuid, scan_first_key, scan_last_key);
  if (out_involved_delta_count) {
    *out_involved_delta_count = delta_count;
  }

  // 根据阈值决定策略
  if (delta_count >= db_options_.delta_options.delta_merge_threshold) {
    return ScanAsCompactionStrategy::kPartialMerge;
  }

  return ScanAsCompactionStrategy::kNoAction;
}

size_t HotspotManager::CountInvolvedDeltas(uint64_t cuid,
                                           const std::string& first_key,
                                           const std::string& last_key) {
  return index_table_.CountOverlappingDeltas(cuid, first_key, last_key);
}

void HotspotManager::FinalizeScanAsCompactionWithStrategy(
    uint64_t cuid, ScanAsCompactionStrategy strategy,
    const std::string& scan_first_key, const std::string& scan_last_key,
    const std::unordered_set<uint64_t>& visited_files,
    const std::vector<std::pair<std::string, std::string>>& scan_data) {
  if (cuid == 0) return;

  switch (strategy) {
    case ScanAsCompactionStrategy::kNoAction:
      return;
    case ScanAsCompactionStrategy::kFullReplace:
      FinalizeScanAsCompaction(cuid, visited_files, scan_first_key,
                               scan_last_key);
      break;
    case ScanAsCompactionStrategy::kPartialMerge:
      EnqueuePartialMerge(cuid, scan_first_key, scan_last_key, scan_data);
      break;
  }
}

// ----------------- L0Compaction --------------------

void HotspotManager::UpdateCompactionDelta(
    uint64_t cuid, const std::vector<uint64_t>& input_files,
    uint64_t output_file_number, const std::string& first_key,
    const std::string& last_key) {
  if (first_key.empty()) return;

  DataSegment seg;
  seg.file_number = output_file_number;
  seg.first_key = first_key;
  seg.last_key = last_key;

  index_table_.UpdateDeltaIndex(cuid, input_files, seg);
}

bool HotspotManager::ShouldSkipObsoleteDelta(
    uint64_t cuid, const std::vector<uint64_t>& input_files) {
  return index_table_.IsDeltaObsolete(cuid, input_files);
}

void HotspotManager::CleanUpMetadataAfterCompaction(
    const std::unordered_set<uint64_t>& involved_cuids,
    const std::vector<uint64_t>& input_files) {
  if (input_files.empty() || involved_cuids.empty()) return;

  // 步骤d：热点索引表的处理
  index_table_.RemoveObsoleteDeltasForCUIDs(involved_cuids, input_files);

  // 步骤c：已删除cuid hotdata文件的处理
  for (uint64_t cuid : involved_cuids) {
    delete_table_.UntrackFiles(cuid, input_files);
  }
}

bool HotspotManager::IsHot(uint64_t cuid) {
  return frequency_table_.IsHot(cuid);
}

// --------------------- HotspotManager Iterator --------------------- //
InternalIterator* HotspotManager::NewBufferIterator(
    uint64_t cuid, const InternalKeyComparator* icmp) {
  return buffer_.NewIterator(cuid, icmp);
}

// --------------------- Pending Init CUID Queue --------------------- //
void HotspotManager::EnqueueForInitScan(uint64_t cuid) {
  std::lock_guard<std::mutex> lock(pending_init_mutex_);
  // 避免重复添加
  for (uint64_t c : pending_init_cuids_) {
    if (c == cuid) return;
  }
  pending_init_cuids_.push_back(cuid);
  DiagLogf("[DIAG_INIT_SCAN_ENQUEUE] CUID %lu\n", cuid);
  ROCKS_LOG_INFO(
      db_options_.info_log,
      "[HotspotManager] Enqueued CUID %" PRIu64 " for initial full scan", cuid);
}

std::vector<uint64_t> HotspotManager::PopPendingInitCuids(size_t max_count) {
  std::lock_guard<std::mutex> lock(pending_init_mutex_);
  std::vector<uint64_t> result;
  for (auto it = pending_init_cuids_.begin();
       it != pending_init_cuids_.end() && result.size() < max_count;) {
    if (TryLockCuid(*it)) {
      result.push_back(*it);
      it = pending_init_cuids_.erase(it);
    } else {
      ++it;
    }
  }
  return result;
}

bool HotspotManager::HasPendingInitCuids() const {
  std::lock_guard<std::mutex> lock(pending_init_mutex_);
  return !pending_init_cuids_.empty();
}

// --------------------- Metadata Scan Queue --------------------- //
void HotspotManager::EnqueueMetadataScan(uint64_t cuid) {
  std::lock_guard<std::mutex> lock(pending_metadata_mutex_);
  for (uint64_t c : pending_metadata_scans_) {
    if (c == cuid) return;
  }
  pending_metadata_scans_.push_back(cuid);
  // fprintf(stdout,
  //         "[HotspotManager] Enqueued CUID %lu for metadata ref-count scan\n",
  //         cuid);
}

std::vector<uint64_t> HotspotManager::PopPendingMetadataScans(size_t max_count) {
  std::lock_guard<std::mutex> lock(pending_metadata_mutex_);
  std::vector<uint64_t> result;
  while (!pending_metadata_scans_.empty() && result.size() < max_count) {
    result.push_back(pending_metadata_scans_.front());
    pending_metadata_scans_.erase(pending_metadata_scans_.begin());
  }
  return result;
}

bool HotspotManager::HasPendingMetadataScans() const {
  std::lock_guard<std::mutex> lock(pending_metadata_mutex_);
  return !pending_metadata_scans_.empty();
}

// --------------------- Partial Merge Queue --------------------- //

void HotspotManager::EnqueuePartialMerge(
    uint64_t cuid, const std::string& scan_first_key,
    const std::string& scan_last_key,
    const std::vector<std::pair<std::string, std::string>>& scan_data) {
  if (!TryLockCuid(cuid)) {
    return;
  }

  std::lock_guard<std::mutex> lock(partial_merge_mutex_);
  // 避免重复添加相同 cuid 的任务?
  for (auto& task : partial_merge_queue_) {
    if (task.cuid == cuid) {
      // 已经有一个队列了，暂时忽略本次小的 scan触发，或者也可以用更大的
      // range更新
      UnlockCuid(cuid);
      return;
    }
  }
  partial_merge_queue_.push_back(
      {cuid, scan_first_key, scan_last_key, scan_data});
  RegisterPmPending(cuid);
  ROCKS_LOG_INFO(db_options_.info_log,
                 "[HotspotManager] Enqueued PartialMerge for CUID %" PRIu64
                 ", range [%zu, %zu], %zu pairs",
                 cuid, scan_first_key.size(), scan_last_key.size(),
                 scan_data.size());
}

bool HotspotManager::HasPendingPartialMerge() const {
  std::lock_guard<std::mutex> lock(partial_merge_mutex_);
  return !partial_merge_queue_.empty();
}

bool HotspotManager::PopPendingPartialMerge(PartialMergePendingTask* task) {
  std::lock_guard<std::mutex> lock(partial_merge_mutex_);
  if (partial_merge_queue_.empty()) {
    return false;
  }

  *task = std::move(partial_merge_queue_.front());
  partial_merge_queue_.erase(partial_merge_queue_.begin());
  return true;
}

bool HotspotManager::TryLockCuid(uint64_t cuid) {
  std::lock_guard<std::mutex> lock(in_progress_mutex_);
  if (in_progress_cuids_.count(cuid)) {
    return false;
  }
  in_progress_cuids_.insert(cuid);
  return true;
}

void HotspotManager::UnlockCuid(uint64_t cuid) {
  std::lock_guard<std::mutex> lock(in_progress_mutex_);
  in_progress_cuids_.erase(cuid);
}

void HotspotManager::RegisterPmPending(uint64_t cuid) {
  std::lock_guard<std::mutex> lock(pm_pending_mutex_);
  // emplace 若已存在则不覆盖
  pm_pending_snapshots_.emplace(cuid, std::vector<DataSegment>{});
}

void HotspotManager::FinalizePmPendingSnapshots(uint64_t cuid) {
  std::vector<DataSegment> pm_ssts;
  {
    std::lock_guard<std::mutex> lock(pm_pending_mutex_);
    auto it = pm_pending_snapshots_.find(cuid);
    if (it != pm_pending_snapshots_.end()) {
      pm_ssts = std::move(it->second);
      pm_pending_snapshots_.erase(it);
    }
    // pm_promoted_snapshots_ 在 SwapOutPmPromoted 中已被取出并清空；
    // 若 AtomicReplace 未被调用（written_count=0 路径），在此处兜底清理。
    pm_promoted_snapshots_.erase(cuid);
  }

  if (pm_ssts.empty()) return;

  // 逐个处理跨线程 flush 产生的 SST。
  // 调用时，-1 段已由 ReplaceOverlappingSegments 建立，由 PromoteSnapshot 找到 -1 段并替换
  for (const auto& seg : pm_ssts) {
    bool promoted = index_table_.PromoteSnapshot(
        cuid, seg, &buffer_, internal_comparator_);
    if (!promoted) {
      DiagLogf("[DIAG][FinalizePmPending] CUID %lu: PromoteSnapshot failed for "
              "pm_pending SST file=%lu [%s - %s], falling back to Append.\n",
              cuid, seg.file_number,
              FormatKeyDisplay(seg.first_key).c_str(),
              FormatKeyDisplay(seg.last_key).c_str());
      // AppendSnapshotSegment 内部会 Ref(+1)，下面的 Unref -> Ref 平衡
      index_table_.AppendSnapshotSegment(cuid, seg);
    }
    // 释放 pm_pending 路径的额外 Ref
    lifecycle_manager_->Unref(seg.file_number);
  }
}

}  // namespace ROCKSDB_NAMESPACE
