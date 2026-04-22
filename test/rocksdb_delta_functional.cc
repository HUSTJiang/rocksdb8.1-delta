#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "db/db_impl/db_impl.h"
#include "delta/delta_perf_counters.h"
#include "delta/global_delete_count_table.h"
#include "delta/hotspot_manager.h"
#include "port/port.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"
#include "rocksdb/write_batch.h"

namespace rocksdb {

namespace {

void Fail(const std::string& message) {
  std::cerr << "[FAIL] " << message << std::endl;
  std::exit(1);
}

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

void ExpectStatus(const Status& s, const std::string& message) {
  if (!s.ok()) {
    Fail(message + ": " + s.ToString());
  }
}

std::string GenerateKey(uint64_t cuid, uint64_t row_id) {
  std::string key(40, '\0');
  unsigned char* p = reinterpret_cast<unsigned char*>(&key[0]) + 16;
  for (int i = 0; i < 8; ++i) {
    p[i] = static_cast<unsigned char>((cuid >> (56 - 8 * i)) & 0xFF);
  }
  char row_buf[16];
  std::snprintf(row_buf, sizeof(row_buf), "%010" PRIu64, row_id);
  std::memcpy(&key[24], row_buf, 10);
  return key;
}

size_t ScanCuid(DB* db, uint64_t cuid) {
  ReadOptions ro;
  std::string upper = GenerateKey(cuid + 1, 0);
  Slice upper_bound(upper);
  ro.iterate_upper_bound = &upper_bound;
  std::unique_ptr<Iterator> iter(db->NewIterator(ro));
  size_t count = 0;
  for (iter->Seek(GenerateKey(cuid, 0)); iter->Valid(); iter->Next()) {
    ++count;
  }
  ExpectStatus(iter->status(), "scan iterator");
  return count;
}

void FlushDb(DB* db) {
  FlushOptions fo;
  fo.wait = true;
  ExpectStatus(db->Flush(fo), "flush");
}

std::set<uint64_t> DeltaFilesFor(HotspotManager* mgr, uint64_t cuid) {
  HotIndexEntry entry;
  Expect(mgr->GetHotIndexEntry(cuid, &entry), "expected hot index entry");
  std::set<uint64_t> files;
  for (const auto& seg : entry.deltas) {
    files.insert(seg.file_number);
  }
  return files;
}

bool SnapshotReady(HotspotManager* mgr, uint64_t cuid) {
  HotIndexEntry entry;
  if (!mgr->GetHotIndexEntry(cuid, &entry)) {
    return false;
  }
  if (!entry.HasSnapshot()) {
    return false;
  }
  for (const auto& seg : entry.snapshot_segments) {
    if (seg.file_number != static_cast<uint64_t>(-1)) {
      return true;
    }
  }
  return false;
}

void TestDeleteCountTableAccounting() {
  GlobalDeleteCountTable table;
  const uint64_t cuid = 77;
  const SequenceNumber delete_seq = 123;

  Expect(table.TrackPhysicalUnit(cuid, 1001), "track memtable 1001");
  Expect(table.TrackPhysicalUnit(cuid, 1002), "track memtable 1002");
  Expect(table.GetRefCount(cuid) == 2, "expected two tracked units");

  table.ApplyFlushChange(cuid, {1001, 1002}, 2001);
  Expect(table.GetRefCount(cuid) == 1,
         "flush should collapse two inputs into one output");

  bool newly_deleted = false;
  Expect(table.MarkDeleted(cuid, delete_seq, &newly_deleted),
         "tracked cuid should be deletable");
  Expect(newly_deleted, "delete should be newly marked");

  table.ApplyCompactionChange(cuid, {2001}, {3001, 3002});
  Expect(table.GetRefCount(cuid) == 2,
         "compaction should expand one input into two outputs");

  table.ApplyCompactionChange(cuid, {3001, 3002}, {});
  Expect(!table.IsTracked(cuid),
         "deleted cuid should disappear after last tracked unit is removed");
}

void TestDeltaLifecycle() {
  const uint64_t cuid = 1001;
  const size_t rows = 64;
  const std::filesystem::path db_path =
      std::filesystem::temp_directory_path() / "rocksdb_delta_functional";
  std::error_code ec;
  std::filesystem::remove_all(db_path, ec);

  Options options;
  options.create_if_missing = true;
  options.enable_delta = true;
  options.compression = kNoCompression;
  options.write_buffer_size = 32 * 1024;
  options.max_write_buffer_number = 4;
  options.max_background_jobs = 2;
  options.max_subcompactions = 1;
  options.num_levels = 2;
  options.level0_file_num_compaction_trigger = 1000;
  options.target_file_size_base = 1 << 20;
  options.target_file_size_multiplier = 1;
  options.delta_options.hotspot_scan_threshold = 2;
  options.delta_options.hotspot_scan_window_sec = 600;
  options.delta_options.hot_data_buffer_threshold_bytes = 1024;
  options.delta_options.compaction_l0_trigger_count = 2;
  options.delta_options.compaction_l0_files_to_pick = 2;
  options.delta_options.compaction_l0_trigger_age_sec = 0;
  options.delta_options.max_delta_threads = 1;
  options.delta_options.gdct_flush_threshold_records = 1;
  options.delta_options.gdct_flush_interval_us = 0;
  options.delta_options.gdct_compact_interval_us = 24ull * 60 * 60 * 1000000;

  DB* raw_db = nullptr;
  ExpectStatus(DB::Open(options, db_path.string(), &raw_db), "open db");
  std::unique_ptr<DB> db(raw_db);
  auto* db_impl = static_cast<DBImpl*>(db.get());
  auto mgr = db_impl->GetHotspotManager();
  Expect(mgr != nullptr, "expected hotspot manager");
  ResetDeltaPerfCounters();

  WriteOptions wo;
  for (size_t row = 0; row < rows; ++row) {
    ExpectStatus(db->Put(wo, GenerateKey(cuid, row), "v0"), "initial put");
  }
  FlushDb(db.get());

  Expect(ScanCuid(db.get(), cuid) == rows, "first scan should read all rows");
  Expect(ScanCuid(db.get(), cuid) == rows, "second scan should read all rows");
  Expect(g_scan_data_rows_captured.load(std::memory_order_relaxed) >= rows * 2,
         "ordinary scans should be captured for delta partial-merge input");
  Expect(g_scan_data_bytes_captured.load(std::memory_order_relaxed) > 0,
         "ordinary scans should account captured delta bytes");
  Expect(mgr->IsHot(cuid), "cuid should become hot after threshold");

  bool snapshot_ready = false;
  for (int attempt = 0; attempt < 50; ++attempt) {
    ExpectStatus(db_impl->TEST_WaitForBackgroundWork(),
                 "wait for init-scan background work");
    if (SnapshotReady(mgr.get(), cuid)) {
      snapshot_ready = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  Expect(snapshot_ready, "expected physical hotspot snapshot after init scan");
  Expect(std::filesystem::exists(db_path / "hotspot_data"),
         "expected hotspot_data directory");
  Expect(!std::filesystem::is_empty(db_path / "hotspot_data"),
         "expected generated hotspot SST files");
  std::string num_files_at_level1;
  Expect(db->GetProperty("rocksdb.num-files-at-level1", &num_files_at_level1),
         "get post-init L1 file count");
  Expect(num_files_at_level1 == "0",
         "delta mode should not create L1 files during background work");

  const int initial_refcount = mgr->GetDeleteTable().GetRefCount(cuid);
  Expect(initial_refcount >= 1,
         "initial full scan should track at least one physical unit");

  for (size_t row = 0; row < rows; ++row) {
    ExpectStatus(db->Put(wo, GenerateKey(cuid, row), "v1"), "delta put v1");
  }
  FlushDb(db.get());

  for (size_t row = 0; row < rows; ++row) {
    ExpectStatus(db->Put(wo, GenerateKey(cuid, row), "v2"), "delta put v2");
  }
  FlushDb(db.get());

  auto old_delta_files = DeltaFilesFor(mgr.get(), cuid);
  Expect(old_delta_files.size() >= 2,
         "expected at least two hot delta files before compaction");

  const int refcount_before_compaction = mgr->GetDeleteTable().GetRefCount(cuid);
  Expect(refcount_before_compaction >= initial_refcount + 2,
         "hot delta flushes should extend GDCT tracking");

  ExpectStatus(db->CompactRange(CompactRangeOptions(), nullptr, nullptr),
               "manual compact range");

  bool compaction_observed = false;
  for (int attempt = 0; attempt < 40; ++attempt) {
    ExpectStatus(db_impl->TEST_WaitForCompact(true), "wait for compaction");
    auto new_delta_files = DeltaFilesFor(mgr.get(), cuid);
    Expect(db->GetProperty("rocksdb.num-files-at-level1", &num_files_at_level1),
           "get L1 file count");
    if (num_files_at_level1 == "0" && new_delta_files != old_delta_files) {
      bool removed_old = false;
      for (uint64_t file : old_delta_files) {
        if (new_delta_files.count(file) == 0) {
          removed_old = true;
          break;
        }
      }
      bool added_new = false;
      for (uint64_t file : new_delta_files) {
        if (old_delta_files.count(file) == 0) {
          added_new = true;
          break;
        }
      }
      if (removed_old && added_new) {
        compaction_observed = true;
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  Expect(compaction_observed,
         "expected delta L0->L0 compaction to rewrite hot delta files");

  const int refcount_after_compaction = mgr->GetDeleteTable().GetRefCount(cuid);
  Expect(refcount_after_compaction == refcount_before_compaction - 1,
         "compaction should merge two tracked inputs into one tracked output");

  for (size_t row = 0; row < rows; ++row) {
    ExpectStatus(db->Delete(wo, GenerateKey(cuid, row)), "delete row");
  }

  const SequenceNumber latest_seq = db->GetLatestSequenceNumber();
  Expect(mgr->GetDeleteTable().IsDeleted(cuid, latest_seq),
         "delete table should mark cuid deleted");
  Expect(ScanCuid(db.get(), cuid) == 0,
         "deleted hot cuid should become invisible immediately");

  db.reset();
  DestroyDB(db_path.string(), Options());
}

}  // namespace

}  // namespace rocksdb

int main() {
  rocksdb::TestDeleteCountTableAccounting();
  rocksdb::TestDeltaLifecycle();
  std::cout << "[PASS] rocksdb delta functional checks" << std::endl;
  return 0;
}
