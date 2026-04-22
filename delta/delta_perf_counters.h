#pragma once
#include <atomic>
#include <cstdint>

// Global path-level performance counters for Delta scan analysis.
// C++20 inline atomics: each TU shares the same instance without a .cc file.

namespace ROCKSDB_NAMESPACE {

// --- scan_data collection (db_iter.cc) ---
// Rows/bytes appended to delta_ctx_.scan_data during partial scan.
// If large: partial-merge input collection is the hot path.
inline std::atomic<uint64_t> g_scan_data_rows_captured{0};
inline std::atomic<uint64_t> g_scan_data_bytes_captured{0};

// --- Hot-path routing (DeltaSwitchingIterator::Seek) ---
// Seek calls where IsHot=true (not skip_hot_path mode).
inline std::atomic<uint64_t> g_hot_path_seek_attempts{0};
// Seek calls where InitHotIter() succeeded (snapshot already ready).
inline std::atomic<uint64_t> g_hot_path_seek_success{0};
// Seek calls where IsHot=true but InitHotIter() failed (warming / no snapshot).
inline std::atomic<uint64_t> g_hot_path_fallbacks{0};

// --- HotDataBuffer::NewIterator() (hot_data_buffer.cc) ---
// Number of NewIterator() calls (one per -1 memory segment accessed).
inline std::atomic<uint64_t> g_buffer_iterator_build_count{0};
// Total rows copied into filtered_entries across all NewIterator() calls.
inline std::atomic<uint64_t> g_buffer_iterator_rows_materialized{0};

inline void ResetDeltaPerfCounters() {
  g_scan_data_rows_captured.store(0, std::memory_order_relaxed);
  g_scan_data_bytes_captured.store(0, std::memory_order_relaxed);
  g_hot_path_seek_attempts.store(0, std::memory_order_relaxed);
  g_hot_path_seek_success.store(0, std::memory_order_relaxed);
  g_hot_path_fallbacks.store(0, std::memory_order_relaxed);
  g_buffer_iterator_build_count.store(0, std::memory_order_relaxed);
  g_buffer_iterator_rows_materialized.store(0, std::memory_order_relaxed);
}

}  // namespace ROCKSDB_NAMESPACE
