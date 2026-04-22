#pragma once

#include <string>

#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

inline uint64_t ExtractCUID(const Slice& key) {
  if (key.size() < 24) return 0;
  const unsigned char* p =
      reinterpret_cast<const unsigned char*>(key.data()) + 16;
  uint64_t c = 0;
  for (int i = 0; i < 8; ++i) {
    c = (c << 8) | p[i];
  }
  return c;
}

inline std::string FormatKeyDisplay(const Slice& key) {
  std::string cuid_part =
      std::to_string(key.size() >= 24 ? ExtractCUID(key) : 0);
  std::string suffix = key.size() > 24 ? key.ToString().substr(24) : "";
  return cuid_part + "..." + suffix;
}

inline uint64_t ExtractRowID(const Slice& key) {
  if (key.size() < 34) return 0;
  // Suffix starts at 24, Row ID is at 24-33
  std::string rid_str = key.ToString().substr(24, 10);
  try {
    return std::stoull(rid_str);
  } catch (...) {
    return 0;
  }
}

}  // namespace ROCKSDB_NAMESPACE