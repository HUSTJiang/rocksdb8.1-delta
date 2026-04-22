//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "delta/diag_log.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>

static FILE* g_diag_file = nullptr;
static std::mutex g_diag_mutex;
static std::atomic<bool> g_diag_stderr_enabled{false};

void DiagLogSetStderrEnabled(bool enabled) {
  g_diag_stderr_enabled.store(enabled, std::memory_order_relaxed);
}

void DiagLogOpen(const char* path) {
  std::lock_guard<std::mutex> lk(g_diag_mutex);
  if (g_diag_file) {
    fclose(g_diag_file);
    g_diag_file = nullptr;
  }
  g_diag_file = fopen(path, "a");  // append so multiple runs accumulate
}

void DiagLogf(const char* fmt, ...) {
  char buf[8192];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);

  if (g_diag_stderr_enabled.load(std::memory_order_relaxed)) {
    // fputs(buf, stderr);
  }

  std::lock_guard<std::mutex> lk(g_diag_mutex);
  if (g_diag_file) {
    fputs(buf, g_diag_file);
    fflush(g_diag_file);  // flush immediately so crash doesn't lose data
  }
}

// ---------------------------------------------------------------------------
// HotSST lifecycle log: writes ONLY to file (not stderr)
// ---------------------------------------------------------------------------

static FILE* g_lifecycle_file = nullptr;
static std::mutex g_lifecycle_mutex;

void LifecycleLogOpen(const char* path) {
  std::lock_guard<std::mutex> lk(g_lifecycle_mutex);
  if (g_lifecycle_file) {
    fclose(g_lifecycle_file);
    g_lifecycle_file = nullptr;
  }
  g_lifecycle_file = fopen(path, "a");
}

void LifecycleLogf(const char* fmt, ...) {
  std::lock_guard<std::mutex> lk(g_lifecycle_mutex);
  if (!g_lifecycle_file) return;
  char buf[4096];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  fputs(buf, g_lifecycle_file);
  fflush(g_lifecycle_file);
}
