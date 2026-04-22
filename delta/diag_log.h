//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

// Delta diagnostic logger: writes key diagnostic lines to BOTH stderr and a
// persistent log file so that important events survive terminal scroll-off.
//
// Usage:
//   DiagLogOpen("/path/to/db/delta_diag.log");   // call once at startup
//   DiagLogf("[DIAG_FOO] some %s message\n", "important");

#pragma once

#ifdef __GNUC__
#define DIAG_PRINTF_ATTR __attribute__((format(printf, 1, 2)))
#else
#define DIAG_PRINTF_ATTR
#endif

// Open (or append to) the diagnostic log file.
// Safe to call multiple times; reopens the file each time.
void DiagLogOpen(const char* path);

// Printf to stderr AND the diag log file (if open via DiagLogOpen).
// The format string must include a trailing newline as needed.
// Note: stderr output is gated by DiagLogSetStderrEnabled(); file is always written.
void DiagLogf(const char* fmt, ...) DIAG_PRINTF_ATTR;

// Enable or disable DiagLogf output entirely.
// When disabled, neither stderr nor the log file receives any output.
// Call with true when ReadOptions::enable_delta_diag_logging is set.
void DiagLogSetStderrEnabled(bool enabled);

// ---------------------------------------------------------------------------
// HotSST lifecycle log: writes ONLY to file (not stderr) to avoid noise.
// Records Register/Ref/Unref events for each tracked SST file number.
// ---------------------------------------------------------------------------

// Open (or append to) the lifecycle log file.
void LifecycleLogOpen(const char* path);

// Printf to the lifecycle log file only (not stderr).
void LifecycleLogf(const char* fmt, ...) DIAG_PRINTF_ATTR;
