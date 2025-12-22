#pragma once

// Minimal dlfcn.h shim for Triton's HIP backend on Windows.
// Provides dlopen/dlsym/dlclose/dlerror and RTLD_* flags.

#ifndef _WIN32
#error "This shim is intended for Windows only."
#endif

#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef RTLD_LAZY
#define RTLD_LAZY 0
#endif

#ifndef RTLD_LOCAL
#define RTLD_LOCAL 0
#endif

#ifndef RTLD_NOLOAD
#define RTLD_NOLOAD 0
#endif

static __declspec(thread) char dlerror_buffer[512];
static __declspec(thread) int dlerror_pending = 0;

static inline void _dl_set_error_from_last_error(void) {
  DWORD err = GetLastError();
  if (err == 0) {
    dlerror_buffer[0] = '\0';
    dlerror_pending = 0;
    return;
  }
  DWORD n = FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL,
      err,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      dlerror_buffer,
      (DWORD)sizeof(dlerror_buffer),
      NULL);
  if (n == 0) {
    dlerror_buffer[0] = '\0';
  }
  dlerror_pending = 1;
}

static inline void *dlopen(const char *filename, int flags) {
  (void)flags;
  dlerror_pending = 0;
  dlerror_buffer[0] = '\0';
  HMODULE handle = LoadLibraryA(filename);
  if (!handle) {
    _dl_set_error_from_last_error();
  }
  return (void *)handle;
}

static inline void *dlsym(void *handle, const char *symbol) {
  dlerror_pending = 0;
  dlerror_buffer[0] = '\0';
  FARPROC p = GetProcAddress((HMODULE)handle, symbol);
  if (!p) {
    _dl_set_error_from_last_error();
  }
  return (void *)p;
}

static inline int dlclose(void *handle) {
  dlerror_pending = 0;
  dlerror_buffer[0] = '\0';
  BOOL ok = FreeLibrary((HMODULE)handle);
  if (!ok) {
    _dl_set_error_from_last_error();
    return -1;
  }
  return 0;
}

static inline char *dlerror(void) {
  if (!dlerror_pending) {
    return NULL;
  }
  dlerror_pending = 0;
  return dlerror_buffer;
}

#ifdef __cplusplus
} // extern "C"
#endif
