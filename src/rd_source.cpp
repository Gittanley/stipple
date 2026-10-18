// SPDX-License-Identifier: GPL-3.0-or-later
// rd_source.cpp -- RAM-first allocation with a memory-mapped disk fallback.
#include "rd_source.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace rd {
namespace {

// A scratch file that deletes itself when the handle closes.
//
// The handle is opened FILE_FLAG_DELETE_ON_CLOSE, so the name is released when the
// LAST handle to the file goes away -- which is not this destructor.  PixelStore
// keeps a second handle open for the file-mapping object, so the delete is deferred
// to ~PixelStore, which is what we want: the bytes stay readable for the whole
// lifetime of the mapping.  `spill_path_` may therefore name a file that does not
// exist yet, and nothing may assume otherwise.
//
// The explicit DeleteFileA is redundant on that path and is kept for the failure
// returns in Create that construct no PixelStore: there ~SpillFile is the only
// destructor that runs.
struct SpillFile {
#ifdef _WIN32
  HANDLE handle = INVALID_HANDLE_VALUE;
#else
  int fd = -1;
#endif
  std::string path;

  ~SpillFile() {
#ifdef _WIN32
    if (handle != INVALID_HANDLE_VALUE) ::CloseHandle(handle);
    if (!path.empty()) ::DeleteFileA(path.c_str());
#else
    // The name was already unlinked in MakeSpillFile, so closing this fd is
    // the whole of the cleanup that is left; the unlink stays to mirror the
    // Windows branch's redundant DeleteFileA for the failure returns that
    // construct no PixelStore (it reports nothing, same as there).
    if (fd >= 0) (void)::close(fd);
    if (!path.empty()) (void)::unlink(path.c_str());
#endif
  }
};

bool MakeSpillFile(SpillFile* out, std::string* error) {
#ifdef _WIN32
  char temp_dir[MAX_PATH + 1] = {0};
  DWORD n = ::GetTempPathA(MAX_PATH, temp_dir);
  // MAX_PATH bounds the BUFFER, not the prefix.  GetTempPathA returns the length
  // excluding the NUL, so a temp directory using all MAX_PATH characters passes
  // `n > MAX_PATH` while leaving no room for what GetTempFileNameA appends:
  // an optional separator, the 3-char prefix, 4 random characters, ".tmp" and a
  // NUL -- 13 bytes.  That wrote up to 271 bytes into the 261-byte `name` below.
  // Only a long %TEMP% (deep profile paths, a redirected TEMP) reaches it.
  constexpr std::size_t kTempFileNameSuffix = 1 + 3 + 4 + 4 + 1;
  if (n == 0 || static_cast<std::size_t>(n) + kTempFileNameSuffix > MAX_PATH) {
    *error = "the temporary directory path is too long for a spill file";
    return false;
  }
  char name[MAX_PATH + 1] = {0};
  if (::GetTempFileNameA(temp_dir, "rdx", 0, name) == 0) {
    *error = "GetTempFileName failed";
    return false;
  }
  out->path = name;
  // FILE_SHARE_DELETE, so the name can be marked delete-on-close below rather than
  // relying on a destructor that runs while the mapping still holds a section.
  out->handle = ::CreateFileA(name, GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                              CREATE_ALWAYS,
                              FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
                              nullptr);
  if (out->handle == INVALID_HANDLE_VALUE) {
    *error = "CreateFile failed for spill file";
    return false;
  }
#else
  // mkstemp creates and opens in one step, and the unlink right after makes
  // the Windows FILE_FLAG_DELETE_ON_CLOSE guarantee stronger rather than
  // weaker: the name never exists long enough for another process to open,
  // while the bytes live as long as this fd -- and then, after the fd closes,
  // as long as any mmap still held, which is the property PixelStore::Create
  // leans on when this destructor runs at the end of that function.
  const char* tmpdir = std::getenv("TMPDIR");
  if (tmpdir == nullptr || tmpdir[0] == '\0') tmpdir = "/tmp";
  std::string tmpl = std::string(tmpdir) + "/rdxXXXXXX";
  // std::string is contiguous and NUL-terminated (C++17), so &tmpl[0] is a
  // mutable buffer of size()+1 that mkstemp can rewrite in place: six X's
  // replaced, the existing NUL at the end never moved.
  const int fd = ::mkstemp(&tmpl[0]);
  if (fd < 0) {
    *error = "mkstemp failed for spill file";
    return false;
  }
  out->path = tmpl;
  (void)::unlink(tmpl.c_str());
  out->fd = fd;
#endif
  return true;
}

}  // namespace

PixelStore* PixelStore::Create(std::size_t width, std::size_t height,
                               std::size_t max_ram_bytes,
                               std::string* error) {
  if (width == 0 || height == 0) {
    *error = "image has zero extent";
    return nullptr;
  }
  PixelStore* store = new (std::nothrow) PixelStore();
  if (store == nullptr) {
    *error = "out of memory";
    return nullptr;
  }
  store->width_ = width;
  store->height_ = height;
  const std::size_t bytes = width * height * sizeof(RgbaF);

  // ---- RAM path -----------------------------------------------------------
  if (bytes <= max_ram_bytes) {
    // Page-locked when CUDA is present.  The blocks engine's upload and download
    // then DMA straight from and into this buffer, instead of a 16 B/px staging
    // memcpy in and the driver's bounce buffer out -- 32 B/px of host DRAM
    // traffic per frame that buys nothing.  CudaAllocPinned returns nullptr in a
    // --no-cuda build, so this falls through unchanged.
    if (void* pin = ::rd::CudaAllocPinned(bytes)) {
      store->data_ = static_cast<RgbaF*>(pin);
      store->pinned_ = true;
      return store;
    }
#ifdef _WIN32
    void* mem = ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT,
                               PAGE_READWRITE);
#else
    // The heap counterpart of VirtualAlloc for this branch: the size was just
    // checked against max_ram_bytes, so this is ordinary allocation, and glibc
    // returns memory aligned to at least 16 bytes -- what RgbaF (16 B) needs.
    // nullptr falls through to the disk path, exactly as VirtualAlloc failing
    // did; the dtor's non-mapped, non-pinned branch frees it below.
    void* mem = std::malloc(bytes);
#endif
    if (mem != nullptr) {
      store->data_ = static_cast<RgbaF*>(mem);
      return store;
    }
    // Fall through to the disk path if the heap commit failed.
  }

  // ---- Disk path ----------------------------------------------------------
  SpillFile spill;
  if (!MakeSpillFile(&spill, error)) {
    delete store;
    return nullptr;
  }
#ifdef _WIN32
  // Extend the file to the full mapping size.
  LARGE_INTEGER file_size;
  file_size.QuadPart = static_cast<LONGLONG>(bytes);
  if (!::SetFilePointerEx(spill.handle, file_size, nullptr, FILE_BEGIN) ||
      !::SetEndOfFile(spill.handle)) {
    *error = "could not size the spill file";
    delete store;
    return nullptr;
  }
  HANDLE mapping = ::CreateFileMappingA(spill.handle, nullptr, PAGE_READWRITE,
                                        0, 0, nullptr);
  if (mapping == nullptr) {
    *error = "CreateFileMapping failed";
    delete store;
    return nullptr;
  }
  void* view = ::MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, bytes);
  if (view == nullptr) {
    ::CloseHandle(mapping);
    *error = "MapViewOfFile failed";
    delete store;
    return nullptr;
  }
  store->data_ = static_cast<RgbaF*>(view);
  store->mapped_ = true;
  store->mapping_ = mapping;
  store->mapping_size_ = bytes;
  store->spill_path_ = spill.path;
  return store;
#else
  // ftruncate is the SetFilePointerEx+SetEndOfFile pair; MAP_SHARED is
  // CreateFileMapping+MapViewOfFile.  There is no separate mapping object to
  // hold on POSIX -- the mapping is `view` itself -- which is why mapping_
  // stays nullptr here: nothing will CloseHandle it, and ~PixelStore munmaps
  // using data_ and mapping_size_ alone.
  if (::ftruncate(spill.fd, static_cast<off_t>(bytes)) != 0) {
    *error = "could not size the spill file";
    delete store;
    return nullptr;
  }
  void* view = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                      spill.fd, 0);
  if (view == MAP_FAILED) {
    *error = "mmap failed for spill file";
    delete store;
    return nullptr;
  }
  // The mapping outlives this scope's fd on purpose: ~SpillFile closes the fd
  // at the end of this function and the pages remain valid -- mmap holds its
  // own reference to the file.  Unlinked at creation, so there is no name left
  // for anyone to unlink underneath it either.
  store->data_ = static_cast<RgbaF*>(view);
  store->mapped_ = true;
  store->mapping_ = nullptr;
  store->mapping_size_ = bytes;
  store->spill_path_ = spill.path;
  return store;
#endif
}

PixelStore::~PixelStore() {
  if (data_ == nullptr) return;
  if (mapped_) {
#ifdef _WIN32
    ::UnmapViewOfFile(data_);
    if (mapping_ != nullptr) ::CloseHandle(static_cast<HANDLE>(mapping_));
#else
    // No section handle exists on this platform (see Create): munmap through
    // the recorded size is the complete release, and the spill fd was already
    // closed by ~SpillFile back in Create.
    (void)::munmap(data_, mapping_size_);
#endif
  } else if (pinned_) {
    ::rd::CudaFreePinned(data_);
  } else {
#ifdef _WIN32
    ::VirtualFree(data_, 0, MEM_RELEASE);
#else
    // Matches the allocation branch: malloc on the RAM path.
    std::free(data_);
#endif
  }
}

bool PixelStore::Prefetch() {
  if (data_ == nullptr) return false;
  const std::size_t bytes = byte_size();
  if (!mapped_) return true;  // already resident
  // Touch one byte per 4 KiB page to pull the mapping into the page cache.
  unsigned char* p = reinterpret_cast<unsigned char*>(data_);
  for (std::size_t off = 0; off < bytes; off += 4096) p[off] = p[off];
  return true;
}

}  // namespace rd
