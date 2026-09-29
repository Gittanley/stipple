// SPDX-License-Identifier: GPL-3.0-or-later
#include "rd_checkpoint.h"

#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace rd {
namespace {

std::string Trim(const std::string& s) {
  std::size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  std::size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

std::string DirName(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  if (slash == std::string::npos) return ".";
  if (slash == 0) return path.substr(0, 1);
  return path.substr(0, slash);
}

// Flushes a directory entry so a rename inside it survives a power cut.  On
// Windows this means opening the directory with backup semantics and flushing it;
// a failure here is not fatal, because MOVEFILE_WRITE_THROUGH has already put the
// rename on stable storage.
void FlushDirectory(const std::string& dir) {
  HANDLE h = CreateFileA(dir.c_str(), GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, OPEN_EXISTING,
                         FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_WRITE_THROUGH,
                         nullptr);
  if (h == INVALID_HANDLE_VALUE) return;
  (void)FlushFileBuffers(h);
  (void)CloseHandle(h);
}

}  // namespace

void FileIdentity(const std::string& path, std::uint64_t* bytes,
                  std::int64_t* mtime) {
  *bytes = 0;
  *mtime = 0;
  WIN32_FILE_ATTRIBUTE_DATA data{};
  if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &data)) return;
  *bytes = (static_cast<std::uint64_t>(data.nFileSizeHigh) << 32) |
           data.nFileSizeLow;
  *mtime = (static_cast<std::int64_t>(data.ftLastWriteTime.dwHighDateTime) << 32) |
           data.ftLastWriteTime.dwLowDateTime;
}

std::uint64_t FileSize(const std::string& path) {
  std::uint64_t bytes = 0;
  std::int64_t mtime = 0;
  FileIdentity(path, &bytes, &mtime);
  return bytes;
}

bool CheckpointSave(const std::string& path, const Checkpoint& cp,
                    std::string* error) {
  const std::string tmp = path + ".tmp";
  {
    std::ostringstream out;
    out << "rdither-checkpoint 1\n";
    out << "input " << cp.input << "\n";
    out << "input_bytes " << cp.input_bytes << "\n";
    out << "input_mtime " << cp.input_mtime << "\n";
    out << "geometry " << cp.width << " " << cp.height << "\n";
    out << "total_frames " << cp.total_frames << "\n";
    out << "colors " << cp.colors << "\n";
    out << "block " << cp.block << "\n";
    out << "segment_frames " << cp.segment_frames << "\n";
    out << "palette " << cp.palette_path << "\n";
    out << "segment_count " << cp.segments.size() << "\n";
    for (const SegmentRecord& s : cp.segments) {
      out << "segment " << s.index << " " << s.first_frame << " " << s.frames
          << " " << s.bytes << " " << s.path << "\n";
    }

    // 1. write the whole record to a temp file
    HANDLE h = CreateFileA(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
      *error = "cannot create " + tmp;
      return false;
    }
    const std::string text = out.str();
    DWORD written = 0;
    const BOOL ok = WriteFile(h, text.data(),
                              static_cast<DWORD>(text.size()), &written, nullptr);
    // 2. push it to stable storage before it is allowed to be the real record
    const BOOL flushed = FlushFileBuffers(h);
    (void)CloseHandle(h);
    if (ok == 0 || flushed == 0 || written != text.size()) {
      *error = "short or unflushed write to " + tmp;
      (void)DeleteFileA(tmp.c_str());
      return false;
    }
  }
  // 3. durable rename: does not return until the move itself is on the platter
  if (MoveFileExA(tmp.c_str(), path.c_str(),
                  MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
    *error = "cannot move " + tmp + " into place";
    (void)DeleteFileA(tmp.c_str());
    return false;
  }
  // 4. and make the directory entry itself durable
  FlushDirectory(DirName(path));
  return true;
}

bool CheckpointLoad(const std::string& path, Checkpoint* cp,
                    std::string* error) {
  std::ifstream in(path.c_str(), std::ios::binary);
  if (!in) {
    *error = "no checkpoint at " + path;
    return false;
  }
  std::string line;
  if (!std::getline(in, line) || line.rfind("rdither-checkpoint 1", 0) != 0) {
    *error = "unrecognised checkpoint format at " + path;
    return false;
  }
  cp->segments.clear();
  std::size_t declared_segments = 0;
  bool have_segments = false;
  while (std::getline(in, line)) {
    line = Trim(line);
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key == "input") {
      std::getline(ls, cp->input);
      cp->input = Trim(cp->input);
    } else if (key == "input_bytes") {
      ls >> cp->input_bytes;
    } else if (key == "input_mtime") {
      ls >> cp->input_mtime;
    } else if (key == "geometry") {
      ls >> cp->width >> cp->height;
    } else if (key == "total_frames") {
      ls >> cp->total_frames;
    } else if (key == "colors") {
      ls >> cp->colors;
    } else if (key == "block") {
      ls >> cp->block;
    } else if (key == "segment_frames") {
      ls >> cp->segment_frames;
    } else if (key == "palette") {
      std::getline(ls, cp->palette_path);
      cp->palette_path = Trim(cp->palette_path);
    } else if (key == "segment_count") {
      ls >> declared_segments;
      have_segments = true;
    } else if (key == "segment") {
      SegmentRecord s;
      s.path.clear();
      if (!(ls >> s.index >> s.first_frame >> s.frames >> s.bytes)) continue;
      std::getline(ls, s.path);
      s.path = Trim(s.path);
      cp->segments.push_back(s);
    }
  }
  if (!have_segments) {
    *error = "checkpoint has no segment count: " + path;
    return false;
  }
  // A truncated record is caught here rather than being half-believed.  This is
  // the parse-side half of "verification": the durable write makes truncation
  // unlikely, and this makes it harmless if it happens anyway.
  if (cp->segments.size() != declared_segments) {
    *error = "checkpoint segment count disagrees with the records present";
    return false;
  }
  return true;
}

bool CheckpointValidate(Checkpoint* cp, const std::string& input,
                        const std::int64_t total_frames, int colors, int block,
                        int segment_frames, std::vector<int>* dropped,
                        std::string* mismatch) {
  dropped->clear();
  mismatch->clear();
  if (cp->input != input) {
    *mismatch = "the checkpoint is for a different input file (" + cp->input + ")";
    return false;
  }
  if (cp->total_frames != total_frames) {
    *mismatch = "the input now reports a different frame count";
    return false;
  }
  if (cp->colors != colors || cp->block != block ||
      cp->segment_frames != segment_frames) {
    *mismatch = "the checkpoint was written with different settings "
                "(colors / --blocks / --segment-frames)";
    return false;
  }
  // Re-verify every recorded segment against the filesystem.  A segment counts as
  // done only if its file is present and exactly the size it was when recorded.
  std::vector<SegmentRecord> keep;
  keep.reserve(cp->segments.size());
  for (const SegmentRecord& s : cp->segments) {
    const std::uint64_t now = FileSize(s.path);
    if (now == 0) {
      dropped->push_back(s.index);
      continue;
    }
    if (now != s.bytes) {
      // Present but the wrong length: either a truncated write or a different
      // file.  Either way it is not the segment that was checkpointed.
      dropped->push_back(s.index);
      continue;
    }
    keep.push_back(s);
  }
  cp->segments.swap(keep);
  return true;
}

std::string CheckpointPathFor(const std::string& out_path) {
  return out_path + ".rdcheck";
}

std::string SegmentPathFor(const std::string& out_path, int index) {
  char tag[16];
  std::snprintf(tag, sizeof(tag), ".part%04d", index);
  const std::size_t dot = out_path.find_last_of('.');
  const std::size_t slash = out_path.find_last_of("/\\");
  // Only treat the extension as such if it is after the last separator, so
  // "dir.v2/out" is not read as extension ".v2/out".
  if (dot != std::string::npos &&
      (slash == std::string::npos || dot > slash)) {
    return out_path.substr(0, dot) + tag + out_path.substr(dot);
  }
  return out_path + tag;
}

}  // namespace rd
