// rd_checkpoint.h -- crash-safe progress record, so a render interrupted by a
// power cut can be finished instead of restarted.
//
// Why a log file and not a resume marker
// -------------------------------------
// Power loss is not process death.  When the power goes the OS page cache goes
// with it, so a record that was merely written or flushed still lives only in
// RAM.  A checkpoint is only worth having if it is on the platter, which means
// FlushFileBuffers (fsync) -- and a record updated in place can be torn by the
// cut and then parse as garbage, or worse, parse as *valid* but wrong.  So:
//
//   1. the whole record is written to a temporary file,
//   2. that file is flushed to stable storage,
//   3. it is moved into place with MOVEFILE_WRITE_THROUGH, which does not return
//      until the rename itself is durable,
//   4. the containing directory is flushed.
//
// A reader therefore only ever sees a complete, older-or-newer record, never a
// partial one.  That is the "verification" part: on resume every recorded
// segment is re-checked against the filesystem, so a segment whose file is
// missing or short is redone even though the record claims it finished.
//
// Granularity is a segment, not a frame.  The output is written as a series of
// independently-finalised chunks precisely so that a crash costs at most one
// chunk: a single long output cannot be resumed safely, because a half-written
// container is not a valid prefix of anything.
#ifndef RD_CHECKPOINT_H_
#define RD_CHECKPOINT_H_

#include <cstdint>
#include <string>
#include <vector>

namespace rd {

// One finished piece of the output.  `bytes` is the size the file had when the
// segment was recorded, and is re-verified on resume: a file that no longer
// matches means the segment is not trustworthy and gets redone.
struct SegmentRecord {
  int index = 0;
  std::int64_t first_frame = 0;
  std::int64_t frames = 0;
  std::string path;
  std::uint64_t bytes = 0;
};

struct Checkpoint {
  int version = 1;
  // Identity of the job.  A checkpoint is only reusable for the same input,
  // geometry, palette and settings; a mismatch is reported, not silently used.
  std::string input;
  std::uint64_t input_bytes = 0;
  std::int64_t input_mtime = 0;
  int width = 0;
  int height = 0;
  std::int64_t total_frames = 0;
  int colors = 0;
  int block = 0;
  int segment_frames = 0;
  std::string palette_path;  // --palette-export output, reused on resume
  std::vector<SegmentRecord> segments;
};

// Identifies a file well enough to notice that it changed.  Size and mtime
// together, because a same-size overwrite is a real case.
void FileIdentity(const std::string& path, std::uint64_t* bytes,
                  std::int64_t* mtime);

std::uint64_t FileSize(const std::string& path);

// Writes `cp` so that it survives an immediate power cut: temp file, flush,
// durable rename.  Returns false with a message on failure.
bool CheckpointSave(const std::string& path, const Checkpoint& cp,
                    std::string* error);

// Reads a checkpoint.  False means "absent or unreadable", which is not an error
// condition -- it just means start from the beginning.
bool CheckpointLoad(const std::string& path, Checkpoint* cp, std::string* error);

// Checks the record against the job it is being resumed into, and against the
// filesystem.  Segments that fail verification are dropped from `cp` and listed in
// `dropped`, so the caller can say what it is redoing.  Returns false only for a
// mismatch that makes resuming meaningless (different input, settings, geometry).
bool CheckpointValidate(Checkpoint* cp, const std::string& input,
                        const std::int64_t total_frames, int colors, int block,
                        int segment_frames, std::vector<int>* dropped,
                        std::string* mismatch);

// Default checkpoint path for an output: "out.mkv" -> "out.mkv.rdcheck".
std::string CheckpointPathFor(const std::string& out_path);

// Segment path for output "out.mkv", segment 3: "out.mkv.part0003.mkv".
std::string SegmentPathFor(const std::string& out_path, int index);

}  // namespace rd

#endif  // RD_CHECKPOINT_H_
