// rd_video.h -- Phase 2: video decode -> GPU dither -> encode, driven by ffmpeg.
//
// Why ffmpeg owns decode and encode
// ---------------------------------
// ImageMagick reads a video one frame at a time through its ffmpeg delegate,
// materialising a full Image (pixel cache, property bags, colormap machinery) per
// frame and going through the filesystem for anything intermediate.  For a
// 1920x1080 sequence that is the dominant cost and it is all overhead.
//
// Instead the pipeline is three processes:
//
//     ffmpeg -i in  -f rawvideo -pix_fmt rgba64le -   ->  our pipe
//     rdither  (GPU block engine, one launch per batch)
//                                                          ->  our pipe
//     ffmpeg -f rawvideo -pix_fmt rgba64le -s WxH -r F -i -  out
//
// rgba64le is chosen deliberately: it is 16 bits per channel little endian, which
// maps one-to-one onto ImageMagick's Q16 Quantum (0..65535) with no scaling, so
// the dither sees the values a 16-bit decode would produce.  Note that ffmpeg and
// ImageMagick may differ by a least significant bit in YUV->RGB conversion; that
// is a decoder difference, not a dither difference, and it applies equally to the
// reference pipeline.
//
// Frames are handed to the GPU in batches because the block engine's parallelism
// *is* the batch: each (frame, block) pair is an independent walk.
#pragma once

#include <cstdint>
#include <string>

#include "rd_im.h"
#include "rd_riemersma.h"
#include "rd_types.h"

namespace rd {

struct VideoInfo {
  int width = 0;
  int height = 0;
  // The display matrix rotation in degrees, from `side_data=rotation`.
  //
  // ffmpeg's rawvideo output AUTO-ROTATES by default, so a stream carrying a 90 or
  // 270 degree display matrix is emitted at the DISPLAYED dimensions -- width and
  // height swapped -- while `width`/`height` above are the CODED ones.  The pixel
  // count is identical either way, so a mismatch is invisible to every bounds check
  // and shows up only as a scrambled picture: the frame is read back at the wrong
  // row width.  Measured on a phone capture tagged -90: 1920x1080 coded, 1080x1920
  // emitted, and the output was the portrait image written into a landscape buffer.
  //
  // So this is not advisory.  VideoRun reads it once and uses the displayed geometry
  // for every buffer, the raw pipe's -s, and the encoder.
  int rotation = 0;
  std::int64_t frames = 0;   // 0 when the container does not say
  double fps = 0.0;
  std::string pix_fmt;
  std::string codec;
};

struct VideoOptions {
  int colors = 16;
  // Curve positions per block (>= 16).  This value is DEAD: the driver unconditionally
  // overwrites it with BlockOptions::block (rd_cli.cpp, `opt.video_opt.block =
  // opt.blocks.block`), so --blocks is the only thing that sets the block length and
  // this initialiser never survives.  It is kept, at the same value, so that a caller
  // constructing VideoOptions directly gets the fast setting rather than a
  // surprisingly slow one -- and it is fixed whenever the assignment above is removed.
  int block = 512;
  // Frames sampled for the palette.  0 (the default) means "as many as
  // palette_budget_ms allows", which on a 3-hour clip is a few hundred rather than the
  // 30 that matched the reference pipeline.  A fixed count is the wrong shape for this
  // problem: the failure being defended against is *all* the samples landing in the
  // dull stretches of a long clip, and coverage grows with the clip's length.  A
  // positive value requests exactly that many samples, for reproducing the reference.
  //
  // Note this moves the default away from the reference pipeline's 30.  It does not
  // make the palette *worse* -- more coverage is strictly better conditioned -- but it
  // does mean the default output is no longer bit-identical to `magick f1..f30
  // +append -colors 16`.  Pass --palette-frames 30 to get that back.  The reason is in
  // the README under "Palette from more frames, within a time budget".
  int palette_frames = 0;
  // Time budget for the palette stage when palette_frames is 0.  The cost is one
  // ffmpeg spawn plus one seek per sample, 6 at a time, measured at 0.12 s per sample,
  // so 60 s buys roughly 500 samples.  A budget rather than a count because a slow disk
  // should take fewer samples, not take longer.
  int palette_budget_ms = 60000;
  // Lattice-sample tile edge per frame; 0 means full resolution.  128 rather than
  // full resolution because the cost difference is 36x (1.4 s versus 49.2 s for 30
  // frames at 1080p) and the palette is measurably no worse: 84.9% versus 84.7%
  // mean saturation with --im-palette, and 42.2% either way without it.  A full
  // resolution montage is 949 MiB and buys nothing.
  int palette_tile = 128;
  int batch_frames = 16;      // frames per work item
  double diffusion = 1.0;
  // Worker pool.  cpu_threads: -1 = auto (all cores minus the reader and
  // writer), 0 = no host workers, >0 = exact count.  use_gpu adds the CUDA
  // engine as one more worker.
  int cpu_threads = -1;
  bool use_gpu = true;
  // Which engine the GPU workers use.  kBlocks is the CUDA block-parallel walk;
  // kOpenCL is the same partition and the same arithmetic reached through OpenCL,
  // so a frame dithered on either is the same frame and the two are comparable
  // rather than merely similar.  kOpenCL refuses the planar-YUV input modes and
  // planar 4:4:4 output -- see docs/OPENCL.md -- so with those selected it
  // reports an error instead of silently decoding something else.
  enum class GpuEngine { kBlocks, kOpenCL };
  GpuEngine gpu_engine = GpuEngine::kBlocks;
  // Variable frame rate.  The DEFAULT behaviour is already correct on duration and
  // A/V sync: the source's frame timings are measured, and a variable-rate source is
  // re-emitted at its true average rather than at r_frame_rate (which is the
  // maximum instantaneous rate, and made a 3.03 s source come out 1.63 s long).
  //
  // What the default does NOT do is preserve the per-frame cadence: the output is
  // constant rate at the true average.  That is the right trade for a ditherer --
  // the picture is right and the clock is right -- but it does resample motion the
  // way a CFR pre-render does, just without the extra pass.
  //
  // `preserve_vfr` asks for the per-frame timing to be restored at the encoder with
  // a piecewise setpts expression.  **NOT VERIFIED END TO END.**  The mechanism is
  // proven in isolation (102 frames through a rawvideo pipe come out with PTS
  // running 0 -> 3.017 s against a 3.0333 s source), but no genuinely
  // variable-timestamp fixture could be built on this machine to confirm it through
  // the full pipeline, and shipping an unverified path as the default is how the
  // next person loses a day.  Off by default; see docs/OPENCL.md for the three
  // things that have to be right and the fourth that is still open.
  bool preserve_vfr = false;
  // Independent GPU workers, each with its own device state, so two dither launches
  // are in flight at once and one could download while the other computes.
  // Measured: 2 is *not* faster than 1 (27.8-28.1 fps either way, and dither
  // worker-time rises from ~18.9 s to ~19.4 s), because the dither stage is already
  // saturated rather than transfer-bound.  The default is therefore 1, which also
  // avoids paying for a second copy of the per-batch device buffers.  The pool and
  // this knob exist because the negative result is worth being able to re-check on
  // a different GPU, not because 2 is wanted here.
  int gpu_workers = 1;
  // What the decoder emits, and how the gather reads it.
  //   "yuv444"          3 bytes/px planar 4:4:4, converted to RGB on the device
  //   "yuv444-prepass"  same result, via a coalesced pass and then the gather
  //   "rgba64"          8 bytes/px, widened with a cast
  //
  // yuv444 is the default.  The colour matrix is swscale's own, ported exactly
  // (see d_sws_yuv_to_rgb16), so for a 4:4:4 source the two modes are
  // bit-identical end to end -- verified over 605 frames -- while moving 2.7x
  // less across the pipe and the H2D.  On a 4:2:0 source they are not identical:
  // letting ffmpeg reconstruct 4:4:4 and then converting applies a different
  // chroma reconstruction than swscale's fused 4:2:0 -> RGBA path, which moves
  // 0.67% of output components (34.3 dB) once the dither has amplified the
  // difference through its own error feedback.  "--input-mode rgba64" selects the
  // old path and remains the reference to compare against.
  //
  // The fused kernel is the default over the prepass because the prepass allocates
  // a second device buffer of 6 bytes/px per queue slot, which the RAM budget does
  // not account for and which would be ~2.4 GB at 4K on a 4 GB card.  It is
  // measured ~2% faster where it fits, and remains available by name.
  std::string input_mode = "yuv444";
  // Use the GPU's hardware H.264/HEVC decoder (NVDEC) instead of software.
  //
  // H.264 decoding is *normative* -- the spec fixes the reconstruction process -- so a
  // conformant hardware decoder must produce bit-identical YUV to software.  Verified
  // rather than assumed: the first 10 frames of the bench clip decoded both ways and
  // written as rgba64le hash identically (MD5 1FD40474DE85816DECB2CBFCEF58F5).
  //
  // ffmpeg still does the swscale colour conversion afterwards, so the RGB the dither
  // sees is unchanged.  Measured 541 fps decode against ~120 for software, on a
  // 6-core Haswell where the software decoder is competing with everything else.
  //
  // The one caveat: on a *corrupt* stream, error concealment is implementation-defined
  // and hardware and software can pick different concealment.  Clean input is
  // unaffected.  Set 0 to force software.
  int hwaccel = 1;
  // ffmpeg thread caps.  0 = automatic (ffmpeg's own choice, which on this machine
  // means every logical processor).  Two ffmpeg processes each taking all of them
  // oversubscribes 6 physical cores roughly threefold, which is why no stage looks
  // saturated while the wall clock stays high.  Split the budget between them.
  int decode_threads = 0;
  int encode_threads = 0;
  // Carry the source's audio into the output, stream-copied.  On by default, because
  // silently dropping a soundtrack is a surprising thing for a video tool to do, and
  // the copy is free: it adds no transcode stage, and the encoder is already ~89% of
  // this pipeline's CPU (README, round twenty-two), so there is no meaningful budget
  // for re-encoding audio even if a user asked for it.  Set 0 to write video only.
  // A source with no audio is reported on stderr and is not an error.
  bool audio = true;
  // Threads for the reader's uint16 -> float widening, which is the reader's
  // dominant cost and the thing that starves the GPU.  0 = half the logical
  // processors, capped at the physical core count.
  int reader_threads = 0;
  // Batch slots in flight.  0 = derive from the RAM budget.  A value below that
  // derived depth is honoured; a value above it is not, because the slots would not
  // fit.  The default pipeline runs at 3, which measured too shallow to keep the GPU
  // fed -- the worker was idle 47% of its stage time waiting on the reader.
  int queue_depth = 0;
  // Force the GPU to return float4 and let the host convert, instead of having the
  // scatter kernel emit packed rgba64le directly.  Slower by design (it doubles the
  // download and adds the host conversion), and it exists so the two output paths
  // can be A/B'd byte-for-byte against each other.  See README section 13.
  bool gpu_float_out = false;
  // How long a host worker waits for the GPU before absorbing a batch.  The GPU
  // is the faster engine, so host workers must not outbid it; this is the knob
  // that makes the mix adaptive.  0 = 60 ms.
  int host_grace_ms = 0;
  // Build and report the palette, then stop.  Tunes the palette without paying
  // for a full dither pass, which makes the saturation trade-off cheap to
  // measure instead of costing a full render per experiment.
  bool palette_only = false;
  // Fraction of physical RAM the in-flight frame buffers may occupy; 0 = auto
  // (about a third).  The queue depth is derived from this.
  double mem_fraction = 0.0;
  std::string codec = "libx264";
  int crf = 18;
  // veryfast, not medium.  Once the dither stopped being the bottleneck the
  // encoder's preset became the lever that matters -- not through its own busy
  // time, which barely moves, but by how fast it drains the pipe and so how long
  // the writer thread blocks.  Measured on 605 frames at 1080p:
  //     medium 19.7 fps, 29.3 MiB      veryfast 25.3 fps, 24.9 MiB
  //     superfast 25.6 fps, 52.8 MiB
  // veryfast is both faster than medium and *smaller*; superfast buys 1% more
  // speed for twice the bitrate, which is a bad trade on a 3-hour render.
  std::string preset = "veryfast";
  // Output pixel format.  yuv420p is the default for compatibility, but 4:2:0
  // chroma subsampling averages colour over 2x2 blocks, which smears a 16-colour
  // dither pattern badly: a frame that contains exactly 16 colours decodes back
  // with tens of thousands.  Use yuv444p (or yuv422p) when the palette has to
  // survive the round trip.
  // yuv444p, not yuv420p: chroma subsampling averages colour over 2x2 blocks, which
  // destroys a per-pixel dither pattern outright (measured 36031 colours out of
  // 16, versus 56 with ffv1 yuv444p).  See --video-lossless.
  std::string pix_fmt = "yuv444p";
  // Shorthand for codec=ffv1 + pix_fmt=yuv444p, i.e. the only combination in
  // which the 16-colour palette survives to the file.
  bool lossless = false;
  // Palette derivation.  Pool (default) quantizes every sampled frame on its own
  // and then quantizes the pool of those per-frame palettes, so each frame
  // contributes the same number of samples no matter how much grey it holds.
  // Montage (legacy) pools raw pixels, where palette slots go to whichever frame
  // has the most pixels of a given colour.  0 = auto, 1 = pool, 2 = montage.
  // Adopt the palette from an existing quantized image instead of building one.
  // Pair with ImageMagick:
  //   magick f1..f30 +append -colors 16 -unique-colors palette.png
  // so palette generation is ImageMagick's and the dithering is ours.
  std::string palette_from;
  // Generate the palette the way the reference ImageMagick pipeline does -- 8-bit,
  // no alpha, +append layout, IM's quantizer, then -unique-colors -- and then
  // dither the video on the GPU.  This is the one-command form of
  //   magick f1..f30 +append -colors 16 -unique-colors palette.png
  //   rdither --video --palette-from palette.png in.mp4 out.mp4
  // Segment support for --resume.  start_frame is the first frame of this
  // segment; rames_to_skip is how many frames the seek still has to drop so
  // the boundary is exact.
  // Crash-safe segmentation.  segment_frames <= 0 means one pass, no
  // segmentation (the default and the fast path).  resume continues from the
  // checkpoint beside the output.
  int segment_frames = 0;
  bool resume = false;
  std::int64_t start_frame = 0;
  // How many frames this segment covers.  VideoProcess stops at exactly this
  // many, so the segment is finalised as a complete file instead of running the
  // decoder to EOF.  0 = no limit (the non-segmented path).
  std::int64_t frames_in_segment = 0;
  bool im_palette = false;
  // Write the palette out after building it, so it can be inspected, kept
  // under version control, and fed back in with --palette-from.
  std::string palette_export;
  int palette_mode = 0;
  // Bit depth fed to the palette quantizer.  8 is the default and matches the
  // reference pipeline, which extracts 8-bit PPM; 16 measurably flattens the
  // palette because IM prunes at Quantum precision.  0 = auto (8).
  int palette_depth = 0;
  // Stage 1 reduces each frame to this many colours, stage 2 to colors.  Stage 1
  // wants more than the final count: it is removing the pixel-population weighting,
  // not trying to pick the answer, and a richer candidate set costs stage 2 nothing.
  int palette_stage1_colors = 64;
  // Ceiling on stage-1 frames.  Full-resolution QuantizeImage costs ~1.0 s per
  // 1080p frame, so sampling every frame of a long clip is hours of work for no
  // extra coverage; 64 evenly spaced frames already see every scene.
  // Ceiling on palette samples, default 256.  This is the limit that actually binds
  // in practice -- the 60 s seek budget works out to ~500 samples and the 64 Mpixel
  // montage cap to 4096 at the default 128px tile, so 256 is the one doing the work.
  //
  // It was 64, set for the *pooled* palette path where each sample costs a
  // full-resolution quantize (~1.0 s at 1080p).  Applying that number to the montage
  // path was a category error: there each sample costs one ffmpeg spawn and one seek
  // (0.12 s), so 64 capped a stage that could afford five hundred -- exactly the
  // coverage the default is for.  See --palette-budget-ms.
  int palette_max_samples = 256;
  // RGB distance (Q16 units) below which two stage-1 swatches count as the same
  // colour.  0 disables deduplication, which measured much worse than the
  // montage on grey-heavy footage.  65535 is the full Q16 range.
  int palette_dedup = 2600;
  bool quiet = false;
  int dump_palette_limit = 0;  // print this many palette entries
};

// Mean HSV saturation of the palette (0..1).  Reported for every run because a
// montage of many scenes silently dilutes the palette into desaturated
// mid-tones, and that is otherwise only visible by eye.
double MeanPaletteSaturation(const Palette& palette);

// How many palette entries are effectively neutral (saturation below 	hreshold).
// Mean saturation cannot see a palette that has lost its colour slots to grey.
int CountNeutralPaletteEntries(const Palette& palette, double threshold);

struct VideoResult {
  std::int64_t frames = 0;
  int palette_colors = 0;
  int palette_sampled = 0;
  int palette_tile = 0;
  // Pool mode: the swatch mosaic stage 2 actually quantizes.  Montage mode: 0.
  int palette_mosaic_w = 0;
  int palette_mosaic_h = 0;
  int palette_neutral = 0;  // entries with saturation below 0.2
  std::size_t palette_pixels = 0;
  double palette_ms = 0.0;
  double palette_secondary_ms = 0.0;  // stage 1 subtotal in pool mode
  double decode_ms = 0.0;
  double dither_ms = 0.0;
  double encode_ms = 0.0;
  double convert_ms = 0.0;   // writer: float -> uint16 (host, per pixel)
  double pipe_ms = 0.0;      // writer: the ffmpeg pipe write
  // End to end, INCLUDING the palette stage.  This used to be the pipeline's own
  // elapsed time, taken inside VideoProcess, which starts after VideoBuildPalette has
  // already returned -- so the reported fps silently excluded the palette, and a run
  // whose palette cost 25 s more than another's reported the same fps.  "Throughput"
  // that ignores a serial stage the user waited through is not throughput.
  double total_ms = 0.0;
  // Pipeline shape, for reporting.
  std::int64_t cpu_frames = 0;
  std::int64_t gpu_frames = 0;
  int queue_depth = 0;
  int batch_frames = 0;
  int cpu_workers = 0;
  double in_flight_mb = 0.0;
  double ram_budget_mb = 0.0;
};

// Reads container metadata with ffprobe.
bool VideoProbe(const std::string& path, VideoInfo* out, std::string* error);

// True when `path` carries at least one audio stream.  Probed once and cached per
// path, so repeated calls in one render do not re-spawn ffprobe.  A silent source is
// not an error: it returns false with *error untouched.
bool VideoHasAudio(const std::string& path, const std::string& ffmpeg,
                   std::string* error);

// Locates ffmpeg/ffprobe; call once at startup.  Honours FFMPEG / FFPROBE
// environment overrides, then PATH.
bool VideoFindTools(std::string* ffmpeg, std::string* ffprobe, std::string* error);

// Quotes a path for a command line, matching the pipeline's own escaping.
std::string VideoQuotePath(const std::string& s);

// Runs an ffmpeg invocation to completion with no pipes, for post-processing
// (joining segments).  Returns false on a non-zero exit or a timeout.
bool VideoRunTool(const std::string& tool, const std::string& args,
                  int timeout_ms, int* exit_code, std::string* error);

// Ctrl-C / console-close handling.  The handler only sets a flag; the pipeline
// threads notice it at their next checkpoint and unwind normally, so the encoder
// gets its stdin closed and can finalise rather than leaving a truncated file.
void InstallInterruptHandler();
void RemoveInterruptHandler();
bool Interrupted();

// Decodes up to `options.palette_frames` evenly spaced frames, box-filters each
// into a palette_tile square, tiles them into one montage, and runs
// ImageMagick's own quantizer over that.  One palette for the whole sequence.
bool VideoBuildPalette(const std::string& path, const VideoOptions& opt,
                       const VideoInfo& info, Palette* palette, ColorTree* tree,
                       VideoResult* result, std::string* error);

// Full pipeline.  Requires a palette and tree from VideoBuildPalette.
bool VideoProcess(const std::string& in, const std::string& out,
                  const VideoOptions& opt, const VideoInfo& info,
                  const Palette& palette, const ColorTree& tree,
                  VideoResult* result, std::string* error);

}  // namespace rd

