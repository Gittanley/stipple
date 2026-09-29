// rdither_riemersma.fx -- Riemersma-flavoured palette dithering as a Reshade effect.
//
// ONE FILE, EVERY RENDER API.  This is HLSL, and it is not written four times
// because Reshade does the translation: it compiles the same source for D3D9,
// D3D10, D3D11, D3D12, and -- with Reshade's own backends -- Vulkan and Metal.
//
// ---------------------------------------------------------------------------
// RESHADE EFFECT SYNTAX -- the things that are NOT plain HLSL.
// ---------------------------------------------------------------------------
// Reshade parses this file with its own preprocessor and effect parser, not
// fxc's, and the differences are load-bearing.  Every item here was a real
// compile error in an earlier version of this shader, and the sequence is worth
// recording because the first version was checked with a HLSL compiler, passed
// clean, and still produced seven errors on a real Reshade install.  A HLSL
// compile is necessary and is NOT sufficient.
//
//   1. Uniform annotations are separated by SEMICOLONS, not commas:
//          < ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; >
//      Commas give "syntax error: unexpected ','" on every annotated uniform.
//
//   2. There is no `compile` in the technique block.  It is fxc effect syntax
//      and Reshade's parser rejects it as a reserved word.  Entry points are
//      named directly:  PixelShader = PS_RDither;
//
//   3. The backbuffer is taken from ReShade.fxh, NOT declared locally.  The two
//      local forms both fail: `sampler BackBuffer : SAMPLER0;` gives
//      "'BackBuffer': missing 'Texture' property", and the braced form needs
//      an EQUALS SIGN -- `sampler X { Texture = texLUT; }` -- which is what the
//      shaders shipped with Reshade actually use.  Rather than depend on a
//      detail of that syntax, this shader declares no sampler at all and uses
//      ReShade::BackBuffer, which is what the shaders in this install do and so
//      is the form most likely to keep working.
//
//   4. The pass is unnamed (`pass { ... }`).
//
//   5. PostProcessVS comes from ReShade.fxh, so there is no hand-written vertex
//      shader to get wrong.
//
//   6. The screen size is NOT named BACKBUFFER_WIDTH / BACKBUFFER_HEIGHT.  Those
//      were 2.x-era spellings; Reshade 3.x -- which ReShade.fxh now requires --
//      provides BUFFER_WIDTH, BUFFER_HEIGHT, BUFFER_RCP_WIDTH and
//      BUFFER_RCP_HEIGHT instead.  The old names gave "undeclared identifier".
//      This shader uses a fixed Hilbert domain instead, so it depends on no
//      screen-size symbol at all and cannot break when they are renamed again;
//      swap in BUFFER_WIDTH / BUFFER_HEIGHT here if you would rather have the
//      pattern resolution-matched.
//
// ---------------------------------------------------------------------------
// CONTENT AWARENESS, and where it comes from.  Read this before tuning it.
// ---------------------------------------------------------------------------
//
// A fixed palette -- however chosen -- quantises every frame the same way, and
// that was the complaint that produced "the colours are very toxic": a 16-entry
// cube spans the whole gamut, so a frame containing no magenta at all can be
// rendered partly magenta, and nothing in the algorithm knows or cares.
//
// The mechanism here is borrowed from SweetFX's Vibrance.fx, which is in the
// same shader folder and whose entire algorithm is one line:
//
//     float sat = max(color) - min(color);
//     color = lerp(luma, color,
//                  1.0 + coeff * (1.0 - sign(coeff) * sat));
//
// The lesson is not the formula, it is the SHAPE: a per-pixel quantity measured
// from the content -- here, the pixel's own saturation -- scales how much the
// operation is allowed to act.  Already-saturated pixels are protected; neutral
// ones are worked on.  That is what "intelligently" means in that shader's
// tooltip, and it is a better principle than any specific weighting of it.
//
// Applied here it becomes: measure the neighbourhood, and penalise palette
// entries that are far from it, MORE where that neighbourhood is neutral.  A
// grey area cannot reach for a saturated entry; an already-vivid area is judged
// on the pixel itself.  See the search below for the implementation.
//
// Two properties worth stating, because they are why it is worth doing at all:
//   * it costs one extra dot product per candidate over the existing search, and
//     five texture taps -- no extra storage, no K-nearest array to spill;
//   * it degrades continuously.  contentAwareness = 0 is exactly the previous
//     global-palette behaviour, so the old rendering is still reachable.
//
// What it is NOT: this is a local, per-pixel adaptation over a ~40-pixel
// neighbourhood.  It is not a histogram, not a global palette fitted to the
// frame, and it will not stop a scene-wide palette shift.  A real quantiser
// samples the whole image; a fragment shader cannot, in one pass, without a
// pre-pass.  This is the closest a single pass gets, and it is a different thing
// from what rdither does -- rdither harvests a montage of sampled frames and
// quantises that.
//
// ---------------------------------------------------------------------------
// WHAT THIS IS NOT.  Read this before judging the output against rdither.exe.
// ---------------------------------------------------------------------------
//
// A Riemersma dither is a SEQUENTIAL walk.  quantize.c visits pixels in Hilbert
// curve order, carries a 16-entry error queue forward through that order, and
// each pixel's palette choice depends on all 16 before it.  A fragment shader is
// SIMULTANEOUS: it cannot know what its neighbours chose, and it cannot write to
// them.  A true Riemersma walk is therefore not expressible in one pass -- not
// with a clever trick, not with more registers.  Getting the real thing needs a
// multi-pass ping-pong (one pass per error-queue slot, or a sparse scan), which
// is several .fx files and a persistent history texture.
//
// So this is an APPROXIMATION, and the approximation is chosen to reproduce the
// part of Riemersma that the eye actually reads.  Riemersma's characteristic
// look is not the arithmetic; it is that the quantisation error is distributed
// ALONG THE HILBERT CURVE, which is why its noise follows the curve's long
// serpentine runs instead of forming a fixed screen-space lattice.  This shader
// perturbs the palette decision along exactly that curve, so the error is
// low-frequency and curve-following like the original, with none of its
// sequential machinery.
//
// The consequence to be honest about: the result is NOT bit-exact with
// rdither.exe, and could not be.  rdither is bit-exact against ImageMagick 7.1.2
// because it ports quantize.c line for line -- the octree-restricted ClosestColor,
// the memo table's order dependence, the Q16 double arithmetic.  A shader has none
// of that: it searches the WHOLE palette (which is actually a different, and by
// rdither's own admission more accurate, metric -- see `d_nearest_linear` in
// include/rd_cuda_common.cuh, which rdither implements and does not use) and it
// has float precision the original does not.  Treat this as "the Riemersma look,
// at N colours, in a game", not as a port of the port.
//
// ---------------------------------------------------------------------------
// SETTINGS.  Colour count and both bit depths are shader-side only, exactly as
// asked.  rdither deliberately has no such options, because for it bit depth is a
// property of the image and a knob there would imply a fidelity it lacks.
// ---------------------------------------------------------------------------

#include "ReShade.fxh"

uniform int   numColors    <
  ui_type = "slider";
  ui_min = 2.0;
  ui_max = 256.0;
  ui_step = 1.0;
> = 16;

// BITS PER CHANNEL FOR THE PALETTE.  Entries are snapped to this many bits BEFORE
// the search runs, so the quantiser chooses between genuinely coarse colours
// rather than between exact colours that are then printed coarsely.  That makes
// it a different algorithm and not a display setting: it changes WHICH entry
// wins.  0 = leave the palette exact, which for an 8-bit backbuffer is the same
// as 8 bits since the sampler has already quantised it.
uniform int   paletteBits  <
  ui_type = "slider";
  ui_min = 0.0;
  ui_max = 16.0;
  ui_step = 1.0;
> = 0;

// BITS PER CHANNEL FOR THE OUTPUT, applied last, to the chosen entry, so it
// decides what reaches the backbuffer.  Independent of paletteBits on purpose:
// you can dither to a 5-bit palette and still write full 8-bit output, or keep an
// exact palette and quantise the write.
uniform int   outputBits   <
  ui_type = "slider";
  ui_min = 0.0;
  ui_max = 16.0;
  ui_step = 1.0;
> = 0;

// How strongly the palette follows the CONTENT, 0..1.  This is the Vibrance
// idea, borrowed -- see the long comment at the search below for the argument and
// the exact line it comes from.
// 0 = a fixed global palette (the "toxic colours" behaviour).  1 = a pixel's
// available colours are biased hard toward what its neighbourhood actually looks
// like.  The default is 0.75: enough to stop the render reaching for colours that
// are nowhere in the picture, without collapsing the palette onto a single hue.
uniform float contentAwareness <
  ui_type = "slider";
  ui_min = 0.0;
  ui_max = 1.0;
> = 0.75;

// Radius of the neighbourhood the content anchor is taken from, as a fraction of
// screen height.  Small sees only the pixel's immediate surroundings, so a
// gradient gives every pixel its own palette; large averages over a big area and
// is smoother but describes a broader region.  0.06 is roughly a 40-pixel
// neighbourhood at 1080p, which spans a local colour but not a whole scene.
uniform float adaptRadius <
  ui_type = "slider";
  ui_min = 0.005;
  ui_max = 0.5;
> = 0.06;

// How hard the curve-following perturbation pushes toward the second-nearest
// entry.  0 = plain nearest-colour banding; 1 = the full spread.  Exposed because
// the useful value depends on how far the content sits from the palette, and no
// single number suits all content.
uniform float strength    <
  ui_type = "slider";
  ui_min = 0.0;
  ui_max = 1.0;
> = 1.0;

// THE PALETTE IS GENERATED, ALWAYS.  There is deliberately no uniform to supply
// one, and the reason is worth recording because it cost a black screen and a
// "toxic colours" round to learn.
//
// The first version took `uniform float3 rd_palette[256]` and expected the user
// to fill it.  That fails twice over:
//
//   * An unset uniform array arrives as all zeros.  Every entry is then identical,
//     the two-nearest search finds d1 == d2, and the shader dutifully writes
//     entry 0 -- which is (0,0,0).  Nothing about that is wrong in a way a
//     compiler or a shader-load log reports: it renders pure black, which reads as
//     a crash rather than as "you forgot to fill in the palette".
//   * 256 float3 values are not a thing anyone fills in through Reshade's UI.
//     So even when it did work it worked for nobody.
//
// A generated palette cannot fail that way: there is no state to be unset, the
// output is fully determined by the four sliders below, and the shader is useful
// the moment it loads.  If a captured palette is ever wanted, the right shape is
// a 1-D palette TEXTURE -- which is exactly what `rdither --palette-export p.png`
// writes, a 256x1 strip -- not a uniform array.  That is a future option, not a
// present one.

// SATURATION OF THE GENERATED PALETTE, and the reason the default is 0.65
// rather than 1.0.
//
// An even RGB cube is the obvious default and it is a BAD one for real content,
// which showed up as "the colours are toxic".  The reason is structural: a cube
// is uniform in RGB, and its corners are the pure primaries.  At sixteen colours
// the generated set contains full-strength cyan, green and magenta.  Quantising a
// photograph to those does not give a punchy version of the original, it gives a
// *different* colour rendition: neon midtones, clipped skin, and a hue shift no
// amount of dithering can hide.  The dither was the least of what was wrong.
//
// Pulling each entry toward its own luma fixes the rendition without touching the
// coverage, because the cube still spans the whole gamut in lightness -- it just
// stops insisting on maximum chroma everywhere.  0.65 keeps colour identity
// while removing the neon; 0 gives a pure grey ramp, which is the classic
// "posterise luminance" look and is a legitimate thing to want.
uniform float paletteSaturation <
  ui_type = "slider";
  ui_min = 0.0;
  ui_max = 1.0;
> = 0.65;

// The Hilbert grid the curve is evaluated on.  A fixed constant rather than the
// backbuffer size, so the shader needs no screen-size symbol; see note 6 above.
// 256 gives 65536 curve positions, which is more than any plausible count of
// distinct dither cells on a screen, and the pattern is stretched rather than
// re-tiled on unusual aspect ratios -- immaterial for a dither, and it avoids a
// seam at the wrap.
#define RD_HILBERT_DOMAIN 256

// Smallest side of a cube holding at least `n` entries.  A loop rather than
// cbrt(), which does not exist in shader model 3 and which this shader keeps
// compatible with D3D9.  At most seven iterations for n <= 256.
int CubeSide(int n)
{
  int s = 1;
  while (s * s * s < n) ++s;
  return s;
}

// Entry `i` of a generated palette: an even RGB cube.
//
// The cube is NOT filled front to back, because numColors rarely fills one.
// Sixteen colours would be the first sixteen of a 3x3x3 -- 27 slots -- and that
// clusters them into one corner of the cube and leaves the rest of the gamut
// unrepresented, which is exactly the "sixteen colours looks like eight" failure.
// So the n entries are spread over the whole cube by striding:
// slot = i * total / n.  For 16 that gives 0,1,3,5,6,8,10,11,13,15,16,18,20,21,23,25
// -- every corner region represented.
float3 CubeColour(int i, int n)
{
  const int side = CubeSide(n);
  const int total = side * side * side;
  const int slot = (total * i) / max(1, n);
  const int b = slot % side;
  const int g = (slot / side) % side;
  const int r = slot / (side * side);
  const float d = (float)(side - 1);   // side >= 2, because numColors is clamped to >= 2
  return float3((float)r, (float)g, (float)b) / d;
}

// The palette entry the search considers, before any bit-depth snap.  The only
// way in: the palette is generated, so this is the single definition of what
// colour index i means and there is no second source that could disagree with it.
float3 PaletteAt(int i, int n, float saturation)
{
  const float3 c = CubeColour(i, n);
  // Rec.709 luma, the usual weighting for sRGB content.
  const float y = dot(c, float3(0.2126, 0.7152, 0.0722));
  return lerp(y.xxx, c, saturation);
}

// ---------------------------------------------------------------------------
// Hilbert d -> position along the curve, 0..1, as the standard bit-interleaving
// form.  Written out rather than read from a 1-D lookup texture: that costs a
// sampler and a texel fetch per pixel to save thirty integer ops.
// ---------------------------------------------------------------------------
float Hilbert01(int x, int y, int n)
{
  int rx, ry, d = 0;
  for (int s = n >> 1; s > 0; s >>= 1)
  {
    rx = (x & s) > 0 ? 1 : 0;
    ry = (y & s) > 0 ? 1 : 0;
    d += s * s * ((3 * rx) ^ ry);
    if (ry == 0)
    {
      if (rx == 1) { x = s - 1 - x; y = s - 1 - y; }
      int t = x; x = y; y = t;
    }
  }
  return (float)d / max(1.0f, (float)(n * n));
}

float4 PS_RDither(in float4 position : SV_Position, in float2 texcoord : TEXCOORD0) : SV_Target
{
  const float3 src = tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0, 0)).rgb;
  const int n = clamp(numColors, 2, 256);
  const float pbits = (float)max(paletteBits, 0);
  const float obits = (float)max(outputBits, 0);
  const float pscale = exp2(pbits);          // 1.0 when pbits == 0
  const float oscale = exp2(obits);
  const float pmax   = pscale - 1.0;         // 0.0 when pbits == 0, disabling the snap

  // Where this pixel sits along the Hilbert curve, 0..1.  This is the whole
  // approximation: rdither's error follows the Hilbert order, so perturbing along
  // the Hilbert order reproduces the signature of error diffusion -- long
  // serpentine runs of one palette entry, sharp turns where the curve doubles
  // back -- without any sequential state.
  const int  D = RD_HILBERT_DOMAIN;
  const float cpos = Hilbert01((int)(texcoord.x * D), (int)(texcoord.y * D), D);

  // ---- the content anchor ------------------------------------------------
  // What this neighbourhood actually looks like, as opposed to what this one
  // pixel is.  Five taps on a cross, at a radius in screen height, corrected for
  // aspect so the neighbourhood is round rather than stretched.  Taken from a
  // low-frequency sample on purpose: the anchor has to vary SMOOTHLY across the
  // image, or every pixel would get a different palette and the result would be
  // noise with extra steps instead of a dither.
  const float3 LW = float3(0.2126, 0.7152, 0.0722);
  float3 base = src;
  if (contentAwareness > 0.0)
  {
    const float ry = adaptRadius;
    const float rx = adaptRadius * (float)BUFFER_WIDTH / (float)BUFFER_HEIGHT;
    float3 acc = src;
    acc += tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2( rx,  0.0), 0, 0)).rgb;
    acc += tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2(-rx,  0.0), 0, 0)).rgb;
    acc += tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2( 0.0,  ry), 0, 0)).rgb;
    acc += tex2Dlod(ReShade::BackBuffer, float4(texcoord + float2( 0.0, -ry), 0, 0)).rgb;
    base = lerp(src, acc * 0.2, contentAwareness);
  }

  // ---- the two nearest entries, ranked by a CONTENT-WEIGHTED distance ----
  //
  // This is where the Vibrance lesson lands, and it lands on the METRIC rather
  // than on the palette.  Vibrance's insight is that a per-pixel quantity -- this
  // pixel's own saturation -- should scale how much the effect is allowed to act:
  //
  //     color = lerp(luma, color, 1.0 + coeff * (1.0 - sign(coeff) * saturation));
  //
  // so already-saturated pixels are left alone and neutral ones are worked on.
  // The same rule here says: penalise palette entries that are far from the local
  // content, and do it MORE where the content is neutral.  A grey region then
  // cannot reach for a saturated entry, and a region that is already vivid is
  // judged on the pixel itself.
  //
  // The weight is (1 - saturation) for exactly that reason, and the whole thing
  // costs one extra dot product per candidate.  Two distances are mixed rather
  // than one because they fail differently: a luma-only guard would let a grey
  // area pick a blue or a red entry of the right brightness, and an RGB-only guard
  // treats a slight hue error like a huge one.  The luma term is weighted up
  // because brightness banding is what the eye forgives least.
  const float satBase = max(base.r, max(base.g, base.b)) - min(base.r, min(base.g, base.b));
  const float guard = contentAwareness * (1.0 - satBase);

  float d1 = 1e30, d2 = 1e30;
  int   i1 = 0,  i2 = 0;
  for (int i = 0; i < 256; ++i)
  {
    if (i >= n) break;
    float3 c = PaletteAt(i, n, paletteSaturation);
    if (pmax > 0.0) c = floor(c * pscale + 0.5) / pscale;

    const float3 ds = src  - c;
    const float3 db = base - c;
    const float dl = dot(LW, db);
    // 0.25 / 0.75: mostly full RGB distance to the content, partly pure luma.
    const float pen = guard * (0.25 * dl * dl + 0.75 * dot(db, db));

    const float dist = dot(ds, ds) + pen;
    if (dist < d1)      { d2 = d1; i2 = i1; d1 = dist; i1 = i; }
    else if (dist < d2) { d2 = dist;  i2 = i; }
  }

  // ---- choose between them, perturbed along the curve --------------------
  // `t` is 0 when the pixel is essentially ON the nearest entry and approaches
  // 0.5 when it sits exactly between the two nearest -- so its useful range is
  // [0, 0.5], and the threshold has to live in that same range or the dither
  // degenerates into "always pick the second nearest".  Getting that range wrong
  // is the whole ballgame here.
  float pick1 = 1.0;
  if (d1 + d2 > 0.0)
  {
    const float t   = d1 / (d1 + d2);        // in [0, 0.5]
    const float thr = cpos * 0.5 * strength;  // in [0, 0.5]
    pick1 = (t < thr) ? 0.0 : 1.0;           // nearer wins unless the curve says otherwise
  }

  // Written through PaletteAt, the same function the search used, rather than
  // recomputed.  A second definition of "what colour is index i" is a second
  // thing that can disagree with the first, and when they disagree the search
  // measures a distance to one colour and writes another -- which is invisible
  // and looks like a colour error, not a logic error.
  float3 outc = (pick1 > 0.5) ? PaletteAt(i1, n, paletteSaturation)
                              : PaletteAt(i2, n, paletteSaturation);
  if (pmax > 0.0) outc = floor(outc * pscale + 0.5) / pscale;
  if (obits > 0.0) outc = floor(outc * oscale + 0.5) / oscale;

  return float4(outc, 1.0);
}

technique RDitherRiemersma
<
  ui_tooltip = "Riemersma-flavoured palette dithering (Hilbert-curve error placement).";
>
{
  pass
  {
    VertexShader = PostProcessVS;
    PixelShader  = PS_RDither;
  }
}






