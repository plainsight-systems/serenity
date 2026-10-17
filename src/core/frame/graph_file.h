#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "core/frame/schedule.h"

namespace serenity::frame {

// Axis: Frame graph (reading it).
//
// A frame graph file: the passes a frame runs, in order, as data. Kept apart
// from scene files (change-axes.md: what is rendered is Scene content; which
// passes run, and in what order, is the Frame graph), so one scene runs under
// any graph: the naive estimator and ReSTIR side by side on the same scene,
// or the test pattern on none. Graph files live in graphs/, written by hand,
// in TOML. Reading them is toml++'s, pinned by commit (NOTICE).
//
// The format:
//
//   passes = ["path", "tone_map"] # pass names, in order; at least one
//   mnee = true                   # a boolean; only with path or a ReSTIR PT pass
//   checkerboard = true           # a boolean; only with ReSTIR PT's passes and denoise
//
//   [tone_map]                    # exactly when a pass is tone_map
//   exposure = 0.0                # stops, within [-10, 10] (core/passes/tone_map.h)
//   bloom = 0.04                  # the fraction of light spread into glare
//
//   [restir_di]                   # exactly when a pass is one of ReSTIR DI's
//   candidates = 8                # M, an integer in [1, 32]
//   bsdf_candidates = 1           # an integer in [0, 32]
//   history = 20                  # H, an integer in [1, 64]; (T)
//   permutation = true            # a boolean; (T)
//   neighbors = 1                 # k, an integer in [1, 32]; (S)
//   radius = 32.0                 # pixels, in [1, 1024]; (S)
//   depth_threshold = 0.1         # in (0, 1]; (T) or (S)
//   normal_threshold = 0.5        # a cosine, in [-1, 1]; (T) or (S)
//   bias_correction = "basic"     # "basic" or "raytraced"; (T) or (S)
//   visibility_reuse = true       # a boolean; (B)
//   visibility_age = 4            # frames, an integer in [0, 255]; (B), if reused
//   visibility_distance = 16.0    # pixels, in [0, 127]; (B), if reused
//
//   [restir_pt]                   # exactly when a pass is one of ReSTIR PT's
//   nee_candidates = 32           # an integer in [1, 32]
//   max_surfaces = 9              # an integer in [1, 14]
//   footprint = 0.02              # in (0, 10]
//   roughness = 0.2               # in [0, 1]
//   neighbors = 3                 # an integer in [1, 3]; (PS)
//   sigma = 16.0                  # pixels, in [0.8, 32]; (PS)
//   confidence_cap = 20.0         # in [1, 60]; (PT)
//   cap_floor = 1.0               # in [0, confidence_cap]; (PD)
//   cap_exponent = 0.1            # in (0, 1]; (PD)
//
//   [denoise]                     # exactly when a pass is denoise
//   weights = "../../serenity-data/denoiser-v1-model/wide"
//                                 # the exported network's directory; a
//                                 # relative path is relative to the graph
//                                 # file's (core/passes/denoise.h)
//
// (T): only in a graph with restir_temporal; (S): only with restir_spatial;
// (B): only in a graph with either, in basic mode. (PS): only with
// restir_pt_spatial; (PT): only with restir_pt_temporal; (PD): only with
// restir_pt_duplication. The values shown are the ones the graphs are
// written with: RTXDI's defaults for [restir_di], ReSTIR PT Enhanced's for
// [restir_pt] (core/passes/restir_pt.h). mnee turns on caustics by manifold
// next event estimation (Schedule::mnee, schedule.h), off unless written;
// checkerboard, the light traced at half its pixels a frame
// (Schedule::checkerboard, schedule.h), off unless written.
//
// ReSTIR PT's settings are core/contracts/path_reuse_settings.h's, which
// says what each does and why its range is what it is; each is required
// exactly when a pass of the graph reads it, as ReSTIR DI's are (below).
//
// The ReSTIR settings are core/contracts/reuse_settings.h's, which says what
// each does and why its range is what it is. Each is required exactly when a
// pass of the graph reads it, as the comments say: a graph without a
// temporal merge that writes `history` has a key nobody reads, and one with
// it that leaves `history` out has asked for no value, which is refused,
// never filled in (ES.45: the values the graphs are written with are
// named in that header, not defaulted here). What a pass reads can depend
// on another setting: visibility_age and visibility_distance are read only
// where visibility_reuse is true, and the three only where bias_correction
// is "basic", so in a raytraced graph they are keys nobody reads. Counts
// are TOML integers: 8.0 is not a count. bias_correction is one of its two
// strings, any other refused; permutation and visibility_reuse are TOML
// booleans.
//
// Every key is checked. A missing passes, a passes that is not a non-empty
// array of strings, an unknown pass name, a [tone_map], [restir_di],
// [restir_pt] or [denoise] table without its passes or the passes without
// the table, a setting missing, of the wrong type or out of range, an mnee
// that is not a boolean or is written, true or false, in a graph with
// neither path nor a ReSTIR PT pass, a checkerboard likewise in a graph
// without both ReSTIR PT's passes and the denoise pass, a schedule the core does not accept
// (schedule.h, invalid()), or a key nobody reads is a GraphFileError, never
// a default or an ignored line: a typo must fail the run that reads it, not
// render something else (E.2, E.14). Each GraphFileError names the file and
// the line, and an unknown pass name lists the known ones. A file that cannot be read whole, a directory's path or one
// that ends early, is a GraphFileError naming it, never an empty graph.
//
// Not performance-sensitive: read once, at start-up.

class GraphFileError : public std::runtime_error {
public:
    explicit GraphFileError(const std::string& what) : std::runtime_error(what) {}
};

// Reads the frame graph file at `path`.
Schedule load_schedule(const std::filesystem::path& path);

// Reads a frame graph from `text`, naming it `source` in errors. For tests,
// and for load_schedule() itself.
Schedule parse_schedule(std::string_view text, std::string_view source);

}  // namespace serenity::frame
