// Which passes the CUDA backend builds (cuda/frame/renderer.h,
// unbuilt_pass), a unit: a graph of what it builds passes, and any other is
// refused naming the first pass it does not build.

#include <doctest/doctest.h>

#include <optional>
#include <string>

#include "core/frame/schedule.h"
#include "cuda/frame/renderer.h"
#include "support/schedules.h"

namespace serenity::cuda {
namespace {

bool names(const std::optional<std::string>& reason, const std::string& pass) {
  return reason &&
         reason->find("has no " + pass + " pass yet") != std::string::npos;
}

TEST_CASE(
    "every pass but the denoiser and the depth of field is built; those are "
    "refused by name") {
  CHECK_FALSE(unbuilt_pass(
      frame::Schedule{{frame::PassKind::test_pattern}, std::nullopt}));
  CHECK_FALSE(unbuilt_pass(frame::Schedule{
      {frame::PassKind::path, frame::PassKind::tone_map}, tests::look}));
  CHECK_FALSE(
      unbuilt_pass(frame::Schedule{{frame::PassKind::features}, std::nullopt}));
  CHECK_FALSE(unbuilt_pass(tests::restir_pt_graph(tests::restir_pt_passes)));
  CHECK(names(
      unbuilt_pass(frame::Schedule{{frame::PassKind::denoise}, std::nullopt}),
      "denoise"));
  CHECK(names(unbuilt_pass(frame::Schedule{{frame::PassKind::depth_of_field},
                                           std::nullopt}),
              "depth_of_field"));
}

}  // namespace
}  // namespace serenity::cuda
