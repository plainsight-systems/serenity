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

TEST_CASE("the test pattern is built; every other pass is refused by name") {
  CHECK_FALSE(unbuilt_pass(
      frame::Schedule{{frame::PassKind::test_pattern}, std::nullopt}));
  CHECK(names(
      unbuilt_pass(frame::Schedule{
          {frame::PassKind::path, frame::PassKind::tone_map}, tests::look}),
      "path"));
  CHECK(names(unbuilt_pass(tests::restir_pt_graph(tests::restir_pt_passes)),
              "restir_pt_initial"));
}

}  // namespace
}  // namespace serenity::cuda
