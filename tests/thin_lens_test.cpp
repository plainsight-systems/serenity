// The thin-lens camera: which cameras can be framed, and the framing itself,
// against the formula in contracts/camera.h.

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

#include <doctest/doctest.h>

#include "core/camera/thin_lens.h"
#include "support/text.h"
#include "support/vector.h"

using serenity::contracts::Camera;
using serenity::contracts::Float3;
using serenity::tests::contains;

namespace {

constexpr float infinite = std::numeric_limits<float>::infinity();
const float not_a_number = std::nanf("");

double dot(Float3 a, Float3 b) {
    return serenity::tests::dot(serenity::tests::vec(a), serenity::tests::vec(b));
}

double length(Float3 a) {
    return std::sqrt(dot(a, a));
}

Camera looking_down_minus_z() {
    return Camera{{0.0f, 1.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 60.0f};
}

// Why `camera` is not valid; it must not be.
std::string why(const Camera& camera) {
    const auto reason = serenity::camera::invalid(camera);
    CHECK(reason.has_value());
    return std::string(reason.value_or(""));
}

bool finite(Float3 a) {
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}

}  // namespace

TEST_CASE("a camera is valid only when it can be framed") {
    CHECK_FALSE(serenity::camera::invalid(looking_down_minus_z()).has_value());

    Camera at_itself = looking_down_minus_z();
    at_itself.look_at = at_itself.position;
    CHECK(contains(why(at_itself), "position"));

    Camera up_along_view = looking_down_minus_z();
    up_along_view.up = {0.0f, 0.0f, -3.0f};
    CHECK(contains(why(up_along_view), "parallel"));

    Camera no_up = looking_down_minus_z();
    no_up.up = {0.0f, 0.0f, 0.0f};
    CHECK(contains(why(no_up), "up"));

    for (const float fov : {0.0f, 180.0f, -10.0f, 200.0f}) {
        Camera wrong_fov = looking_down_minus_z();
        wrong_fov.vertical_fov_degrees = fov;
        CHECK(contains(why(wrong_fov), "vertical_fov_degrees"));
    }

    Camera not_finite = looking_down_minus_z();
    not_finite.position.x = not_a_number;
    CHECK(contains(why(not_finite), "finite"));
}

TEST_CASE("framing: the basis spans the field of view, with square pixels") {
    const Camera camera = looking_down_minus_z();
    const auto data = serenity::camera::shader_form(camera, {1600, 900});
    const double half_height = std::tan(30.0 * std::numbers::pi / 180.0);

    CHECK(data.origin.x == 0.0f);
    CHECK(data.origin.y == 1.0f);
    CHECK(data.origin.z == 5.0f);
    CHECK(data.forward.z == doctest::Approx(-1.0f).scale(0).epsilon(1e-6));
    CHECK(length(data.forward) == doctest::Approx(1.0).scale(0).epsilon(1e-6));
    CHECK(length(data.up) == doctest::Approx(half_height).scale(0).epsilon(1e-6));
    CHECK(length(data.right) == doctest::Approx(half_height * 1600.0 / 900.0).scale(0).epsilon(1e-6));
    // Square to each other: within 1e-6 of 0, absolutely.
    CHECK(dot(data.forward, data.up) == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(dot(data.forward, data.right) == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(dot(data.up, data.right) == doctest::Approx(0.0).epsilon(1e-6));
    // Right is +x and up is +y for a camera looking down -z with y up: the
    // image is not mirrored or upside down.
    CHECK(data.right.x > 0.0f);
    CHECK(data.up.y > 0.0f);
}

TEST_CASE("framing: up need not be perpendicular to the view") {
    Camera tilted = looking_down_minus_z();
    tilted.look_at = {0.0f, 0.0f, 0.0f};  // looking down a little
    const auto data = serenity::camera::shader_form(tilted, {100, 100});
    CHECK(dot(data.forward, data.up) == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(data.up.y > 0.0f);
    CHECK(data.forward.y < 0.0f);
}

TEST_CASE("a lens is framed with the camera, its radius 0 or more and its focus greater than 0") {
    Camera lensed = looking_down_minus_z();
    lensed.lens_radius = 0.012f;
    lensed.focus_distance = 1.4f;
    CHECK_FALSE(serenity::camera::invalid(lensed).has_value());
    const auto data = serenity::camera::shader_form(lensed, {1600, 900});
    CHECK(data.lens_radius == 0.012f);
    CHECK(data.focus_distance == 1.4f);
    // Without one, a pinhole.
    CHECK(serenity::camera::shader_form(looking_down_minus_z(), {16, 9}).lens_radius == 0.0f);

    for (const float radius : {-0.001f, not_a_number, infinite}) {
        Camera wrong = lensed;
        wrong.lens_radius = radius;
        CHECK(contains(why(wrong), "lens's radius"));
    }
    for (const float focus : {0.0f, -1.0f, not_a_number, infinite}) {
        Camera wrong = lensed;
        wrong.focus_distance = focus;
        CHECK(contains(why(wrong), "lens's focus"));
    }
}

TEST_CASE("a camera far from the origin is judged, and framed, without overflowing a float") {
    // look_at - position is past float's range; in double it is not, so the
    // camera frames to finite numbers rather than to a NaN that was called
    // valid.
    const Camera far_apart{{3e38f, 0.0f, 0.0f}, {-3e38f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 40.0f};
    CHECK_FALSE(serenity::camera::invalid(far_apart).has_value());
    const auto data = serenity::camera::shader_form(far_apart, {16, 9});
    CHECK(finite(data.forward));
    CHECK(finite(data.right));
    CHECK(finite(data.up));
    CHECK(data.forward.x == doctest::Approx(-1.0f).scale(0).epsilon(1e-6));

    // Squared in float, 1e20 overflows, and up was once called parallel to
    // the view; it is not.
    const Camera far_out{{1e20f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 40.0f};
    CHECK_FALSE(serenity::camera::invalid(far_out).has_value());
    CHECK(finite(serenity::camera::shader_form(far_out, {16, 9}).right));
}

TEST_CASE("framing refuses what it cannot frame: an invalid camera, an empty image") {
    Camera at_itself = looking_down_minus_z();
    at_itself.look_at = at_itself.position;
    CHECK_THROWS_AS(serenity::camera::shader_form(at_itself, {16, 9}), std::invalid_argument);
    CHECK_THROWS_AS(serenity::camera::shader_form(looking_down_minus_z(), {16, 0}), std::invalid_argument);
    CHECK_THROWS_AS(serenity::camera::shader_form(looking_down_minus_z(), {0, 9}), std::invalid_argument);
    // A Camera given nothing is not valid (contracts/camera.h).
    CHECK(contains(why(Camera{}), "position"));
}
