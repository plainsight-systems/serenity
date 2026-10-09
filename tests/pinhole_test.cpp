// The pinhole camera: which cameras can be framed, and the framing itself,
// against the formula in contracts/camera.h.

#include <cmath>
#include <numbers>
#include <string>

#include <doctest/doctest.h>

#include "core/camera/pinhole.h"

using serenity::contracts::Camera;
using serenity::contracts::Float3;

namespace {

float dot(Float3 a, Float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length(Float3 a) {
    return std::sqrt(dot(a, a));
}

Camera looking_down_minus_z() {
    return Camera{{0.0f, 1.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 60.0f};
}

const char* why(const Camera& camera) {
    const char* reason = nullptr;
    CHECK_FALSE(serenity::camera::valid(camera, &reason));
    return reason == nullptr ? "" : reason;
}

}  // namespace

TEST_CASE("a camera is valid only when it can be framed") {
    const char* reason = "unset";
    CHECK(serenity::camera::valid(looking_down_minus_z(), &reason));
    CHECK(reason == nullptr);

    Camera at_itself = looking_down_minus_z();
    at_itself.look_at = at_itself.position;
    CHECK(std::string(why(at_itself)).find("position") != std::string::npos);

    Camera up_along_view = looking_down_minus_z();
    up_along_view.up = {0.0f, 0.0f, -3.0f};
    CHECK(std::string(why(up_along_view)).find("parallel") != std::string::npos);

    Camera no_up = looking_down_minus_z();
    no_up.up = {0.0f, 0.0f, 0.0f};
    CHECK(std::string(why(no_up)).find("up") != std::string::npos);

    for (float fov : {0.0f, 180.0f, -10.0f, 200.0f}) {
        Camera wrong_fov = looking_down_minus_z();
        wrong_fov.vertical_fov_degrees = fov;
        CHECK(std::string(why(wrong_fov)).find("vertical_fov_degrees") != std::string::npos);
    }

    Camera not_finite = looking_down_minus_z();
    not_finite.position.x = std::nanf("");
    CHECK(std::string(why(not_finite)).find("finite") != std::string::npos);
}

TEST_CASE("framing: the basis spans the field of view, with square pixels") {
    const Camera camera = looking_down_minus_z();
    const auto data = serenity::camera::shader_form(camera, {1600, 900});
    const float half_height = std::tan(30.0f * std::numbers::pi_v<float> / 180.0f);

    CHECK(data.origin.x == 0.0f);
    CHECK(data.origin.y == 1.0f);
    CHECK(data.origin.z == 5.0f);
    CHECK(data.forward.z == doctest::Approx(-1.0f));
    CHECK(length(data.forward) == doctest::Approx(1.0f));
    CHECK(length(data.up) == doctest::Approx(half_height));
    CHECK(length(data.right) == doctest::Approx(half_height * 1600.0f / 900.0f));
    CHECK(dot(data.forward, data.up) == doctest::Approx(0.0f).epsilon(1e-6));
    CHECK(dot(data.forward, data.right) == doctest::Approx(0.0f).epsilon(1e-6));
    CHECK(dot(data.up, data.right) == doctest::Approx(0.0f).epsilon(1e-6));
    // Right is +x and up is +y for a camera looking down -z with y up: the
    // image is not mirrored or upside down.
    CHECK(data.right.x > 0.0f);
    CHECK(data.up.y > 0.0f);
}

TEST_CASE("framing: up need not be perpendicular to the view") {
    Camera tilted = looking_down_minus_z();
    tilted.look_at = {0.0f, 0.0f, 0.0f};  // looking down a little
    const auto data = serenity::camera::shader_form(tilted, {100, 100});
    CHECK(dot(data.forward, data.up) == doctest::Approx(0.0f).epsilon(1e-6));
    CHECK(data.up.y > 0.0f);
    CHECK(data.forward.y < 0.0f);
}
