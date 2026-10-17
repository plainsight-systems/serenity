#pragma once

// The one sphere light the probe kernels that ask the emitter read (kernels/
// restir_probe.metal, kernels/path_probe.metal), bound as the passes bind
// theirs, and a probe run with it (ES.3, F.10): one definition for every
// test that runs such a probe.

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <vector>

#include "core/contracts/float3.h"
#include "core/contracts/transform.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
#include "gpu/support/probe_runner.h"

namespace serenity::tests {

// Sphere light record 0, of `radius` at `center`, `radiance` every channel,
// as the probes bind it: the records, the spheres, the transforms, the
// glows, which light each shape is, at bindings 1 to 5.
struct ProbeLight {
    contracts::Float3 center;
    float radius;
    float radiance;

    std::array<Binding, 5> bindings() const {
        return {Binding::of(std::vector<lights::LightRecord>{{lights::LightKind::sphere, 0u}}),
                Binding::of(std::vector<lights::SphereLightData>{{{radiance, radiance, radiance}, 0u}}),
                Binding::of(std::vector<contracts::Transform>{contracts::placed(center, radius)}),
                Binding::of(std::vector<float>{1.0f}), Binding::of(std::vector<std::uint32_t>{0u})};
    }
};

// The light's bindings, then `rest`.
inline std::vector<Binding> with_light(const ProbeLight& light, std::initializer_list<Binding> rest) {
    const std::array<Binding, 5> first = light.bindings();
    std::vector<Binding> all(first.begin(), first.end());
    all.insert(all.end(), rest.begin(), rest.end());
    return all;
}

// Runs `kernel` over `threads` with the light and `rest` bound from 1, its
// output of Out at 0.
template <GpuBytes Out>
std::vector<Out> run_with_light(ProbeRunner& gpu, const char* kernel, std::size_t threads, const ProbeLight& light,
                                std::initializer_list<Binding> rest) {
    std::vector<Binding> bindings{Binding::zeroed(threads * sizeof(Out))};
    const std::vector<Binding> inputs = with_light(light, rest);
    bindings.insert(bindings.end(), inputs.begin(), inputs.end());
    return gpu.dispatch(kernel, threads, std::span<const Binding>(bindings)).template read<Out>(0);
}

}  // namespace serenity::tests
