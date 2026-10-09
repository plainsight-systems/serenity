#include "core/measurement/frame_times.h"

#include <algorithm>

namespace serenity::measurement {

void FrameTimes::add(frame::Seconds gpu_start, frame::Seconds gpu_end) {
    const frame::Seconds duration = gpu_end - gpu_start;
    if (current_.frames == 0) {
        current_.shortest = duration;
        current_.longest = duration;
    } else {
        current_.shortest = std::min(current_.shortest, duration);
        current_.longest = std::max(current_.longest, duration);
    }
    ++current_.frames;
    total_ += duration.count();
}

std::optional<Summary> FrameTimes::take(frame::Seconds now) {
    if (!started_) {
        started_ = now;
        return std::nullopt;
    }
    if (now - *started_ < period_ || current_.frames == 0) {
        return std::nullopt;
    }
    Summary summary = current_;
    summary.mean = frame::Seconds(total_ / static_cast<double>(current_.frames));
    started_ = now;
    current_ = Summary{};
    total_ = 0.0;
    return summary;
}

}  // namespace serenity::measurement
