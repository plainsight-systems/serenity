#include "core/measurement/frame_times.h"

#include <algorithm>
#include <stdexcept>

namespace serenity::measurement {

void FrameTimes::add(frame::Seconds gpu_start, frame::Seconds gpu_end) {
    if (!(gpu_end >= gpu_start)) {
        throw std::invalid_argument("FrameTimes::add: a frame that ended before it began");
    }
    // From the frame before's end if later, never past this one's (above).
    const frame::Seconds from = last_end_ ? std::clamp(*last_end_, gpu_start, gpu_end) : gpu_start;
    const frame::Seconds duration = gpu_end - from;
    last_end_ = gpu_end;
    if (current_.frames == 0) {
        current_.shortest = duration;
        current_.longest = duration;
    } else {
        current_.shortest = std::min(current_.shortest, duration);
        current_.longest = std::max(current_.longest, duration);
    }
    ++current_.frames;
    total_ += duration;
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
    summary.mean = total_ / static_cast<double>(current_.frames);
    started_ = now;
    current_ = Summary{};
    total_ = frame::Seconds{0.0};
    return summary;
}

}  // namespace serenity::measurement
