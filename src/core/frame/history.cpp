#include "core/frame/history.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace serenity::frame {

namespace {

std::string at(Extent size) {
    return std::to_string(size.width) + " x " + std::to_string(size.height);
}

}  // namespace

Joined History::join(const FrameInputs& inputs, Extent size, bool scene_changes) {
    if (inputs.accumulated_since > inputs.index) {
        throw HistoryError("frame " + std::to_string(inputs.index) + " accumulates since frame " +
                           std::to_string(inputs.accumulated_since) + ", after itself");
    }
    const std::uint64_t held = inputs.index - inputs.accumulated_since;
    if (held > max_accumulated_frames) {
        throw HistoryError("frame " + std::to_string(inputs.index) + " would join " + std::to_string(held) +
                           " accumulated frames; an image holds at most " + std::to_string(max_accumulated_frames) +
                           " (start over sooner)");
    }

    if (held == 0) {
        const bool remade = !made_ || !(size == size_);
        made_ = true;
        size_ = size;
        since_ = inputs.accumulated_since;
        next_ = inputs.index + 1;
        time_ = inputs.time;
        return Joined{0, remade};
    }

    if (!made_ || inputs.accumulated_since != since_ || inputs.index != next_ || !(size == size_)) {
        const std::string holds =
            made_ ? "frames " + std::to_string(since_) + " .. " + std::to_string(next_ - 1) + " at " + at(size_)
                  : "nothing yet";
        throw HistoryError("frame " + std::to_string(inputs.index) + " claims an image accumulated since frame " +
                           std::to_string(inputs.accumulated_since) + " at " + at(size) +
                           ", which is not what the image holds (" + holds +
                           "); a frame skipped, or a change without starting over");
    }
    if (scene_changes && inputs.time != time_) {
        throw HistoryError("frame " + std::to_string(inputs.index) + ", at " + std::to_string(inputs.time.count()) +
                           " s, would join frames at " + std::to_string(time_.count()) +
                           " s; the scene changes, so an image holds one instant (start over at each new time)");
    }
    next_ = inputs.index + 1;
    return Joined{static_cast<std::uint32_t>(held), false};
}

std::uint64_t LiveHistory::since(std::uint64_t index, bool view_changed) {
    // Checked before index - since_ is taken, which would wrap (ES.104).
    if (index < since_) {
        throw HistoryError("frame " + std::to_string(index) + " asked after frame " + std::to_string(since_) +
                           "; a running window's frames are asked for in order");
    }
    if (view_changed || changes_ || index - since_ > max_accumulated_frames) {
        since_ = index;
    }
    return since_;
}

Sample HeadlessPlan::sample(std::uint64_t frame, std::uint64_t s) const {
    if (!(s < samples)) {
        throw std::invalid_argument("HeadlessPlan::sample: sample " + std::to_string(s) + " of a frame of " +
                                    std::to_string(samples));
    }
    // frame * samples + s, and the start's * samples, checked before they
    // are taken, so an index, which seeds every random number, never wraps
    // to one used before (ES.103). s < samples, so frame * samples + s
    // fits whenever (frame + 1) * samples does.
    constexpr std::uint64_t most = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t start = instants ? frame : first;
    if (frame >= most / samples || start > most / samples) {
        throw std::overflow_error("HeadlessPlan::sample: frame " + std::to_string(frame) +
                                  "'s samples pass the largest frame index");
    }
    return Sample{frame * samples + s, start * samples};
}

HeadlessPlan plan_headless(std::uint64_t first, std::uint64_t samples, bool accumulates, bool scene_changes,
                           bool time_frozen) {
    return HeadlessPlan{first, accumulates ? samples : 1u, scene_changes && !time_frozen};
}

}  // namespace serenity::frame
