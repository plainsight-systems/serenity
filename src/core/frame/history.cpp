#include "core/frame/history.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace serenity::frame {

bool HeldFrames::start(std::uint64_t since, std::uint64_t index, Extent size) {
  const bool remade = !made_ || !(size == size_);
  made_ = true;
  size_ = size;
  since_ = since;
  next_ = index + 1;
  return remade;
}

bool HeldFrames::continues(std::uint64_t since, std::uint64_t index,
                           Extent size) const {
  return made_ && since == since_ && index == next_ && size == size_;
}

std::string HeldFrames::holds() const {
  return made_ ? "frames " + std::to_string(since_) + " .. " +
                     std::to_string(next_ - 1) + " at " + to_string(size_)
               : "nothing yet";
}

Joined History::join(const FrameInputs& inputs, Extent size,
                     SceneMotion motion) {
  if (inputs.accumulated_since > inputs.index) {
    throw HistoryError(
        "frame " + std::to_string(inputs.index) + " accumulates since frame " +
        std::to_string(inputs.accumulated_since) + ", after itself");
  }
  const std::uint64_t held = inputs.index - inputs.accumulated_since;
  if (held > max_accumulated_frames) {
    throw HistoryError(
        "frame " + std::to_string(inputs.index) + " would join " +
        std::to_string(held) + " accumulated frames; an image holds at most " +
        std::to_string(max_accumulated_frames) + " (start over sooner)");
  }

  if (held == 0) {
    const bool remade =
        held_.start(inputs.accumulated_since, inputs.index, size);
    time_ = inputs.time;
    return Joined{0, remade};
  }

  if (!held_.continues(inputs.accumulated_since, inputs.index, size)) {
    throw HistoryError("frame " + std::to_string(inputs.index) +
                       " claims an image accumulated since frame " +
                       std::to_string(inputs.accumulated_since) + " at " +
                       to_string(size) +
                       ", which is not what the image holds (" + held_.holds() +
                       "); a frame skipped, or a change without starting over");
  }
  if (motion == SceneMotion::changing && inputs.time != time_) {
    throw HistoryError("frame " + std::to_string(inputs.index) + ", at " +
                       std::to_string(inputs.time.count()) +
                       " s, would join frames at " +
                       std::to_string(time_.count()) +
                       " s; the scene changes, so an image holds one instant "
                       "(start over at each new time)");
  }
  held_.take(inputs.index);
  return Joined{static_cast<std::uint32_t>(held), false};
}

std::uint64_t LiveHistory::since(std::uint64_t index, View view) {
  // Checked before index - since_ is taken, which would wrap (ES.104).
  if (index < since_) {
    throw HistoryError("frame " + std::to_string(index) +
                       " asked after frame " + std::to_string(since_) +
                       "; a running window's frames are asked for in order");
  }
  if (view == View::changed || changes_ ||
      index - since_ > max_accumulated_frames) {
    since_ = index;
  }
  return since_;
}

ReuseJoined ReuseHistory::join(const FrameInputs& inputs, Extent size) {
  if (inputs.reused_since > inputs.index) {
    throw HistoryError("frame " + std::to_string(inputs.index) +
                       " reuses history since frame " +
                       std::to_string(inputs.reused_since) + ", after itself");
  }
  const std::uint64_t held = inputs.index - inputs.reused_since;
  if (held == 0) {
    return ReuseJoined{0, held_.start(inputs.reused_since, inputs.index, size)};
  }
  if (!held_.continues(inputs.reused_since, inputs.index, size)) {
    throw HistoryError(
        "frame " + std::to_string(inputs.index) +
        " claims reservoirs reused since frame " +
        std::to_string(inputs.reused_since) + " at " + to_string(size) +
        ", which is not what the history holds (" + held_.holds() +
        "); a frame skipped, or a change without starting over");
  }
  held_.take(inputs.index);
  return ReuseJoined{held, false};
}

std::uint64_t LiveReuse::since(std::uint64_t index, View view) {
  if (started_ && index < last_) {
    throw HistoryError("frame " + std::to_string(index) +
                       " asked after frame " + std::to_string(last_) +
                       "; a running window's frames are asked for in order");
  }
  if (started_ && index == last_) {
    return since_;  // asked again: a frame the window had no drawable for
  }
  if (!started_ || view == View::changed || index != last_ + 1) {
    since_ = index;
  }
  started_ = true;
  last_ = index;
  return since_;
}

Sample HeadlessPlan::sample(SampleOf of) const {
  const std::uint64_t frame = of.frame;
  if (!(of.sample < samples)) {
    throw std::invalid_argument("HeadlessPlan::sample: sample " +
                                std::to_string(of.sample) + " of a frame of " +
                                std::to_string(samples));
  }
  // frame * samples + of.sample, and the start's * samples, checked before
  // they are taken, so an index, which seeds every random number, never
  // wraps to one used before (ES.103). of.sample < samples, so frame *
  // samples + of.sample fits whenever (frame + 1) * samples does.
  constexpr std::uint64_t most = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t start = instants ? frame : first;
  if (frame >= most / samples || start > most / samples) {
    throw std::overflow_error("HeadlessPlan::sample: frame " +
                              std::to_string(frame) +
                              "'s samples pass the largest frame index");
  }
  return Sample{frame * samples + of.sample, start * samples, 0};
}

HeadlessPlan plan_headless(const HeadlessRun& run) {
  // A graph that reuses history replays the run's frames from the
  // timeline's start, writing none of them (above).
  return HeadlessPlan{run.reuses ? 0u : run.first,
                      run.accumulates ? run.samples : 1u,
                      run.scene_changes && !run.time_frozen, run.first};
}

}  // namespace serenity::frame
