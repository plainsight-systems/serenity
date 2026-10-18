#include "metal/frame/display_frames.h"

#include <chrono>

namespace serenity::metal {

namespace {

// How long the window may wait for an event (display_frames.h): frames come
// from the run loop meanwhile, and a frame's failure is noticed when the
// wait returns, so within a tenth of a second.
constexpr frame::Seconds event_wait = std::chrono::milliseconds{100};

}  // namespace

DisplayFrames::DisplayFrames(Submission& submission, Presenter& presenter, Renderer& renderer,
                             presentation::FrameRate rate)
    : submission_(submission), presenter_(presenter), renderer_(renderer), rate_(rate) {}

void DisplayFrames::run(presentation::FrameClient& client) {
    // The client is held for the run alone, however it ends (R.1).
    struct Running {
        DisplayFrames& frames;
        Running(DisplayFrames& f, presentation::FrameClient& c) : frames(f) {
            frames.client_ = &c;
            frames.first_frame_ = frames.submission_.next_sequence();
            frames.view_changed_ = false;
            frames.failure_ = nullptr;
        }
        Running(const Running&) = delete;
        Running& operator=(const Running&) = delete;
        Running(Running&&) = delete;
        Running& operator=(Running&&) = delete;
        ~Running() { frames.client_ = nullptr; }
    };
    const Running running(*this, client);
    {
        // Frames are called from here on, while the client waits for events;
        // none after the link is destroyed, at the end of this block.
        const DisplayLink link(presenter_, rate_, *this);
        while (!failure_ && client.events(event_wait)) {
            // Nothing between waits: the frames run inside them.
        }
    }
    if (failure_) {
        std::rethrow_exception(failure_);
    }
}

void DisplayFrames::frame(const DisplayUpdate& update) noexcept {
    if (client_ == nullptr || failure_) {
        return;
    }
    try {
        render(update, *client_);
    } catch (...) {
        failure_ = std::current_exception();
    }
}

void DisplayFrames::render(const DisplayUpdate& update, presentation::FrameClient& client) {
    // Steps 1 to 6 (display_frames.h).
    const frame::Extent wanted = client.size();
    if (wanted.width == 0 || wanted.height == 0) {
        return;
    }
    if (wanted != presenter_.size()) {
        view_changed_ = true;
    }
    presenter_.resize(wanted);
    if (!presenter_.renders_into(update.drawable)) {
        return;
    }
    const frame::FrameInputs inputs =
        client.inputs(update.presentation, view_changed_ ? frame::View::changed : frame::View::same);
    const WindowFrame rendered = render_to_window(submission_, presenter_, renderer_, inputs, update.drawable);
    view_changed_ = false;
    presentation::FrameReport report{
        .size = presenter_.size(),
        .settled = std::nullopt,
        .non_finite_samples = renderer_.non_finite_samples(),
        .mnee_lost_requests = renderer_.mnee_lost_requests(),
    };
    if (rendered.settled && rendered.settled->sequence >= first_frame_) {
        report.settled = presentation::SettledFrame{rendered.settled->gpu_start, rendered.settled->gpu_end};
    }
    client.shown(report);
}

}  // namespace serenity::metal
