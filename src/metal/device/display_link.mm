// The display link (display_link.h), in Objective-C++: the one file of the
// program that names Core Animation's Objective-C types, since metal-cpp
// has no CAMetalDisplayLink. Compiled with ARC (CMakeLists.txt), so the
// link and its delegate, held in State, are released by State's
// destruction, after the link is invalidated.

#include "metal/device/display_link.h"

#import <Foundation/Foundation.h>
#import <QuartzCore/CAMetalDisplayLink.h>
#import <QuartzCore/CAMetalLayer.h>

#include <cmath>
#include <memory>

#include "metal/device/error.h"
#include "metal/device/submission.h"

namespace serenity::metal {
namespace {

// preferredFrameLatency takes 1 or 2 frames, and ours is the frames the
// submission keeps in flight (display_link.h).
static_assert(frames_in_flight == 2,
              "the display link's latency is the frames in flight, which it takes as 1 or 2");
constexpr float frame_latency = 2.0F;

void check(presentation::FrameRate rate) {
  const bool positive =
      std::isfinite(rate.minimum) && std::isfinite(rate.maximum) && rate.minimum > 0.0F;
  if (!positive || !(rate.minimum <= rate.preferred && rate.preferred <= rate.maximum)) {
    throw MetalError(
        "DisplayLink: a frame rate range must be positive and in order (minimum <= preferred <= "
        "maximum)");
  }
}

}  // namespace
}  // namespace serenity::metal

// The link's delegate: calls the client for each update (display_link.h).
// The link holds its delegate weakly; State holds it.
@interface SerenityDisplayLinkDelegate : NSObject <CAMetalDisplayLinkDelegate>
- (instancetype)initWithClient:(serenity::metal::DisplayLink::Client*)client;
@end

@implementation SerenityDisplayLinkDelegate {
  serenity::metal::DisplayLink::Client* _client;  // outlives the link (display_link.h)
}

- (instancetype)initWithClient:(serenity::metal::DisplayLink::Client*)client {
  self = [super init];
  if (self != nil) {
    _client = client;
  }
  return self;
}

- (void)metalDisplayLink:(CAMetalDisplayLink*)link needsUpdate:(CAMetalDisplayLinkUpdate*)update {
  (void)link;
  // The update's drawable is the frame's to render into; Core Animation
  // documents none missing, and a frame is never called with none
  // (DisplayUpdate).
  id<CAMetalDrawable> drawable = update.drawable;
  if (drawable == nil) {
    return;
  }
  // noexcept: nothing unwinds through Core Animation's run loop
  // (display_link.h, Client).
  _client->frame(serenity::metal::DisplayUpdate{
      .drawable = (__bridge CA::MetalDrawable*)drawable,
      .presentation = serenity::frame::Seconds{update.targetPresentationTimestamp},
  });
}

@end

namespace serenity::metal {

struct DisplayLink::State {
  SerenityDisplayLinkDelegate* delegate = nil;
  CAMetalDisplayLink* link = nil;

  State() = default;
  State(const State&) = delete;
  State& operator=(const State&) = delete;
  State(State&&) = delete;
  State& operator=(State&&) = delete;
  // Off the run loop and the layer before either is released; also on a
  // constructor's failure after the link was made.
  ~State() { [link invalidate]; }
};

DisplayLink::DisplayLink(const Presenter& presenter, presentation::FrameRate rate, Client& client)
    : state_(std::make_unique<State>()) {
  // The frames are called on the main run loop, the one SDL runs
  // (display_link.h, Threads).
  if (![NSThread isMainThread]) {
    throw MetalError(
        "DisplayLink: must be made on the main thread, whose run loop calls its frames");
  }
  check(rate);
  state_->delegate = [[SerenityDisplayLinkDelegate alloc] initWithClient:&client];
  state_->link =
      [[CAMetalDisplayLink alloc] initWithMetalLayer:(__bridge CAMetalLayer*)presenter.layer()];
  if (state_->link == nil) {
    throw MetalError("DisplayLink: Core Animation made no display link for the layer");
  }
  state_->link.preferredFrameRateRange =
      CAFrameRateRangeMake(rate.minimum, rate.maximum, rate.preferred);
  state_->link.preferredFrameLatency = frame_latency;
  state_->link.delegate = state_->delegate;
  // Common modes, as Apple's sample adds it: the default mode, which SDL
  // waits for events in, and the modes AppKit runs while a window is
  // resized or a menu tracked.
  [state_->link addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
}

DisplayLink::~DisplayLink() = default;

}  // namespace serenity::metal
