// The denoiser's network on the Neural Engine (neural_engine.h), in
// Objective-C++: the one file of the backend that names Core ML, Core Video
// and IOSurface. Compiled with ARC (CMakeLists.txt), so the model, the
// listener and the predictions' arguments, held in State, are released with
// it; the pixel buffers, Core Foundation objects, by their owners'
// deleters (PixelBuffer).

#include "metal/passes/denoise/neural_engine.h"

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "metal/device/error.h"

namespace serenity::metal {
namespace {

// The model's file and its features' names (training/export_coreml.py).
NSString* const kModelFile = @"unet.mlmodelc";
NSString* const kInput = @"features_in";
NSString* const kOutput = @"features_out";

// The model's shapes, (N, C, H, W): their rank and each axis's place.
constexpr NSUInteger kRank = 4;
constexpr NSUInteger kBatch = 0;
constexpr NSUInteger kChannels = 1;
constexpr NSUInteger kHeight = 2;
constexpr NSUInteger kWidth = 3;

std::string text(NSString* s) {
  const char* const utf8 = s.UTF8String;  // nullable, and nil for nil
  return utf8 == nullptr ? std::string() : std::string(utf8);
}
std::string text(NSError* e) { return e == nil ? "no error given" : text(e.localizedDescription); }

// A pixel buffer's owner: Core Video's release, once (R.3, E.6).
struct PixelBufferRelease {
  void operator()(CVPixelBufferRef buffer) const noexcept { CVPixelBufferRelease(buffer); }
};
using PixelBuffer = std::unique_ptr<std::remove_pointer_t<CVPixelBufferRef>, PixelBufferRelease>;

// A pixel buffer of half floats `width` x `rows`, over an IOSurface, which
// Metal may use; null if Core Video makes none.
PixelBuffer surface_buffer(std::uint32_t width, std::uint32_t rows) {
  NSDictionary* const attributes = @{
    (__bridge NSString*)kCVPixelBufferIOSurfacePropertiesKey : @{},
    (__bridge NSString*)kCVPixelBufferMetalCompatibilityKey : @YES,
  };
  CVPixelBufferRef made = nullptr;
  const CVReturn status =
      CVPixelBufferCreate(kCFAllocatorDefault, width, rows, kCVPixelFormatType_OneComponent16Half,
                          (__bridge CFDictionaryRef)attributes, &made);
  PixelBuffer owned(made);
  if (status != kCVReturnSuccess) {
    owned.reset();
  }
  return owned;
}

// A buffer of Metal's over `buffer`'s memory, the same pages
// (neural_engine.h).
NS::SharedPtr<MTL::Buffer> metal_over(MTL::Device* device, CVPixelBufferRef buffer,
                                      std::string_view what) {
  IOSurfaceRef const surface = CVPixelBufferGetIOSurface(buffer);
  if (surface == nullptr) {
    throw MetalError("NeuralEngineNetwork: the " + std::string(what) + " has no IOSurface");
  }
  auto made = NS::TransferPtr(device->newBuffer(IOSurfaceGetBaseAddress(surface),
                                                IOSurfaceGetAllocSize(surface),
                                                MTL::ResourceStorageModeShared, nullptr));
  if (!made) {
    throw MetalError("NeuralEngineNetwork: Metal made no buffer over the " + std::string(what));
  }
  return made;
}

// The model at `path`, loaded for the CPU and the Neural Engine: one input
// and one output, half floats, of the names above (neural_engine.h).
MLModel* load_model(NSString* path) {
  if (![NSFileManager.defaultManager fileExistsAtPath:path]) {
    throw MetalError("NeuralEngineNetwork: no " + text(path) +
                     " beside the network's weights; make it with training/export_coreml.py");
  }
  MLModelConfiguration* const configuration = [[MLModelConfiguration alloc] init];
  configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
  NSError* error = nil;
  MLModel* const model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                           configuration:configuration
                                                   error:&error];
  if (model == nil) {
    throw MetalError("NeuralEngineNetwork: Core ML loads no model from " + text(path) + ": " +
                     text(error));
  }
  MLModelDescription* const description = model.modelDescription;
  MLFeatureDescription* const in = description.inputDescriptionsByName[kInput];
  MLFeatureDescription* const out = description.outputDescriptionsByName[kOutput];
  if (in == nil || out == nil || description.inputDescriptionsByName.count != 1 ||
      description.outputDescriptionsByName.count != 1 ||
      in.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16 ||
      out.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16) {
    throw MetalError("NeuralEngineNetwork: " + text(path) + " is not one half-float input " +
                     text(kInput) + " and one half-float output " + text(kOutput));
  }
  return model;
}

// The model's shapes as width x height: its input's, enumerated, each
// (1, `channels`, H, W).
std::vector<frame::Extent> read_rungs(MLModel* model, NSString* path, std::uint32_t channels) {
  MLMultiArrayShapeConstraint* const shapes =
      model.modelDescription.inputDescriptionsByName[kInput].multiArrayConstraint.shapeConstraint;
  if (shapes.type != MLMultiArrayShapeConstraintTypeEnumerated) {
    throw MetalError("NeuralEngineNetwork: " + text(path) +
                     "'s shapes are not enumerated: the Neural Engine runs known shapes");
  }
  std::vector<frame::Extent> rungs;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables): fast enumeration's
  for (NSArray<NSNumber*>* shape in shapes.enumeratedShapes) {
    if (shape.count != kRank || shape[kBatch].unsignedIntValue != 1 ||
        shape[kChannels].unsignedIntValue != channels) {
      throw MetalError("NeuralEngineNetwork: " + text(path) + " has a shape not (1, " +
                       std::to_string(channels) + ", H, W)");
    }
    rungs.push_back({shape[kWidth].unsignedIntValue, shape[kHeight].unsignedIntValue});
  }
  return rungs;
}

// A rung's memory (neural_engine.h, "The memory"), made whole before any of
// it is kept (prepare()).
struct RungMemory {
  PixelBuffer input;
  PixelBuffer features;
  NS::SharedPtr<MTL::Buffer> input_metal;
  NS::SharedPtr<MTL::Buffer> features_metal;
  MLDictionaryFeatureProvider* provider = nil;
  MLPredictionOptions* options = nil;
};

// `rung`'s pixel buffers of `input_channels` and `feature_channels` planes,
// their rows `row_bytes` apart; Metal's buffers over them; Core ML's
// arrays over them, (1, C, H, W), as a prediction's input and output
// backing.
RungMemory make_rung(MTL::Device* device, frame::Extent rung, std::size_t row_bytes,
                     std::uint32_t input_channels, std::uint32_t feature_channels) {
  RungMemory memory;
  memory.input = surface_buffer(rung.width, input_channels * rung.height);
  memory.features = surface_buffer(rung.width, feature_channels * rung.height);
  if (!memory.input || !memory.features) {
    throw MetalError("NeuralEngineNetwork: Core Video made no pixel buffer for " +
                     frame::to_string(rung));
  }
  for (CVPixelBufferRef const buffer : {memory.input.get(), memory.features.get()}) {
    if (CVPixelBufferGetBytesPerRow(buffer) != row_bytes) {
      throw MetalError("NeuralEngineNetwork: Core Video pads " + frame::to_string(rung) +
                       "'s rows to " + std::to_string(CVPixelBufferGetBytesPerRow(buffer)) +
                       " bytes, not the kernels' " + std::to_string(row_bytes));
    }
  }
  memory.input_metal = metal_over(device, memory.input.get(), "input");
  memory.features_metal = metal_over(device, memory.features.get(), "features");
  MLMultiArray* const in = [[MLMultiArray alloc]
      initWithPixelBuffer:memory.input.get()
                    shape:@[ @1, @(input_channels), @(rung.height), @(rung.width) ]];
  MLMultiArray* const out = [[MLMultiArray alloc]
      initWithPixelBuffer:memory.features.get()
                    shape:@[ @1, @(feature_channels), @(rung.height), @(rung.width) ]];
  NSError* error = nil;
  memory.provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{kInput : in}
                                                                      error:&error];
  if (memory.provider == nil) {
    throw MetalError("NeuralEngineNetwork: Core ML takes no input over the pixel buffer: " +
                     text(error));
  }
  memory.options = [[MLPredictionOptions alloc] init];
  memory.options.outputBackings = @{kOutput : out};
  return memory;
}

// The rung's first prediction, at prepare(), not in a frame
// (neural_engine.h), and Core ML answering in our memory.
void first_prediction(MLModel* model, const RungMemory& memory, frame::Extent rung) {
  NSError* error = nil;
  const id<MLFeatureProvider> result = [model predictionFromFeatures:memory.provider
                                                             options:memory.options
                                                               error:&error];
  if (result == nil) {
    throw MetalError("NeuralEngineNetwork: Core ML's first prediction at " +
                     frame::to_string(rung) + " failed: " + text(error));
  }
  if ([result featureValueForName:kOutput].multiArrayValue.pixelBuffer != memory.features.get()) {
    throw MetalError(
        "NeuralEngineNetwork: Core ML answered in its own memory, not the features' buffer");
  }
}

// The worker (neural_engine.h, "The frame"), on the listener's queue: the
// prediction, its report, then `done` set to `value`. noexcept: it runs
// inside an Objective-C block on a dispatch queue, which an exception must
// not unwind through, so one (a failed allocation while building the
// message, or Core ML's own) ends the program at once instead (E.12), as
// Submission's feedback handler does.
void predict(MLModel* model, MLDictionaryFeatureProvider* provider, MLPredictionOptions* options,
             CVPixelBufferRef features, HandOffReport& report, id<MTLSharedEvent> done,
             std::uint64_t value) noexcept {
  @autoreleasepool {
    NSError* error = nil;
    const id<MLFeatureProvider> result = [model predictionFromFeatures:provider
                                                               options:options
                                                                 error:&error];
    if (result == nil) {
      report.report_failure("Core ML's prediction on the Neural Engine: " + text(error));
    } else if ([result featureValueForName:kOutput].multiArrayValue.pixelBuffer != features) {
      report.report_failure("Core ML answered in its own memory, not the features' buffer");
    } else {
      report.report_success();
    }
    done.signaledValue = value;
  }
}

}  // namespace

// What Core ML and Core Video hold, shared with the listener's blocks, so a
// block still queued keeps what it uses alive (neural_engine.h, R.20).
struct NeuralEngineNetwork::State {
  MLModel* model = nil;
  dispatch_queue_t queue = nil;
  MTLSharedEventListener* listener = nil;
  // The rung prepared: its pixel buffers and a prediction's arguments.
  PixelBuffer input;
  PixelBuffer features;
  MLDictionaryFeatureProvider* provider = nil;
  MLPredictionOptions* options = nil;

  State() = default;
  State(const State&) = delete;
  State& operator=(const State&) = delete;
  State(State&&) = delete;
  State& operator=(State&&) = delete;
  ~State() = default;

  void release_buffers() noexcept {
    provider = nil;
    options = nil;
    input.reset();
    features.reset();
  }
};

NeuralEngineNetwork::NeuralEngineNetwork(const Device& device, Submission& submission,
                                         const std::filesystem::path& directory,
                                         std::uint32_t input_channels,
                                         std::uint32_t feature_channels)
    : device_(NS::RetainPtr(device.handle())),
      submission_(&submission),
      input_channels_(input_channels),
      features_channels_(feature_channels),
      state_(std::make_shared<State>()) {
  @autoreleasepool {
    NSString* const path = [[NSString stringWithUTF8String:directory.c_str()]
        stringByAppendingPathComponent:kModelFile];
    state_->model = load_model(path);
    rungs_ = read_rungs(state_->model, path, input_channels);
    state_->queue = dispatch_queue_create(
        "serenity.neural-engine", dispatch_queue_attr_make_with_qos_class(
                                      DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
    state_->listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:state_->queue];
  }
}

NeuralEngineNetwork::~NeuralEngineNetwork() {
  // The Residents wait for the GPU, so for every hand-off's `done`, before
  // the buffers go; a block still queued holds State.
  input_resident_.reset();
  features_resident_.reset();
}

const std::vector<frame::Extent>& NeuralEngineNetwork::rungs() const noexcept { return rungs_; }

void NeuralEngineNetwork::prepare(frame::Extent rung, std::size_t row_bytes) {
  if (rung == prepared_) {
    if (row_bytes != row_bytes_) {
      throw MetalError("NeuralEngineNetwork: " + frame::to_string(rung) + "'s rows are " +
                       std::to_string(row_bytes_) + " bytes apart, not " +
                       std::to_string(row_bytes));
    }
    return;
  }
  if (std::ranges::find(rungs_, rung) == rungs_.end()) {
    throw MetalError("NeuralEngineNetwork: " + frame::to_string(rung) +
                     " is not one of the model's shapes");
  }
  // The last rung's memory goes first, so two are never held at once; the
  // Residents wait for the GPU.
  input_resident_.reset();
  features_resident_.reset();
  input_.reset();
  features_.reset();
  state_->release_buffers();
  prepared_ = {};
  row_bytes_ = 0;
  @autoreleasepool {
    RungMemory memory =
        make_rung(device_.get(), rung, row_bytes, input_channels_, features_channels_);
    first_prediction(state_->model, memory, rung);
    // Kept only now, every step done (E.4).
    State& state = *state_;
    state.input = std::move(memory.input);
    state.features = std::move(memory.features);
    state.provider = memory.provider;
    state.options = memory.options;
    input_ = std::move(memory.input_metal);
    features_ = std::move(memory.features_metal);
  }
  input_resident_ = submission_->keep_resident(input_.get());
  features_resident_ = submission_->keep_resident(features_.get());
  prepared_ = rung;
  row_bytes_ = row_bytes;
}

MTL::GPUAddress NeuralEngineNetwork::input() const noexcept {
  return prepared_ == frame::Extent{} ? 0 : input_->gpuAddress();
}
MTL::GPUAddress NeuralEngineNetwork::features() const noexcept {
  return prepared_ == frame::Extent{} ? 0 : features_->gpuAddress();
}

void NeuralEngineNetwork::run(const HandOff& hand_off) const {
  if (prepared_ == frame::Extent{} || hand_off.ready == nullptr || hand_off.done == nullptr ||
      !hand_off.report) {
    throw MetalError("NeuralEngineNetwork::run: not prepared, or no hand-off");
  }
  // What the block uses, held by it (neural_engine.h, R.20).
  const std::shared_ptr<State> state = state_;
  const std::shared_ptr<HandOffReport> report = hand_off.report;
  const std::uint64_t value = hand_off.value;
  id<MTLSharedEvent> const ready = (__bridge id<MTLSharedEvent>)hand_off.ready;
  id<MTLSharedEvent> const done = (__bridge id<MTLSharedEvent>)hand_off.done;
  MLDictionaryFeatureProvider* const provider = state->provider;
  MLPredictionOptions* const options = state->options;
  CVPixelBufferRef const features_buffer = state->features.get();
  [ready notifyListener:state->listener
                atValue:value
                  block:^(id<MTLSharedEvent>, uint64_t) {
                    predict(state->model, provider, options, features_buffer, *report, done, value);
                  }];
}

}  // namespace serenity::metal
