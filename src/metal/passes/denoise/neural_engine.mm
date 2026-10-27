// The denoiser's network on the Neural Engine (neural_engine.h), in
// Objective-C++: the one file of the backend that names Core ML, Core Video
// and IOSurface. Compiled with ARC (CMakeLists.txt), so the model, the
// listener and the predictions' arguments, held in State, are released with
// it; the pixel buffers, Core Foundation objects, by State's destructor.

#include "metal/passes/denoise/neural_engine.h"

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <cstddef>
#include <string>
#include <utility>

#include "metal/device/error.h"

namespace serenity::metal {
namespace {

// The model's file and its features' names (training/export_coreml.py).
NSString* const kModelFile = @"unet.mlmodelc";
NSString* const kInput = @"features_in";
NSString* const kOutput = @"features_out";

std::string text(NSString* s) { return s == nil ? std::string() : std::string(s.UTF8String); }
std::string text(NSError* e) { return e == nil ? "no error given" : text(e.localizedDescription); }

// A pixel buffer of half floats `width` x `rows`, over an IOSurface, which
// Metal may use; null if Core Video makes none.
CVPixelBufferRef surface_buffer(std::uint32_t width, std::uint32_t rows) {
  CFMutableDictionaryRef attributes = CFDictionaryCreateMutable(
      kCFAllocatorDefault, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFDictionaryRef surface =
      CFDictionaryCreate(kCFAllocatorDefault, nullptr, nullptr, 0, &kCFTypeDictionaryKeyCallBacks,
                         &kCFTypeDictionaryValueCallBacks);
  CFDictionarySetValue(attributes, kCVPixelBufferIOSurfacePropertiesKey, surface);
  CFDictionarySetValue(attributes, kCVPixelBufferMetalCompatibilityKey, kCFBooleanTrue);
  CVPixelBufferRef made = nullptr;
  const CVReturn status = CVPixelBufferCreate(
      kCFAllocatorDefault, width, rows, kCVPixelFormatType_OneComponent16Half, attributes, &made);
  CFRelease(surface);
  CFRelease(attributes);
  return status == kCVReturnSuccess ? made : nullptr;
}

}  // namespace

// What Core ML and Core Video hold, shared with the listener's blocks, so a
// block still queued keeps what it uses alive (neural_engine.h, R.20).
struct NeuralEngineNetwork::State {
  MLModel* model = nil;
  dispatch_queue_t queue = nil;
  MTLSharedEventListener* listener = nil;
  // The rung prepared: its pixel buffers and a prediction's arguments.
  CVPixelBufferRef input = nullptr;
  CVPixelBufferRef features = nullptr;
  MLDictionaryFeatureProvider* provider = nil;
  MLPredictionOptions* options = nil;

  State() = default;
  State(const State&) = delete;
  State& operator=(const State&) = delete;
  State(State&&) = delete;
  State& operator=(State&&) = delete;

  void release_buffers() noexcept {
    provider = nil;
    options = nil;
    if (input != nullptr) CVPixelBufferRelease(input);
    if (features != nullptr) CVPixelBufferRelease(features);
    input = nullptr;
    features = nullptr;
  }
  ~State() { release_buffers(); }
};

NeuralEngineNetwork::NeuralEngineNetwork(const Device& device, Submission& submission,
                                         const std::filesystem::path& directory,
                                         std::uint32_t inputs, std::uint32_t features)
    : device_(NS::RetainPtr(device.handle())),
      submission_(&submission),
      input_channels_(inputs),
      features_channels_(features),
      state_(std::make_shared<State>()) {
  @autoreleasepool {
    NSString* path = [[NSString stringWithUTF8String:directory.c_str()]
        stringByAppendingPathComponent:kModelFile];
    if (![NSFileManager.defaultManager fileExistsAtPath:path]) {
      throw MetalError("NeuralEngineNetwork: no " + text(path) +
                       " beside the network's weights; make it with training/export_coreml.py");
    }
    MLModelConfiguration* configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    NSError* error = nil;
    MLModel* model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                       configuration:configuration
                                               error:&error];
    if (model == nil) {
      throw MetalError("NeuralEngineNetwork: Core ML loads no model from " + text(path) + ": " +
                       text(error));
    }
    MLFeatureDescription* in = model.modelDescription.inputDescriptionsByName[kInput];
    MLFeatureDescription* out = model.modelDescription.outputDescriptionsByName[kOutput];
    if (in == nil || out == nil || model.modelDescription.inputDescriptionsByName.count != 1 ||
        in.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16) {
      throw MetalError("NeuralEngineNetwork: " + text(path) + " is not one half-float input " +
                       text(kInput) + " and an output " + text(kOutput));
    }
    MLMultiArrayShapeConstraint* shapes = in.multiArrayConstraint.shapeConstraint;
    if (shapes.type != MLMultiArrayShapeConstraintTypeEnumerated) {
      throw MetalError("NeuralEngineNetwork: " + text(path) +
                       "'s shapes are not enumerated: the Neural Engine runs known shapes");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables): fast enumeration's
    for (NSArray<NSNumber*>* shape in shapes.enumeratedShapes) {
      if (shape.count != 4 || shape[0].unsignedIntValue != 1 ||
          shape[1].unsignedIntValue != inputs) {
        throw MetalError("NeuralEngineNetwork: " + text(path) + " has a shape not (1, " +
                         std::to_string(inputs) + ", H, W)");
      }
      rungs_.push_back({shape[3].unsignedIntValue, shape[2].unsignedIntValue});
    }
    state_->model = model;
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
    return;
  }
  bool known = false;
  for (const frame::Extent& r : rungs_) known = known || r == rung;
  if (!known) {
    throw MetalError("NeuralEngineNetwork: " + frame::to_string(rung) +
                     " is not one of the model's shapes");
  }
  input_resident_.reset();
  features_resident_.reset();
  input_.reset();
  features_.reset();
  state_->release_buffers();
  prepared_ = {};
  @autoreleasepool {
    const std::uint32_t inputs = input_channels_;
    State& s = *state_;
    s.input = surface_buffer(rung.width, inputs * rung.height);
    s.features = surface_buffer(rung.width, features_channels_ * rung.height);
    if (s.input == nullptr || s.features == nullptr) {
      throw MetalError("NeuralEngineNetwork: Core Video made no pixel buffer for " +
                       frame::to_string(rung));
    }
    for (CVPixelBufferRef b : {s.input, s.features}) {
      if (CVPixelBufferGetBytesPerRow(b) != row_bytes) {
        throw MetalError("NeuralEngineNetwork: Core Video pads " + frame::to_string(rung) +
                         "'s rows to " + std::to_string(CVPixelBufferGetBytesPerRow(b)) +
                         " bytes, not the kernels' " + std::to_string(row_bytes));
      }
    }
    // Metal's buffers over the surfaces' memory (neural_engine.h).
    const auto over = [&](CVPixelBufferRef b, const char* what) {
      IOSurfaceRef surface = CVPixelBufferGetIOSurface(b);
      if (surface == nullptr) {
        throw MetalError(std::string("NeuralEngineNetwork: the ") + what + " has no IOSurface");
      }
      auto buffer = NS::TransferPtr(device_->newBuffer(IOSurfaceGetBaseAddress(surface),
                                                       IOSurfaceGetAllocSize(surface),
                                                       MTL::ResourceStorageModeShared, nullptr));
      if (!buffer) {
        throw MetalError(std::string("NeuralEngineNetwork: Metal made no buffer over the ") + what);
      }
      return buffer;
    };
    input_ = over(s.input, "input");
    features_ = over(s.features, "features");
    // Core ML's arrays over the same buffers, (1, C, H, W).
    MLMultiArray* in = [[MLMultiArray alloc]
        initWithPixelBuffer:s.input
                      shape:@[ @1, @(inputs), @(rung.height), @(rung.width) ]];
    MLMultiArray* out = [[MLMultiArray alloc]
        initWithPixelBuffer:s.features
                      shape:@[ @1, @(features_channels_), @(rung.height), @(rung.width) ]];
    NSError* error = nil;
    s.provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{kInput : in}
                                                                   error:&error];
    s.options = [[MLPredictionOptions alloc] init];
    s.options.outputBackings = @{kOutput : out};
    if (s.provider == nil) {
      throw MetalError("NeuralEngineNetwork: Core ML takes no input over the pixel buffer: " +
                       text(error));
    }
    // The rung's first prediction, here, not in a frame (neural_engine.h).
    id<MLFeatureProvider> result = [s.model predictionFromFeatures:s.provider
                                                           options:s.options
                                                             error:&error];
    if (result == nil) {
      throw MetalError("NeuralEngineNetwork: Core ML's first prediction at " +
                       frame::to_string(rung) + " failed: " + text(error));
    }
    if ([result featureValueForName:kOutput].multiArrayValue.pixelBuffer != s.features) {
      throw MetalError(
          "NeuralEngineNetwork: Core ML answered in its own memory, not the features' buffer");
    }
  }
  input_resident_ = submission_->keep_resident(input_.get());
  features_resident_ = submission_->keep_resident(features_.get());
  prepared_ = rung;
}

MTL::GPUAddress NeuralEngineNetwork::input() const noexcept {
  return input_ ? input_->gpuAddress() : 0;
}
MTL::GPUAddress NeuralEngineNetwork::features() const noexcept {
  return features_ ? features_->gpuAddress() : 0;
}

void NeuralEngineNetwork::run(const HandOff& hand_off) const {
  if (!input_ || hand_off.ready == nullptr || hand_off.done == nullptr || !hand_off.report) {
    throw MetalError("NeuralEngineNetwork::run: not prepared, or no hand-off");
  }
  // What the block uses, held by it (neural_engine.h, R.20).
  std::shared_ptr<State> state = state_;
  std::shared_ptr<HandOffReport> report = hand_off.report;
  const std::uint64_t value = hand_off.value;
  const std::uint64_t sequence = value - 1;
  id<MTLSharedEvent> ready = (__bridge id<MTLSharedEvent>)hand_off.ready;
  id<MTLSharedEvent> done = (__bridge id<MTLSharedEvent>)hand_off.done;
  MLDictionaryFeatureProvider* provider = state->provider;
  MLPredictionOptions* options = state->options;
  CVPixelBufferRef features = state->features;
  [ready
      notifyListener:state->listener
             atValue:value
               block:^(id<MTLSharedEvent>, uint64_t) {
                 @autoreleasepool {
                   NSError* error = nil;
                   id<MLFeatureProvider> result = [state->model predictionFromFeatures:provider
                                                                               options:options
                                                                                 error:&error];
                   if (result == nil) {
                     report->failed(sequence,
                                    "Core ML's prediction on the Neural Engine: " + text(error));
                   } else if ([result featureValueForName:kOutput].multiArrayValue.pixelBuffer !=
                              features) {
                     report->failed(sequence,
                                    "Core ML answered in its own memory, not the features' buffer");
                   } else {
                     report->succeeded(sequence);
                   }
                   done.signaledValue = value;
                 }
               }];
}

}  // namespace serenity::metal
