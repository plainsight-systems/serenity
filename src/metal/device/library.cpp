#include "metal/device/library.h"

#include <memory>
#include <string>
#include <type_traits>

#include <dispatch/dispatch.h>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

namespace {

// A dispatch data object, released when it goes out of scope (R.1, E.19).
struct ReleaseDispatchData {
    void operator()(std::remove_pointer_t<dispatch_data_t>* data) const { dispatch_release(data); }
};
using DispatchData = std::unique_ptr<std::remove_pointer_t<dispatch_data_t>, ReleaseDispatchData>;

NS::SharedPtr<MTL::Library> load(MTL::Device* device, std::span<const std::byte> metallib) {
    const auto pool = scoped_pool();
    // DISPATCH_DATA_DESTRUCTOR_DEFAULT copies the bytes into the dispatch
    // object, which is what lets the span die after this call.
    const DispatchData data{
        dispatch_data_create(metallib.data(), metallib.size(), nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT)};
    NS::Error* error = nullptr;
    auto library = NS::TransferPtr(device->newLibrary(data.get(), &error));
    if (!library) {
        throw Error("Metal rejected the library (" + std::to_string(metallib.size()) + " bytes): " +
                    describe(error));
    }
    return library;
}

}  // namespace

Library::Library(const Device& device, std::span<const std::byte> metallib)
    : device_(NS::RetainPtr(device.handle())), library_(load(device_.get(), metallib)) {}

NS::SharedPtr<MTL::ComputePipelineState> Library::compute_pipeline(const char* function) const {
    if (function == nullptr) {
        throw Error("compute_pipeline: the function name is null");
    }
    const auto pool = scoped_pool();

    auto name = NS::TransferPtr(NS::String::alloc()->init(function, NS::UTF8StringEncoding));
    auto fn = NS::TransferPtr(library_->newFunction(name.get()));
    if (!fn) {
        throw Error(std::string{"no function '"} + function + "' in the library");
    }

    NS::Error* error = nullptr;
    auto pipeline = NS::TransferPtr(device_->newComputePipelineState(fn.get(), &error));
    if (!pipeline) {
        throw Error(std::string{"no compute pipeline for '"} + function + "': " + describe(error));
    }
    return pipeline;
}

}  // namespace serenity::metal
