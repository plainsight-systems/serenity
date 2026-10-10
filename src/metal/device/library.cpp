#include "metal/device/library.h"

#include <string>

#include <dispatch/dispatch.h>

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

// Metal's description of `error`, or a fixed text when it gave none. Called
// inside the pool that owns `error`, so the text is copied out before the
// pool drains.
std::string describe(const NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) {
        return "Metal gave no description";
    }
    return error->localizedDescription()->utf8String();
}

NS::SharedPtr<NS::AutoreleasePool> pool() {
    return NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
}

}  // namespace

Library::Library(const Device& device, std::span<const std::byte> metallib)
    : device_(NS::RetainPtr(device.handle())) {
    auto drained = pool();

    // DISPATCH_DATA_DESTRUCTOR_DEFAULT copies the bytes into the dispatch
    // object, which is what lets the span die after this call.
    dispatch_data_t data = dispatch_data_create(metallib.data(), metallib.size(), nullptr,
                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NS::Error* error = nullptr;
    library_ = NS::TransferPtr(device_->newLibrary(data, &error));
    dispatch_release(data);

    if (!library_) {
        throw Error("Metal rejected the library (" + std::to_string(metallib.size()) +
                    " bytes): " + describe(error));
    }
}

NS::SharedPtr<MTL::ComputePipelineState> Library::compute_pipeline(const char* function) const {
    if (function == nullptr) {
        throw Error("compute_pipeline: the function name is null");
    }
    auto drained = pool();

    auto name = NS::TransferPtr(NS::String::alloc()->init(function, NS::UTF8StringEncoding));
    auto fn = NS::TransferPtr(library_->newFunction(name.get()));
    if (!fn) {
        throw Error(std::string("no function '") + function + "' in the library");
    }

    NS::Error* error = nullptr;
    auto pipeline = NS::TransferPtr(device_->newComputePipelineState(fn.get(), &error));
    if (!pipeline) {
        throw Error(std::string("no compute pipeline for '") + function + "': " + describe(error));
    }
    return pipeline;
}

}  // namespace serenity::metal
